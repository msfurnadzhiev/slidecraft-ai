"""LLM-based text summarization with rate-limit-aware batching."""

import os
import re
import logging
import time
from typing import List, Tuple

from google.genai import errors as genai_errors

from src.agents.client import GeminiClient
from src.utils.singleton import SingletonMeta
from src.utils.profiling import trace_runtime

_PROMPT_OVERHEAD_TOKENS = 60
_OUTPUT_TOKENS_PER_SUMMARY = 100
_MAX_PAGES_PER_BATCH = 10
_WINDOW_SECONDS = 60.0

_DEFAULT_RPM = 30
_DEFAULT_TPM = 15_000

# tiktoken (cl100k_base) undercounts compared to Gemini's tokenizer.
# Empirically ~1.45x; we use 1.5x for safety.
_TOKEN_SAFETY_MULTIPLIER = 1.5

_429_DEFAULT_WAIT = 60.0
_429_MAX_RETRIES = 5

log = logging.getLogger(__name__)

class _RateWindow:
    """Fixed-window rate limiter for RPM and TPM."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self._window_start: float = 0.0
        self._req_count: int = 0
        self._token_count: int = 0

    def _reset_if_expired(self) -> None:
        """Reset the rate limiter if the window has expired."""
        now = time.time()
        if now - self._window_start >= _WINDOW_SECONDS:
            self._window_start = now
            self._req_count = 0
            self._token_count = 0

    def wait_if_needed(self, estimated_tokens: int) -> None:
        """Block until the next request fits within both RPM and TPM limits."""
        self._reset_if_expired()

        if (
            self._req_count < self.rpm
            and self._token_count + estimated_tokens <= self.tpm
        ):
            return

        sleep_sec = _WINDOW_SECONDS - (time.time() - self._window_start) + 1
        log.debug(
            "Rate limit reached (requests=%d/%d, tokens=%d/%d). Waiting %.0fs for window reset.",
            self._req_count, self.rpm, self._token_count, self.tpm, sleep_sec,
        )
        time.sleep(max(sleep_sec, 1))
        self._window_start = time.time()
        self._req_count = 0
        self._token_count = 0

    def force_reset(self) -> None:
        """Force-expire the current window (used after a 429 wait)."""
        self._window_start = time.time()
        self._req_count = 0
        self._token_count = 0

    def record(self, tokens_used: int) -> None:
        self._reset_if_expired()
        self._req_count += 1
        self._token_count += tokens_used


class Summarizer(metaclass=SingletonMeta):
    """Generate concise chunk summaries with rate-limit-aware batching."""

    def __init__(self):
        self.genai = GeminiClient.get_instance()
        self._rate = _RateWindow(
            rpm=int(os.getenv("LLM_RPM_LIMIT", _DEFAULT_RPM)),
            tpm=int(os.getenv("LLM_TPM_LIMIT", _DEFAULT_TPM)),
        )
        log.info("Summarizer initialised (RPM=%d, TPM=%d).", self._rate.rpm, self._rate.tpm)

    @trace_runtime
    def summarize_pages(
        self, pages: List[Tuple[str, int]]
    ) -> List[str]:
        """Summarize page texts respecting RPM / TPM limits.

        Args:
            pages: List of (text, token_count) pairs, one per page.

        Returns:
            Summaries in the same order (empty string for blank pages).
        """
        non_empty = [
            (i, text, tc)
            for i, (text, tc) in enumerate(pages)
            if text.strip()
        ]

        if not non_empty:
            return [""] * len(pages)

        batches = self._build_batches(non_empty)
        log.info(
            "Summarising %d non-empty page(s) in %d batch(es).",
            len(non_empty), len(batches),
        )

        summary_map: dict[int, str] = {}

        for batch_no, batch in enumerate(batches, 1):
            indices = [i for i, _, _ in batch]
            texts = [t for _, t, _ in batch]
            input_tokens = sum(tc for _, _, tc in batch)
            estimated = int(
                (
                    _PROMPT_OVERHEAD_TOKENS
                    + input_tokens
                    + _OUTPUT_TOKENS_PER_SUMMARY * len(texts)
                )
                * _TOKEN_SAFETY_MULTIPLIER
            )

            self._rate.wait_if_needed(estimated)

            log.info(
                "Batch %d/%d – %d page(s), ~%d est. tokens (pages %s).",
                batch_no, len(batches), len(texts), estimated,
                ", ".join(str(idx + 1) for idx in indices),
            )

            try:
                summaries = self._call_with_retry(texts)
            except Exception:
                log.exception(
                    "Batch %d/%d failed – pages %s will have empty summaries.",
                    batch_no, len(batches),
                    ", ".join(str(idx + 1) for idx in indices),
                )
                summaries = [""] * len(texts)

            self._rate.record(estimated)

            for idx, summary in zip(indices, summaries):
                summary_map[idx] = summary

            log.info("Batch %d/%d completed.", batch_no, len(batches))

        produced = sum(1 for v in summary_map.values() if v)
        log.info(
            "Summary generation finished – %d/%d pages summarised (%d empty).",
            produced, len(pages), len(pages) - produced,
        )
        return [summary_map.get(i, "") for i in range(len(pages))]

    def _call_with_retry(self, texts: List[str]) -> List[str]:
        """Call _summarize_batch with automatic retry on 429."""
        for attempt in range(1, _429_MAX_RETRIES + 1):
            try:
                return self._summarize_batch(texts)
            except genai_errors.ClientError as exc:
                if exc.code != 429 or attempt == _429_MAX_RETRIES:
                    raise
                wait = self._parse_retry_delay(str(exc))
                log.warning(
                    "429 from API – waiting %.0fs before retry (attempt %d/%d).",
                    wait, attempt, _429_MAX_RETRIES,
                )
                time.sleep(wait)
                self._rate.force_reset()

        raise RuntimeError("Unreachable")

    @staticmethod
    def _parse_retry_delay(error_text: str) -> float:
        """Extract 'retry in Xs' from the Gemini 429 error body."""
        match = re.search(r"retry in ([\d.]+)s", error_text, re.IGNORECASE)
        if match:
            return float(match.group(1)) + 2.0
        return _429_DEFAULT_WAIT

    def _build_batches(
        self, items: List[Tuple[int, str, int]],
    ) -> List[List[Tuple[int, str, int]]]:
        """Greedily pack items into batches bounded by token budget and page count.

        Token budget per batch is TPM // 3 (after safety multiplier)
        to leave headroom for multiple batches within a single window.
        """
        max_input = int(self._rate.tpm / _TOKEN_SAFETY_MULTIPLIER) // 3

        batches: List[List[Tuple[int, str, int]]] = []
        current: List[Tuple[int, str, int]] = []
        current_tokens = 0

        for item in items:
            _, _, tc = item
            if current and (
                current_tokens + tc > max_input
                or len(current) >= _MAX_PAGES_PER_BATCH
            ):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += tc

        if current:
            batches.append(current)

        return batches

    def _summarize_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        if len(texts) == 1:
            return [self._summarize_single(texts[0])]

        numbered = "\n\n".join(
            f"[{idx + 1}]\n{text}" for idx, text in enumerate(texts)
        )
        prompt = (
            f"Summarize each of the following {len(texts)} numbered texts "
            "in at most 2 concise sentences each. "
            "Preserve key facts, numbers, and terminology. "
            "Return plain text only, prefixed with the matching number "
            "in the format [N] summary.\n\n"
            f"{numbered}"
        )

        raw = self.genai.generate_content(prompt)
        return self._parse_numbered_response(raw, len(texts))

    def _summarize_single(self, text: str) -> str:
        if not text.strip():
            return ""

        prompt = (
            "Summarize the following text in at most 2 concise sentences. "
            "Preserve key facts, numbers, and terminology. "
            "Return plain text only.\n\n"
            f"Text:\n{text}"
        )
        return self.genai.generate_content(prompt)

    @staticmethod
    def _parse_numbered_response(raw: str, expected_count: int) -> List[str]:
        """Parse '[N] summary' lines from the LLM response."""
        pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)
        matches = {
            int(m.group(1)): m.group(2).strip()
            for m in pattern.finditer(raw)
        }
        result = [matches.get(idx, "") for idx in range(1, expected_count + 1)]
        missing = [idx for idx in range(1, expected_count + 1) if not matches.get(idx)]
        if missing:
            log.warning(
                "LLM response missing summaries for item(s) %s out of %d expected.",
                missing, expected_count,
            )
        return result
