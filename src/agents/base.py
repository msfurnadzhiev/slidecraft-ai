"""Common base for LLM-based content processors with rate-limit-aware batching."""

import os
import re
import logging
import time
from typing import Any, Callable, List, TypeVar

from google.genai import errors as genai_errors

from src.agents.client import GeminiClient
from src.utils.singleton import SingletonMeta

T = TypeVar("T")

WINDOW_SECONDS = 60.0
DEFAULT_RPM = 30
DEFAULT_TPM = 15_000

# tiktoken (cl100k_base) undercounts compared to Gemini's tokenizer.
# Empirically ~1.45x; we use 1.5x for safety.
TOKEN_SAFETY_MULTIPLIER = 1.5

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
        now = time.time()
        if now - self._window_start >= WINDOW_SECONDS:
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

        sleep_sec = WINDOW_SECONDS - (time.time() - self._window_start) + 1
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


class BaseLLMProcessor(metaclass=SingletonMeta):
    """Shared infrastructure for LLM content processors: client, rate limiting, and retry."""

    def __init__(self) -> None:
        self.genai = GeminiClient.get_instance()
        self._rate = _RateWindow(
            rpm=int(os.getenv("LLM_RPM_LIMIT", DEFAULT_RPM)),
            tpm=int(os.getenv("LLM_TPM_LIMIT", DEFAULT_TPM)),
        )

    def _call_with_retry(self, fn: Callable[..., T], *args: Any) -> T:
        """Call *fn* with automatic retry on HTTP 429."""
        for attempt in range(1, _429_MAX_RETRIES + 1):
            try:
                return fn(*args)
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

    @staticmethod
    def _parse_numbered_response(raw: str, expected_count: int) -> List[str]:
        """Parse '[N] …' lines from a numbered LLM response."""
        pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)
        matches = {
            int(m.group(1)): m.group(2).strip()
            for m in pattern.finditer(raw)
        }
        result = [matches.get(idx, "") for idx in range(1, expected_count + 1)]
        missing = [idx for idx in range(1, expected_count + 1) if not matches.get(idx)]
        if missing:
            log.warning(
                "LLM response missing entries for item(s) %s out of %d expected.",
                missing, expected_count,
            )
        return result
