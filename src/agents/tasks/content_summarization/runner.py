"""Task runner for content summarization with rate-limit-aware batching."""

import logging
from typing import Callable, Dict, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.core.utils import parse_numbered_response
from src.agents.core.utils import TOKEN_SAFETY_MULTIPLIER
from src.agents.core.rate_limiter import RateLimiter
from src.agents.tasks.content_summarization.prompts import build_single_prompt, build_numbered_prompt

_PROMPT_OVERHEAD_TOKENS = 60
_OUTPUT_TOKENS_PER_SUMMARY = 100
_MAX_PAGES_PER_BATCH = 10

log = logging.getLogger(__name__)


class SummarizationTask:
    """Handles rate-limit-aware batch summarization."""

    def __init__(
        self,
        chat_model: ChatGoogleGenerativeAI,
        rate_limiter: RateLimiter,
        call_with_retry: Callable,
    ) -> None:
        self.chat_model = chat_model
        self.rate_limiter = rate_limiter
        self._call_with_retry = call_with_retry

    def run(self, pages: List[Tuple[str, int]]) -> List[str]:
        """Summarize page texts respecting RPM / TPM limits."""
        non_empty = self._filter_non_empty_pages(pages)

        if not non_empty:
            return [""] * len(pages)

        batches = self._build_batches(non_empty)
        log.info("Summarising %d non-empty page(s) in %d batch(es).", len(non_empty), len(batches))

        summary_map = self._process_batches(batches)

        produced = sum(1 for v in summary_map.values() if v)
        log.info(
            "Summary generation finished - %d/%d pages summarised (%d empty).",
            produced, len(pages), len(pages) - produced,
        )

        return [summary_map.get(i, "") for i in range(len(pages))]

    @staticmethod
    def _filter_non_empty_pages(
        pages: List[Tuple[str, int]],
    ) -> List[Tuple[int, str, int]]:
        return [
            (i, text, tc)
            for i, (text, tc) in enumerate(pages)
            if text.strip()
        ]

    def _build_batches(
        self,
        items: List[Tuple[int, str, int]],
    ) -> List[List[Tuple[int, str, int]]]:
        max_input = int(self.rate_limiter.rate_limits.tpm / TOKEN_SAFETY_MULTIPLIER) // 3
        batches: List[List[Tuple[int, str, int]]] = []
        current: List[Tuple[int, str, int]] = []
        current_tokens = 0

        for item in items:
            _, _, tc = item
            if current and (current_tokens + tc > max_input or len(current) >= _MAX_PAGES_PER_BATCH):
                batches.append(current)
                current = []
                current_tokens = 0
            current.append(item)
            current_tokens += tc

        if current:
            batches.append(current)

        return batches

    def _process_batches(
        self, batches: List[List[Tuple[int, str, int]]]
    ) -> Dict[int, str]:
        summary_map: Dict[int, str] = {}

        for batch_no, batch in enumerate(batches, 1):
            indices = [i for i, _, _ in batch]
            texts = [t for _, t, _ in batch]
            estimated = self._estimate_tokens(batch, texts)

            self.rate_limiter.wait_if_needed(estimated)
            log.debug(
                "Batch %d/%d - %d page(s), ~%d est. tokens (pages %s).",
                batch_no, len(batches), len(texts), estimated,
                ", ".join(str(idx + 1) for idx in indices),
            )

            summaries = self._generate_batch_safe(batch_no, len(batches), texts, indices)
            self.rate_limiter.record(estimated)

            for idx, summary in zip(indices, summaries):
                summary_map[idx] = summary

            log.debug("Batch %d/%d completed.", batch_no, len(batches))

        return summary_map

    @staticmethod
    def _estimate_tokens(batch: List[Tuple[int, str, int]], texts: List[str]) -> int:
        input_tokens = sum(tc for _, _, tc in batch)
        return int(
            (_PROMPT_OVERHEAD_TOKENS + input_tokens + _OUTPUT_TOKENS_PER_SUMMARY * len(texts))
            * TOKEN_SAFETY_MULTIPLIER
        )

    def _generate_batch_safe(
        self,
        batch_no: int,
        total_batches: int,
        texts: List[str],
        indices: List[int],
    ) -> List[str]:
        try:
            return self._call_with_retry(
                self._summarize_batch,
                texts,
                on_retry=self.rate_limiter.force_reset,
            )
        except Exception:
            log.exception(
                "Batch %d/%d failed - pages %s will have empty summaries.",
                batch_no, total_batches,
                ", ".join(str(idx + 1) for idx in indices),
            )
            return [""] * len(texts)

    def _summarize_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        prompt = build_single_prompt(texts[0]) if len(texts) == 1 else build_numbered_prompt(texts)
        raw = self.chat_model.invoke(prompt).content

        if len(texts) == 1:
            return [raw]
        return parse_numbered_response(raw, len(texts))
