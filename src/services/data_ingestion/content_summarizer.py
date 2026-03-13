"""LLM-based text summarization with rate-limit-aware batching."""

import logging
from typing import List, Tuple, Dict

from src.agents.base import BaseLLMProcessor, TOKEN_SAFETY_MULTIPLIER
from src.utils.profiling import trace_runtime

_PROMPT_OVERHEAD_TOKENS = 60
_OUTPUT_TOKENS_PER_SUMMARY = 100
_MAX_PAGES_PER_BATCH = 10

log = logging.getLogger(__name__)


class ContentSummarizer(BaseLLMProcessor):
    """Generate concise chunk summaries with rate-limit-aware batching."""

    def __init__(self) -> None:
        super().__init__()
        log.info(
            "ContentSummarizer initialised (RPM=%d, TPM=%d).",
            self._rate.rpm, self._rate.tpm,
        )

    @trace_runtime
    def summarize_pages(self, pages: List[Tuple[str, int]]) -> List[str]:
        """Summarize page texts respecting RPM / TPM limits.

        Args:
            pages: List of (text, token_count) pairs, one per page.

        Returns:
            Summaries in the same order (empty string for blank pages).
        """
        
        non_empty = self._filter_non_empty_pages(pages)
        
        # If there are no non-empty pages, return an empty list
        if not non_empty:
            return [""] * len(pages)

        # Build batches of pages respecting token and page count limits
        batches = self._build_batches(non_empty)

        log.info(
            "Summarising %d non-empty page(s) in %d batch(es).",
            len(non_empty), len(batches),
        )

        summary_map = self._process_batches(batches)

        produced = sum(1 for v in summary_map.values() if v)
        
        log.info(
            "Summary generation finished - %d/%d pages summarised (%d empty).",
            produced, len(pages), len(pages) - produced,
        )

        return [summary_map.get(i, "") for i in range(len(pages))]

    def _filter_non_empty_pages(
        self, pages: List[Tuple[str, int]]
    ) -> List[Tuple[int, str, int]]:
        """Filter out empty pages and keep track of their original indices."""
        return [
            (i, text, tc)
            for i, (text, tc) in enumerate(pages)
            if text.strip()
        ]

    def _process_batches(
        self, batches: List[List[Tuple[int, str, int]]]
    ) -> Dict[int, str]:
        """Iterate over batches, respecting rate limits, and collect summaries."""
        summary_map: Dict[int, str] = {}

        for batch_no, batch in enumerate(batches, 1):
            indices = [i for i, _, _ in batch]
            texts = [t for _, t, _ in batch]
            estimated = self._estimate_tokens(batch, texts)

            self._rate.wait_if_needed(estimated)

            log.debug(
                "Batch %d/%d - %d page(s), ~%d est. tokens (pages %s).",
                batch_no, len(batches), len(texts), estimated,
                ", ".join(str(idx + 1) for idx in indices),
            )

            summaries = self._generate_summaries_safe(batch_no, len(batches), texts, indices)
            self._rate.record(estimated)

            for idx, summary in zip(indices, summaries):
                summary_map[idx] = summary

            log.debug("Batch %d/%d completed.", batch_no, len(batches))

        return summary_map

    def _estimate_tokens(self, batch, texts: List[str]) -> int:
        """Estimate token usage for a batch with safety multiplier."""
        input_tokens = sum(tc for _, _, tc in batch)
        return int(
            (_PROMPT_OVERHEAD_TOKENS + input_tokens + _OUTPUT_TOKENS_PER_SUMMARY * len(texts))
            * TOKEN_SAFETY_MULTIPLIER
        )

    def _generate_summaries_safe(
        self, batch_no: int, total_batches: int, texts: List[str], indices: List[int]
    ) -> List[str]:
        """Call the LLM for a batch, returning empty summaries on failure."""
        try:
            return self._call_with_retry(self._summarize_batch, texts)
        except Exception:
            log.exception(
                "Batch %d/%d failed – pages %s will have empty summaries.",
                batch_no, total_batches,
                ", ".join(str(idx + 1) for idx in indices),
            )
            return [""] * len(texts)


    def _build_batches(
        self, items: List[Tuple[int, str, int]],
    ) -> List[List[Tuple[int, str, int]]]:
        """Greedily pack items into batches bounded by token budget and page count."""
        max_input = int(self._rate.tpm / TOKEN_SAFETY_MULTIPLIER) // 3
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


    def _summarize_batch(self, texts: List[str]) -> List[str]:
        """Summarize multiple texts in a single prompt."""
        if not texts:
            return []

        if len(texts) == 1:
            prompt = self._build_single_prompt(texts[0])
        else:
            prompt = self._build_numbered_prompt(texts)

        raw = self.genai.generate_content(prompt)

        if len(texts) == 1:
            return [raw]
        return self._parse_numbered_response(raw, len(texts))


    def _build_single_prompt(self, text: str) -> str:
        """Return a prompt string for a single-text summary."""
        return (
            "Summarize the following text in at most 2 concise sentences. "
            "Preserve key facts, numbers, and terminology. "
            "Return plain text only.\n\n"
            f"Text:\n{text}"
        )

    def _build_numbered_prompt(self, texts: List[str]) -> str:
        """Build a single prompt for multiple texts with numbering."""
        numbered = "\n\n".join(f"[{i+1}]\n{text}" for i, text in enumerate(texts))
        return (
            f"Summarize each of the following {len(texts)} numbered texts "
            "in at most 2 concise sentences each. "
            "Preserve key facts, numbers, and terminology. "
            "Return plain text only, prefixed with the matching number "
            "in the format [N] summary.\n\n"
            f"{numbered}"
        )


