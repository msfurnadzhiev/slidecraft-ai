"""LLM-based text summarization with rate-limit-aware batching."""

import logging
import os
from typing import List, Tuple

from src.agents.core.agent_base import BaseAgent
from src.agents.tasks.content_summarization.runner import SummarizationTask
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)


class ContentSummarizerAgent(BaseAgent, name="content-summarizer"):
    """Generate concise chunk summaries with rate-limit-aware batching."""

    def __init__(self) -> None:
        super().__init__()
        self._task = SummarizationTask(
            chat_model=self.chat_model,
            rate_limiter=self.rate_limiter,
            call_with_retry=self.retry_policy.execute,
        )

    @trace_runtime
    def summarize_pages(self, pages: List[Tuple[str, int]]) -> List[str]:
        """Summarize page texts respecting RPM / TPM limits.

        Args:
            pages: List of (text, token_count) pairs, one per page.

        Returns:
            Summaries in the same order (empty string for blank pages).
        """
        return self._task.run(pages)
