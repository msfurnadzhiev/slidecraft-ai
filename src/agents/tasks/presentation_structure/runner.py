"""Task runner for presentation structure generation."""

import logging
import time
from typing import Callable
from uuid import UUID

from langchain_google_genai import ChatGoogleGenerativeAI

from src.schemas.presentation.presentation import PresentationStructure
from src.agents.core.instrumentation import LLMInstrumentationCallback
from src.agents.core.rate_limiter import RateLimiter
from src.agents.tasks.presentation_structure.prompts import build_structure_prompt

log = logging.getLogger(__name__)


class PresentationStructureTask:
    """Generates a PresentationStructure from a user request."""

    def __init__(
        self,
        chat_model: ChatGoogleGenerativeAI,
        rate_limiter: RateLimiter,
        call_with_retry: Callable,
    ) -> None:
        self.chat_model = chat_model
        self.rate_limiter = rate_limiter
        self._call_with_retry = call_with_retry

    def generate(self, document_id: UUID, user_request: str) -> PresentationStructure:
        """Generate a validated PresentationStructure object."""
        instrumentation = LLMInstrumentationCallback(rate_limiter=self.rate_limiter)

        log.debug("Invoking LLM for presentation structure (document_id=%s)", document_id)
        t0 = time.perf_counter()

        # Attach directly to the model so callbacks fire at the model level,
        # not just at the outer runnable wrapper created by with_structured_output.
        instrumented_model = self.chat_model.with_config(callbacks=[instrumentation])
        structured_llm = instrumented_model.with_structured_output(PresentationStructure)
        prompt = build_structure_prompt(document_id, user_request)

        result = self._call_with_retry(
            structured_llm.invoke,
            prompt,
            on_retry=self.rate_limiter.force_reset,
        )

        elapsed = time.perf_counter() - t0
        log.debug(
            "LLM returned presentation structure — %d slides in %.2fs",
            len(result.slides),
            elapsed,
        )
        instrumentation.log_summary(f"structure document_id={document_id}")
        return result
