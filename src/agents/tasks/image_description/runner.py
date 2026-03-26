"""Task runner for image description with rate-limit-aware batching."""

import logging
from typing import Callable, List, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.core.utils import parse_numbered_response
from src.agents.core.utils import TOKEN_SAFETY_MULTIPLIER
from src.agents.core.rate_limiter import RateLimiter
from src.agents.tasks.image_description.prompts import build_single_prompt, build_numbered_prompt

_TOKENS_PER_IMAGE = 1000
_OUTPUT_TOKENS_PER_DESC = 100
_PROMPT_OVERHEAD_TOKENS = 60
_MAX_IMAGES_PER_BATCH = 5

log = logging.getLogger(__name__)


class ImageDescriptionTask:
    """Handles rate-limit-aware batch image description."""

    def __init__(
        self,
        chat_model: ChatGoogleGenerativeAI,
        rate_limiter: RateLimiter,
        call_with_retry: Callable,
    ) -> None:
        self.chat_model = chat_model
        self.rate_limiter = rate_limiter
        self._call_with_retry = call_with_retry

    def run(self, images: List[Tuple[bytes, str]]) -> List[str]:
        """Generate descriptions for a list of (image_bytes, mime_type) tuples."""
        if not images:
            return []

        batches = self._build_batches(images)
        log.info("Describing %d image(s) in %d batch(es).", len(images), len(batches))

        descriptions: List[str] = []
        for batch_no, batch in enumerate(batches, 1):
            estimated = self._estimate_tokens(batch)

            self.rate_limiter.wait_if_needed(estimated)
            log.info(
                "Batch %d/%d - %d image(s), ~%d est. tokens.",
                batch_no, len(batches), len(batch), estimated,
            )

            batch_descriptions = self._generate_batch(batch_no, len(batches), batch)
            self.rate_limiter.record(estimated)
            descriptions.extend(batch_descriptions)

            log.info("Batch %d/%d completed.", batch_no, len(batches))

        produced = sum(1 for d in descriptions if d)
        log.info("Image description finished – %d/%d image(s) described.", produced, len(images))

        return descriptions

    @staticmethod
    def _build_batches(
        images: List[Tuple[bytes, str]],
    ) -> List[List[Tuple[bytes, str]]]:
        return [
            images[i: i + _MAX_IMAGES_PER_BATCH]
            for i in range(0, len(images), _MAX_IMAGES_PER_BATCH)
        ]

    @staticmethod
    def _estimate_tokens(batch: List[Tuple[bytes, str]]) -> int:
        return int(
            (_PROMPT_OVERHEAD_TOKENS + _TOKENS_PER_IMAGE * len(batch) + _OUTPUT_TOKENS_PER_DESC * len(batch))
            * TOKEN_SAFETY_MULTIPLIER
        )

    def _generate_batch(
        self,
        batch_no: int,
        total_batches: int,
        batch: List[Tuple[bytes, str]],
    ) -> List[str]:
        """Describe one batch, raising on any failure so the caller can decide how to handle it."""
        descriptions = self._call_with_retry(
            self._describe_batch,
            batch,
            on_retry=self.rate_limiter.force_reset,
        )

        # Validate that every image in the batch received a non-empty description.
        # parse_numbered_response returns "" for images the model skipped or mis-formatted;
        # that is a silent data-quality failure we must surface here.
        empty_indices = [i for i, d in enumerate(descriptions) if not d]
        if empty_indices:
            raise RuntimeError(
                f"Batch {batch_no}/{total_batches}: "
                f"{len(empty_indices)} image(s) received empty descriptions "
                f"(indices {empty_indices}). "
                "Check the model's response format or the image content."
            )

        return descriptions

    def _describe_batch(self, images: List[Tuple[bytes, str]]) -> List[str]:
        if not images:
            return []

        prompt = build_single_prompt(images[0]) if len(images) == 1 else build_numbered_prompt(images)
        raw = self.chat_model.invoke(prompt).content

        if len(images) == 1:
            return [raw]
        return parse_numbered_response(raw, len(images))
