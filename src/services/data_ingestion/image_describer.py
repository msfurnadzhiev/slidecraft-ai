"""LLM-based image description generation with rate-limit-aware batching."""

import logging
from typing import List, Tuple

from google.genai import types as genai_types

from src.agents.base import BaseLLMProcessor, TOKEN_SAFETY_MULTIPLIER
from src.utils.profiling import trace_runtime

_TOKENS_PER_IMAGE = 1000
_OUTPUT_TOKENS_PER_DESC = 100
_PROMPT_OVERHEAD_TOKENS = 60
_MAX_IMAGES_PER_BATCH = 5

log = logging.getLogger(__name__)


class ImageDescriber(BaseLLMProcessor):
    """Generate concise image descriptions with rate-limit-aware batching."""

    def __init__(self) -> None:
        super().__init__()
        log.info(
            "ImageDescriber initialised (RPM=%d, TPM=%d).",
            self._rate.rpm, self._rate.tpm,
        )

    @trace_runtime
    def describe_images(self, images: List[Tuple[bytes, str]]) -> List[str]:
        """High-level orchestrator: generate descriptions for a list of images.
        
        Args:
            images: List of (image_bytes, mime_type) tuples.

        Returns:
            List of image descriptions (empty string for blank images).
        """
        if not images:
            return []

        batches = self._build_batches(images)
        log.info("Describing %d image(s) in %d batch(es).", len(images), len(batches))

        descriptions: List[str] = []
        for batch_no, batch in enumerate(batches, 1):
            estimated = self._estimate_tokens(batch)

            self._rate.wait_if_needed(estimated)
            log.info(
                "Batch %d/%d - %d image(s), ~%d est. tokens.",
                batch_no, len(batches), len(batch), estimated,
            )

            batch_descriptions = self._generate_batch_safe(batch_no, len(batches), batch)
            self._rate.record(estimated)
            descriptions.extend(batch_descriptions)

            log.info("Batch %d/%d completed.", batch_no, len(batches))

        produced = sum(1 for d in descriptions if d)
        log.info("Image description finished – %d/%d image(s) described.", produced, len(images))

        return descriptions


    @staticmethod
    def _build_batches(images: List[Tuple[bytes, str]]) -> List[List[Tuple[bytes, str]]]:
        """Split images into fixed-size batches."""
        return [
            images[i: i + _MAX_IMAGES_PER_BATCH] for i in range(0, len(images), _MAX_IMAGES_PER_BATCH)
        ]

    @staticmethod
    def _estimate_tokens(batch: List[Tuple[bytes, str]]) -> int:
        """Estimate the token usage for a batch with safety multiplier."""
        return int(
            (_PROMPT_OVERHEAD_TOKENS + _TOKENS_PER_IMAGE * len(batch) + _OUTPUT_TOKENS_PER_DESC * len(batch))
            * TOKEN_SAFETY_MULTIPLIER
        )


    def _generate_batch_safe(self, batch_no: int, total_batches: int, batch: List[Tuple[bytes, str]]) -> List[str]:
        """Call the LLM for a batch, returning empty descriptions on failure."""
        try:
            return self._describe_batch(batch)
        except Exception:
            log.exception(
                "Batch %d/%d failed – images will have empty descriptions.",
                batch_no, total_batches,
            )
            return [""] * len(batch)


    def _describe_batch(self, images: List[Tuple[bytes, str]]) -> List[str]:
        """Describe multiple images in a single prompt."""
        if not images:
            return []

        # Build prompt depending on single vs multiple images
        if len(images) == 1:
            prompt = self._build_single_prompt(images[0])
        else:
            prompt = self._build_numbered_prompt(images)

        raw = self.genai.generate_content(prompt)

        # Parse numbered response only for multi-image batch
        if len(images) == 1:
            return [raw]
        return self._parse_numbered_response(raw, len(images))


    def _build_single_prompt(self, image_info: Tuple[bytes, str]) -> list:
        """Return the prompt contents for a single image."""
        image_bytes, mime_type = image_info
        return [
            genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
            "Describe this image in 1-2 concise sentences. "
            "Focus on the visual content, any text visible, "
            "and the overall purpose of the image. "
            "Return plain text only.",
        ]


    def _build_numbered_prompt(self, images: List[Tuple[bytes, str]]) -> list:
        """Return the prompt contents for multiple images with numbering."""
        contents: list = []
        for idx, (image_bytes, mime_type) in enumerate(images, 1):
            contents.append(f"[{idx}]")
            contents.append(genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        contents.append(
            f"Describe each of the {len(images)} numbered images above "
            "in 1-2 concise sentences each. "
            "Focus on the visual content, any text visible, "
            "and the overall purpose of the image. "
            "Return plain text only, prefixed with the matching number "
            "in the format [N] description."
        )
        return contents
