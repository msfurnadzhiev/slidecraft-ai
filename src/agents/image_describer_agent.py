"""LLM-based image description generation with rate-limit-aware batching."""

import logging
import os
from typing import List, Tuple

from src.agents.core import BaseAgent
from src.agents.tasks.image_description.runner import ImageDescriptionTask
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)

class ImageDescriberAgent(BaseAgent, name="image-describer"):
    """Generate concise image descriptions with rate-limit-aware batching."""

    def __init__(self) -> None:
        super().__init__()
        self._task = ImageDescriptionTask(
            chat_model=self.chat_model,
            rate_limiter=self.rate_limiter,
            call_with_retry=self.retry_policy.execute,
        )

    @trace_runtime
    def describe_images(self, images: List[Tuple[bytes, str]]) -> List[str]:
        """Generate descriptions for a list of (image_bytes, mime_type) tuples."""
        return self._task.run(images)
