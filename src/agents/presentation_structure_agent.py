"""Agent that suggests a list of slides for a presentation."""

import logging
import os
from uuid import UUID

from src.agents.core import BaseLLMProcessor
from src.schemas.presentation.presentation import PresentationStructure
from src.agents.tasks.presentation.runner import PresentationStructureTask
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)


class PresentationStructureAgent(BaseLLMProcessor, name="presentation-structure"):
    """Suggests a presentation outline as a list of slides."""

    def __init__(self) -> None:
        super().__init__()
        self._task = PresentationStructureTask(
            chat_model=self.chat_model
        )

    @trace_runtime
    def suggest_structure(self, document_id: UUID, user_request: str) -> PresentationStructure:
        """Generate a suggested presentation structure (list of slides) from user input."""
        log.info(
            "Generating presentation structure — document_id=%s, request='%s'",
            document_id,
            user_request[:120],
        )
        result = self.retry_policy.execute(self._task.generate, document_id, user_request)
        log.info(
            "Presentation structure ready — %d slides (document_id=%s)",
            len(result.slides),
            document_id,
        )
        return result
