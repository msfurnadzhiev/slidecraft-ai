"""Agent that suggests a list of slides for a presentation."""

import logging
from uuid import UUID

from src.agents.core import BaseAgent
from src.schemas.presentation.presentation import PresentationStructure
from src.agents.tasks.presentation_structure.runner import PresentationStructureTask
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)


class PresentationStructureAgent(BaseAgent, name="presentation-structure"):
    """Suggests a presentation outline as a list of slides."""

    def __init__(self) -> None:
        super().__init__()
        self._task = PresentationStructureTask(
            chat_model=self.chat_model,
            rate_limiter=self.rate_limiter,
            call_with_retry=self.retry_policy.execute,
        )

    @trace_runtime
    def suggest_structure(self, document_id: UUID, user_request: str) -> PresentationStructure:
        """Generate a suggested presentation structure (list of slides) from user input."""
        log.info(
            "Generating presentation structure — document_id=%s, request='%s'",
            document_id,
            user_request[:120],
        )
        result = self._task.generate(document_id, user_request)
        log.info(
            "Presentation structure ready — %d slides (document_id=%s)",
            len(result.slides),
            document_id,
        )
        return result
