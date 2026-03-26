"""Agent that validates the quality of a slide assignment."""

import logging

from src.agents.tasks.quality_validation.runner import QualityValidationTask
from src.schemas.presentation.slide import SlideAssignment, SlideContent
from src.schemas.presentation.validation import SlideValidationResult
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)


# TODO: Convert to a BaseAgent subclass once the LLM evaluation is implemented
class QualityValidatorAgent:
    """Validates the quality of a SlideAssignment."""

    def __init__(self) -> None:
        self._task = QualityValidationTask()

    @trace_runtime
    def validate(
        self,
        slide_content: SlideContent,
        slide_assignment: SlideAssignment,
    ) -> SlideValidationResult:
        """Validate slide_assignment against slide_content.

        Args:
            slide_content: Source slide data used as the evaluation reference.
            slide_assignment: Layout selection and placeholder fills to assess.

        Returns:
            SlideValidationResult - always passed=True until the
            LLM evaluation is implemented.
        """
        return self._task.validate(slide_content, slide_assignment)
