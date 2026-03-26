"""Agent that validates the quality of a slide assignment.

TODO: Implement QualityValidatorAgent as a full LLM agent.

Implementation notes
--------------------
When ready to implement, convert this class to extend :class:`BaseAgent`:

    class QualityValidatorAgent(BaseAgent, name="quality-validator"):
        ...

and register the model in ``src/agents/core/agent_models.py``:

    AGENT_MODELS["quality-validator"] = os.getenv("QUALITY_VALIDATOR_AGENT_MODEL_NAME")

and expose the model env var in ``.env.example`` and
``deploy/docker-compose.yml`` as ``QUALITY_VALIDATOR_AGENT_MODEL_NAME``.

The agent should:
  - Receive a :class:`SlideContent` and the :class:`SlideAssignment` produced
    by the SlideBuilderAgent.
  - Use a structured-output LLM call (via :class:`QualityValidationTask`) to
    score the assignment across multiple quality dimensions.
  - Return a :class:`SlideValidationResult` with ``passed``, ``score``,
    ``feedback``, and ``issues``.
  - Set ``passed=False`` and provide specific, actionable ``feedback`` whenever
    ``score < MIN_QUALITY_SCORE`` so the SlideBuilderAgent can self-correct.

See :mod:`src.agents.tasks.quality_validation.runner` for detailed criteria.
"""

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
