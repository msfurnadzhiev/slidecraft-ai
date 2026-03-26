"""Agent that selects the best slide layout and distributes content across placeholders."""

import logging
from typing import List, Optional, Set

from src.agents.core import BaseAgent
from src.agents.tasks.slide_builder.runner import SlideAssignmentTask
from src.schemas.presentation.slide import SlideAssignment, SlideContent
from src.schemas.template import SlideLayoutResponse
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)

# Each slide assignment needs: 1 select_layout + N fill_placeholder calls + retries.
# 10 iterations covers layouts with up to ~6 placeholders plus one retry each.
MAX_TOOL_CALLS = 10


class SlideBuilderAgent(BaseAgent, name="slide-builder"):
    """Choose the best slide layout and assign text to its placeholders.

    Uses a tool-calling agent with two tools:
    - ``select_layout``    — picks the layout; the tool validates the choice.
    - ``fill_placeholder`` — fills one placeholder at a time; the tool validates
                             the index and returns remaining placeholders so the
                             agent knows what still needs filling.

    If the agent selects an invalid layout or submits a wrong placeholder index,
    the tool returns an actionable error message and the agent can self-correct
    within the same run rather than failing outright.
    """

    def __init__(self) -> None:
        super().__init__()
        self._task = SlideAssignmentTask(
            chat_model=self.chat_model,
            rate_limiter=self.rate_limiter,
            call_with_retry=self.retry_policy.execute,
        )

    @trace_runtime
    def assign(
        self,
        slide: SlideContent,
        layouts: List[SlideLayoutResponse],
        used_layout_indices: Optional[Set[int]] = None,
        revision_feedback: Optional[str] = None,
    ) -> SlideAssignment:
        """Return a layout selection and per-placeholder content assignment for *slide*.

        Args:
            slide: Populated slide content to assign a layout and distribute text for.
            layouts: All layouts available in the target template.
            used_layout_indices: Layout indices already chosen for earlier slides in
                this presentation run, passed to the agent to encourage visual variety.
            revision_feedback: Actionable feedback from the QualityValidatorAgent
                for a previous failed attempt.  When provided it is prepended to
                the agent prompt so the model can self-correct.

        Returns:
            :class:`SlideAssignment` with layout_index and placeholder_fills.
        """
        valid_indices = {lay.layout_index for lay in layouts}

        result = self._task.run(
            slide=slide,
            layouts=layouts,
            used_layout_indices=used_layout_indices,
            max_tool_calls=MAX_TOOL_CALLS,
            revision_feedback=revision_feedback,
        )

        if result.layout_index not in valid_indices:
            log.warning(
                "Task returned layout_index=%d outside valid set %s "
                "for slide #%d — falling back to min index.",
                result.layout_index,
                sorted(valid_indices),
                slide.slide_number,
            )
            fallback_index = min(valid_indices) if valid_indices else 0
            result = SlideAssignment(
                layout_index=fallback_index,
                placeholder_fills=result.placeholder_fills,
                reasoning=f"[fallback] {result.reasoning}",
            )

        log.info(
            "Slide #%d '%s' [%s] → layout_index=%d, %d fill(s) | %s",
            slide.slide_number,
            slide.title,
            slide.slide_type,
            result.layout_index,
            len(result.placeholder_fills),
            result.reasoning[:100],
        )
        return result
