"""Layout selection tool for the slide-builder agent."""

import logging

from langchain_core.tools import BaseTool, tool

from src.agents.tasks.slide_builder.state import SlideAssignmentState
from src.agents.tasks.slide_builder.utils import split_placeholders

log = logging.getLogger(__name__)


def build_layout_selector_tool(state: SlideAssignmentState) -> BaseTool:
    """Create a layout-selection tool that writes its choice into *state*.

    The tool validates the requested index against the available layouts and
    returns a confirmation listing the fillable placeholders so the agent
    knows exactly what to fill next.
    """

    @tool("select_layout")
    def select_layout(layout_index: int, reasoning: str) -> str:
        """Select a slide layout by its index.

        Call this exactly once per slide.  The tool confirms the selection and
        returns the placeholder table for the chosen layout so you know which
        placeholder_idx values you must fill with fill_placeholder.

        Args:
            layout_index: The layout_index of the chosen layout (from the
                          Available layouts table).
            reasoning: One sentence explaining why this layout fits the slide.
        """
        if layout_index not in state.valid_indices:
            return (
                f"ERROR: layout_index={layout_index} is not valid. "
                f"Valid choices: {sorted(state.valid_indices)}. "
                "Call select_layout again with a valid index."
            )

        state.selected_layout_index = layout_index
        state.layout_reasoning = reasoning

        layout = state.selected_layout
        _, fillable = split_placeholders(layout)

        if not fillable:
            return (
                f"Layout {layout_index} ('{layout.name or '—'}') selected. "
                "This layout has no fillable placeholders — nothing to fill."
            )

        placeholder_table = "\n".join(
            f"  placeholder_idx={el.placeholder_idx}  role={el.role or 'content'}"
            f"  w={el.width:.2f}  h={el.height:.2f}"
            for el in fillable
        )

        return (
            f"Layout {layout_index} ('{layout.name or '—'}') selected.\n"
            f"Now call fill_placeholder for EACH of the following placeholders:\n"
            f"{placeholder_table}"
        )

    return select_layout
