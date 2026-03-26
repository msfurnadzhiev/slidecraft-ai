"""Placeholder content filler tool for the slide-builder agent."""

import logging

from langchain_core.tools import BaseTool, tool

from src.agents.tasks.slide_builder.state import PlaceholderFillData, SlideAssignmentState
from src.agents.tasks.slide_builder.utils import split_placeholders

log = logging.getLogger(__name__)


def build_placeholder_filler_tool(state: SlideAssignmentState) -> BaseTool:
    """Create a placeholder-filler tool that records content into *state*.

    The tool validates each call against the layout that was selected by
    select_layout, prevents writing to auto-filled placeholders (e.g. title),
    and tracks which placeholders still need filling.
    """

    @tool("fill_placeholder")
    def fill_placeholder(placeholder_idx: int, text: str, reasoning: str = "") -> str:
        """Write content into a single slide placeholder.

        Call once per fillable placeholder listed by select_layout.  Do NOT
        call this for title placeholders — those are auto-filled.

        Args:
            placeholder_idx: The placeholder_idx of the target placeholder
                             (as shown in the select_layout confirmation).
            text: Presentation-ready bullet-point text.  Use 3-6 bullets
                  separated by newlines, each 5-10 words.  Synthesise from
                  the source material — do NOT copy verbatim.
            reasoning: Brief explanation of what content was placed here and why.
        """
        if state.selected_layout_index is None:
            return (
                "ERROR: No layout has been selected yet. "
                "Call select_layout first, then fill_placeholder for each placeholder."
            )

        layout = state.selected_layout
        auto_filled, fillable = split_placeholders(layout)

        fillable_indices = {el.placeholder_idx for el in fillable}
        auto_indices = {el.placeholder_idx for el in auto_filled}

        if placeholder_idx in auto_indices:
            return (
                f"ERROR: placeholder_idx={placeholder_idx} is auto-filled (title) "
                "and must NOT be written by you. "
                f"Fillable indices: {sorted(fillable_indices)}"
            )

        if placeholder_idx not in fillable_indices:
            return (
                f"ERROR: placeholder_idx={placeholder_idx} does not exist in "
                f"layout {state.selected_layout_index}. "
                f"Valid fillable indices: {sorted(fillable_indices)}"
            )

        if not text.strip():
            return (
                f"ERROR: text for placeholder_idx={placeholder_idx} is empty. "
                "Provide concise bullet-point content."
            )

        state.fills[placeholder_idx] = PlaceholderFillData(
            text=text.strip(),
            reasoning=reasoning.strip(),
        )

        remaining = fillable_indices - set(state.fills.keys())
        if remaining:
            return (
                f"placeholder_idx={placeholder_idx} filled. "
                f"Still need to fill: {sorted(remaining)}"
            )

        return (
            f"placeholder_idx={placeholder_idx} filled. "
            "All placeholders complete."
        )

    return fill_placeholder
