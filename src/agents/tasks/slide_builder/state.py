"""Mutable state shared between slide-builder tools for a single agent run."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.schemas.template import SlideLayoutResponse


@dataclass
class PlaceholderFillData:
    """Content written by the agent for a single placeholder."""

    text: str
    reasoning: str = ""


@dataclass
class SlideAssignmentState:
    """Accumulates the agent's tool calls for one slide assignment run.

    Both tools capture a reference to the same instance so they can share
    data (selected layout -> placeholder validation -> fill recording) without
    any global state.
    """

    layouts: List[SlideLayoutResponse]
    selected_layout_index: Optional[int] = None
    layout_reasoning: str = ""
    fills: Dict[int, PlaceholderFillData] = field(default_factory=dict)

    @property
    def valid_indices(self) -> set:
        """Set of layout_index values available in this run."""
        return {lay.layout_index for lay in self.layouts}

    @property
    def selected_layout(self) -> Optional[SlideLayoutResponse]:
        """The layout the agent has selected, or None if not yet chosen."""
        if self.selected_layout_index is None:
            return None
        for layout in self.layouts:
            if layout.layout_index == self.selected_layout_index:
                return layout
        return None
