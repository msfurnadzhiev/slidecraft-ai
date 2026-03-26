"""LangGraph workflow state and tuning constants."""

import operator
from typing import Annotated, List, Optional, TypeAlias
from uuid import UUID

from typing_extensions import TypedDict

from src.schemas.presentation.presentation import PresentationStructure
from src.schemas.presentation.slide import SlideAssignment, SlideContent
from src.schemas.presentation.validation import SlideValidationResult
from src.schemas.template import SlideLayoutResponse

# Constants
MAX_QUALITY_ITERATIONS = 3
MIN_QUALITY_SCORE = 0.8

# Type aliases
SlideContents: TypeAlias = Annotated[List[SlideContent], operator.add]
Assignments: TypeAlias = Annotated[List[SlideAssignment], operator.add]
LayoutIndices: TypeAlias = Annotated[List[int], operator.add]

class WorkflowState(TypedDict):
    """Shared mutable state that flows through every node of the graph."""

    # -------- Inputs --------
    document_id: UUID
    user_request: str
    template_id: UUID
    template_file_path: str
    template_layouts: List[SlideLayoutResponse]
    presentation_name: Optional[str]

    # -------- Stage 1 --------
    structure: Optional[PresentationStructure]

    # -------- Stage 2 --------
    slide_contents: SlideContents
    content_slide_index: int

    # -------- Stage 3 --------
    assignments: Assignments
    used_layout_indices: LayoutIndices
    build_slide_index: int

    # Quality loop
    quality_attempts: int
    total_revisions: int
    revision_feedback: Optional[str]

    # Current processing state
    current_assignment: Optional[SlideAssignment]
    last_validation: Optional[SlideValidationResult]

    # -------- Stage 4 --------
    storage_path: Optional[str]


def _base_state() -> dict:
    """Internal helper to provide default state values."""
    return {
        "structure": None,
        "slide_contents": [],
        "content_slide_index": 0,
        "assignments": [],
        "used_layout_indices": [],
        "build_slide_index": 0,
        "quality_attempts": 0,
        "total_revisions": 0,
        "revision_feedback": None,
        "current_assignment": None,
        "last_validation": None,
        "storage_path": None,
    }


def make_initial_state(
    *,
    document_id: UUID,
    user_request: str,
    template_id: UUID,
    template_file_path: str,
    template_layouts: List[SlideLayoutResponse],
    presentation_name: Optional[str],
) -> WorkflowState:
    """Build the graph invocation payload with all keys at their starting values."""
    return WorkflowState(
        document_id=document_id,
        user_request=user_request,
        template_id=template_id,
        template_file_path=template_file_path,
        template_layouts=template_layouts,
        presentation_name=presentation_name,
        **_base_state(),
    )
