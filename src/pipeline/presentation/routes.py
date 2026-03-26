"""Conditional edge routers for the presentation workflow graph."""

import logging
from typing import Final

from src.pipeline.presentation.state import MAX_QUALITY_ITERATIONS, WorkflowState

log = logging.getLogger(__name__)

# Route constants
MORE_CONTENT: Final = "more_content"
CONTENT_DONE: Final = "content_done"

RETRY: Final = "retry"
ACCEPTED: Final = "accepted"

MORE_SLIDES: Final = "more_slides"
ALL_DONE: Final = "all_done"


def route_content(state: WorkflowState) -> str:
    """Loop until all slide content is generated."""
    if state["content_slide_index"] < _total_slides(state):
        return MORE_CONTENT

    log.info(
        "route_content: all %d slides generated — proceeding to build",
        len(state["slide_contents"]),
    )
    return CONTENT_DONE

def route_validation(state: WorkflowState) -> str:
    """Handle validation retry / accept logic."""
    validation = state["last_validation"]

    if validation is None:
        log.warning("route_validation: missing validation result — retrying")
        return RETRY

    if validation.passed:
        return ACCEPTED

    if state["quality_attempts"] >= MAX_QUALITY_ITERATIONS:
        log.warning(
            "route_validation: slide #%d exhausted %d iteration(s) — accepting last assignment.",
            _current_slide_number(state),
            MAX_QUALITY_ITERATIONS,
        )
        return ACCEPTED

    return RETRY

def route_accept(state: WorkflowState) -> str:
    """Advance slide building or finish workflow."""
    if state["build_slide_index"] < len(state["slide_contents"]):
        return MORE_SLIDES

    log.info(
        "route_accept: all %d slide(s) built — proceeding to render",
        len(state["assignments"]),
    )
    return ALL_DONE

def _total_slides(state: WorkflowState) -> int:
    """Total number of slides in the presentation."""
    return len(state["structure"].slides)

def _current_slide_number(state: WorkflowState) -> int:
    """Current slide number."""
    return state["slide_contents"][state["build_slide_index"]].slide_number
