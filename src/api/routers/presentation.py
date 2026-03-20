"""Presentation router: generate presentation structure and content."""

import logging
import time
from uuid import UUID

from fastapi import APIRouter, HTTPException

from src.api.dependencies import (
    PresentationStructureAgent,
    ContentGeneratorAgent,
)
from src.schemas.base import UserRequest
from src.schemas.presentation.presentation import PresentationStructure, PresentationContent
from src.schemas.presentation.slide import SlideContent

log = logging.getLogger(__name__)

router = APIRouter(prefix="/presentation", tags=["presentation"])


@router.post(
    "/structure/{document_id}",
    response_model=PresentationStructure,
)
def suggest_presentation_structure(
    document_id: UUID,
    user_request: UserRequest,
    presentation_agent: PresentationStructureAgent,
):
    """Generate the structure for the presentation.

    Uses an LLM agent to generate the structure for the presentation.

    Args:
        document_id: Target document (path parameter).
        user_request: User request to generate the presentation structure.
        presentation_agent: Agent that generates the presentation structure.

    Raises:
        HTTPException: 404 if the document is not found.

    Returns:
        PresentationStructure with the suggested structure as text.
    """
    try:
        structure: PresentationStructure = \
            presentation_agent.suggest_structure(document_id, user_request.user_request)

        return structure
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post(
    "/content/{document_id}",
    response_model=PresentationContent,
)
def suggest_presentation_content(
    document_id: UUID,
    structure: PresentationStructure,
    content_generator_agent: ContentGeneratorAgent,
):
    """Generate the content for the presentation structure.

    Uses an LLM agent to generate the content for the presentation structure.

    Args:
        document_id: Target document (path parameter).
        structure: Presentation structure to generate the content for.
        content_generator_agent: Agent that generates the presentation content.

    Raises:
        HTTPException: 404 if the document is not found.

    Returns:
        PresentationContent with the suggested content for the presentation.
    """
    try:
        total_slides = len(structure.slides)
        log.info(
            "Content generation started — document_id=%s, slides=%d",
            document_id,
            total_slides,
        )
        wall_start = time.perf_counter()

        presentation_content: PresentationContent = PresentationContent(
            document_id=document_id,
            slides=[],
        )

        for i, slide in enumerate(structure.slides, 1):
            log.info(
                "Slide %d/%d — '%s' [%s]",
                i,
                total_slides,
                slide.title,
                slide.slide_type,
            )
            slide_content: SlideContent = content_generator_agent.generate_structure(document_id, slide)
            presentation_content.slides.append(slide_content)

        elapsed = time.perf_counter() - wall_start
        total_chunks = sum(len(s.content or []) for s in presentation_content.slides)
        total_images = sum(len(s.images or []) for s in presentation_content.slides)
        log.info(
            "Content generation complete — %d slides, %d text chunks, %d images in %.1fs (avg %.1fs/slide)",
            total_slides,
            total_chunks,
            total_images,
            elapsed,
            elapsed / total_slides if total_slides else 0,
        )

        return presentation_content

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
