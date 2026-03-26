"""Schemas for suggested presentation structure."""

from typing import List
from uuid import UUID

from src.schemas.base import BaseSchema
from src.schemas.presentation.slide import SlideStructure, SlideContent


class PresentationStructure(BaseSchema):
    """Structure of a presentation."""

    document_id: UUID
    slides: List[SlideStructure]


class PresentationContent(BaseSchema):
    """Content of a presentation."""

    document_id: UUID
    slides: List[SlideContent]


class PresentationGenerateRequest(BaseSchema):
    """Request body for the presentation generation endpoint."""

    template_id: UUID
    content: PresentationContent


class PresentationGenerateResponse(BaseSchema):
    """Response returned after a presentation has been generated and saved."""

    storage_path: str
    total_slides: int


class PresentationWorkflowRequest(BaseSchema):
    """Request body for the end-to-end workflow endpoint.

    Combines everything the full pipeline needs in a single request so callers
    do not have to chain structure → content → generate calls manually.
    """

    document_id: UUID
    template_id: UUID
    user_request: str


class PresentationWorkflowResponse(BaseSchema):
    """Response returned after the full workflow has completed."""

    storage_path: str
    total_slides: int
    quality_revisions: int = 0