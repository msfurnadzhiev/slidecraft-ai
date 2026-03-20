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