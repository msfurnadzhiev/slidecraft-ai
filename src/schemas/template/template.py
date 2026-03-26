"""Schemas for template-related operations."""

from typing import List, Optional
from uuid import UUID

from src.schemas.base import BaseSchema
from src.schemas.template.slide_layout import SlideLayoutCreate, SlideLayoutResponse

class TemplateContent(BaseSchema):
    """In-memory representation of a parsed PowerPoint template.

    Returned by PPTLoader and consumed by TemplateService before persistence.
    """

    name: str
    file_path: str
    layouts: List["SlideLayoutCreate"] = []


class TemplateCreate(BaseSchema):
    """Schema for creating a template with its slide layouts."""

    name: str
    file_path: str
    layouts: List[SlideLayoutCreate] = []


class TemplateResponse(BaseSchema):
    """Flat template response without nested layouts."""

    template_id: UUID
    name: str
    file_path: str


class TemplateWithLayoutsResponse(TemplateResponse):
    """Template response including all slide layouts and their elements."""

    layouts: List[SlideLayoutResponse] = []
