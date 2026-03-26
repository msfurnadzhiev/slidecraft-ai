"""Schemas for slide layout-related operations."""

from typing import List, Optional
from uuid import UUID
from src.schemas.base import BaseSchema
from src.schemas.template.layout_element import LayoutElementCreate, LayoutElementResponse

class SlideLayoutCreate(BaseSchema):
    """Schema for creating a slide layout (without its parent template_id)."""

    layout_index: int
    name: Optional[str] = None
    elements: List[LayoutElementCreate] = []


class SlideLayoutResponse(BaseSchema):
    """Schema for a slide layout response."""

    id: int
    template_id: UUID
    layout_index: int
    name: Optional[str] = None
    elements: List[LayoutElementResponse] = []
