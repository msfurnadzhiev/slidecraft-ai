"""Schemas for layout element-related operations."""

from typing import Optional

from src.schemas.base import BaseSchema

class LayoutElementCreate(BaseSchema):
    """Schema for creating a layout placeholder element."""

    placeholder_idx: int
    role: Optional[str] = None
    x: float
    y: float
    width: float
    height: float


class LayoutElementResponse(LayoutElementCreate):
    """Schema for a layout placeholder element response."""

    id: int
    layout_id: int