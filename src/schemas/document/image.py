from typing import Optional
from uuid import UUID

from src.schemas.base import BaseSchema

class ImageRawContent(BaseSchema):
    """Schema for a image extracted from a document."""

    page_number: int
    image_bytes: bytes
    image_mime_type: str
    file_name: str

class ImageBase(BaseSchema):
    """Base schema for image."""

    document_id: UUID
    storage_path: str
    page_number: int
    file_name: str

class ImageCreate(ImageBase):
    """Schema for creating an image."""

    image_id: UUID
    description: Optional[str] = None
    description_vector: Optional[list[float]] = None

class ImageSearchResult(BaseSchema):
    """Image semantic-search result used by retrieval context."""

    image_id: UUID
    page_number: int
    storage_path: str
    file_name: str
    score: float
    description: Optional[str] = None
