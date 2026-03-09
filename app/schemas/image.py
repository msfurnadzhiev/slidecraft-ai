from typing import Dict, List, Optional

from app.schemas.base import BaseSchema

class ImageBase(BaseSchema):
    """Base schema for image."""

    document_id: str
    storage_path: str
    page_number: int
    file_name: str


class ImageCreate(ImageBase):
    """Schema for creating an image."""

    image_id: str
    vector: Optional[list[float]] = None


class ImageExtractionResult(BaseSchema):
    """Result of image extraction."""

    images: List[ImageCreate]
    image_bytes: Dict[str, bytes]


class ImageObject(ImageBase):
    """Schema for image object returned directly from storage/DB."""

    image_id: str
    vector: Optional[list[float]] = None


class ImageResponse(ImageBase):
    """Schema returned by document image-list endpoints."""

    image_id: str


class ImageSearchResult(BaseSchema):
    """Image semantic-search result used by retrieval context."""

    image_id: str
    page_number: int
    storage_path: str
    file_name: str
    score: float


class ImageReference(BaseSchema):
    """Schema for an image reference used by context assembly."""

    image_id: str
    page_number: int
    storage_path: str
    file_name: str
    score: float