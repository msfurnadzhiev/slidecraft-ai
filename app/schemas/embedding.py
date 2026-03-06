from typing import List, Optional

from app.schemas.base import BaseSchema


class EmbeddingBase(BaseSchema):
    """Base schema for embedding."""

    vector: List[float]


class EmbeddingCreate(EmbeddingBase):
    """Unified schema for creating an embedding (chunk or image)."""

    document_id: str
    object_id: str
    object_type: str  # 'chunk' | 'image'
    page_number: Optional[int] = None


class EmbeddingResponse(EmbeddingBase):
    """Schema for embedding response."""

    embedding_id: str
    document_id: str
    object_id: str
    object_type: str
    page_number: Optional[int] = None

