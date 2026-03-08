"""Schemas for chunk and image embeddings (separate vector spaces)."""

from typing import List, Optional

from app.schemas.base import BaseSchema


class ChunkEmbeddingCreate(BaseSchema):
    """Schema for creating a chunk embedding (sentence-transformers, 384-dim)."""

    document_id: str
    chunk_id: str
    vector: List[float]
    page_number: Optional[int] = None


class ImageEmbeddingCreate(BaseSchema):
    """Schema for creating an image embedding (CLIP, 512-dim)."""

    document_id: str
    image_id: str
    vector: List[float]
    page_number: Optional[int] = None
