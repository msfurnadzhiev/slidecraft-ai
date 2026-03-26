from typing import Optional
from uuid import UUID

from src.schemas.base import BaseSchema

class ChunkRawContent(BaseSchema):
    """Schema for a page extracted from a document."""

    page_number: int
    text: str
    char_count: int

class ChunkBase(BaseSchema):
    """Base schema for a page-level chunk."""

    document_id: UUID
    page_number: int
    token_count: int

class ChunkCreate(ChunkBase):
    """Schema for creating a chunk."""

    chunk_id: UUID
    content: str = ""
    summary: Optional[str] = None
    content_vector: Optional[list[float]] = None
    summary_vector: Optional[list[float]] = None

class ChunkSearchResult(BaseSchema):
    """Chunk semantic-search result used by retrieval context."""

    chunk_id: UUID
    page_number: int
    content: str
    summary: Optional[str] = None
    score: float