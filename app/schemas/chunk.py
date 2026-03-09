from typing import Optional

from app.schemas.base import BaseSchema


class ChunkBase(BaseSchema):
    """Base schema for chunk. Text is re-extracted from PDF using start/end char offsets on the page."""

    document_id: str
    page_number: int
    chunk_index: int
    token_count: int
    start_char_offset: int  # offset into page text (0-based)
    end_char_offset: int    # end offset (exclusive) into page text


class ChunkCreate(ChunkBase):
    """Schema for creating a chunk. text is used at ingest (e.g. embedder) only; not persisted."""

    chunk_id: str
    text: str = ""  # used at ingest; precise text later from PDF via offsets
    vector: Optional[list[float]] = None


class ChunkObject(ChunkBase):
    """Schema for chunk object returned directly from storage/DB."""

    chunk_id: str
    vector: Optional[list[float]] = None


class ChunkSearchResult(BaseSchema):
    """Chunk semantic-search result used by retrieval context."""

    chunk_id: str
    page_number: int
    chunk_index: int
    text: str
    score: float


class ChunkReference(BaseSchema):
    """Schema for a chunk reference used by context assembly."""

    chunk_id: str
    page_number: int
    chunk_index: int
    score: float