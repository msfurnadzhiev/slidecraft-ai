"""Schemas for semantic search (internal service; no public endpoint)."""

from typing import List

from pydantic import Field

from app.schemas.base import BaseSchema


class SearchRequest(BaseSchema):
    """Parameters for a document-scoped semantic search."""

    document_id: str
    query: str
    limit: int = 25
    threshold: float | None = Field(default=0.70)


class SearchResultItem(BaseSchema):
    """A single chunk result with re-extracted text and similarity score."""

    chunk_id: str
    page_number: int
    chunk_index: int
    text: str
    score: float  # 0..1, higher = more similar


class SearchResponse(BaseSchema):
    """Response of a semantic search over one document."""

    document_id: str
    query: str
    results: List[SearchResultItem]
