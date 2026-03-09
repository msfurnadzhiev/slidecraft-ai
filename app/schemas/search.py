"""Schemas for semantic search (internal service; no public endpoint)."""

from typing import List

from pydantic import Field

from app.schemas.base import BaseSchema

# Default retrieval limits and similarity thresholds
_DEFAULT_CHUNK_LIMIT = 50
_DEFAULT_IMAGE_LIMIT = 10
_DEFAULT_SIMILARITY_THRESHOLD = 0.60
_DEFAULT_IMAGE_THRESHOLD = 0.22

class SearchRequest(BaseSchema):
    """Parameters for a document-scoped semantic search."""

    document_id: str
    query: str
    chunk_limit: int = _DEFAULT_CHUNK_LIMIT
    image_limit: int = _DEFAULT_IMAGE_LIMIT
    chunk_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
    image_threshold: float = _DEFAULT_IMAGE_THRESHOLD


class SearchResultItem(BaseSchema):
    """A single chunk result with re-extracted text and similarity score."""

    chunk_id: str
    page_number: int
    chunk_index: int
    text: str
    score: float  # 0..1, higher = more similar


class ImageResultItem(BaseSchema):
    """A single image result with similarity score."""

    image_id: str
    page_number: int
    storage_path: str
    file_name: str
    score: float  # 0..1, higher = more similar


class SearchResponse(BaseSchema):
    """Response of a semantic search over one document."""

    document_id: str
    query: str
    results: List[SearchResultItem]
    image_results: List[ImageResultItem] = []
