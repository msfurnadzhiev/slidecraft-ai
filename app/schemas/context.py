"""Schemas for assembled retrieval context (document-ordered passages + images)."""

from typing import List

from app.schemas.base import BaseSchema
from app.schemas.chunk import ChunkSearchResult
from app.schemas.image import ImageSearchResult

# Default retrieval limits and similarity thresholds
_DEFAULT_CHUNK_LIMIT = 50
_DEFAULT_IMAGE_LIMIT = 10
_DEFAULT_SIMILARITY_THRESHOLD = 0.60
_DEFAULT_IMAGE_THRESHOLD = 0.22

class ContextRequest(BaseSchema):
    """Request parameters for retrieving context."""
    query: str | None = None
    chunk_limit: int = _DEFAULT_CHUNK_LIMIT
    chunk_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
    image_limit: int = _DEFAULT_IMAGE_LIMIT
    image_threshold: float = _DEFAULT_IMAGE_THRESHOLD

class ChunkReference(BaseSchema):
    """A single chunk result item."""
    chunk_id: str
    page_number: int
    chunk_index: int
    score: float

class ImageReference(BaseSchema):
    """A single image result item."""
    image_id: str
    page_number: int
    storage_path: str
    file_name: str
    score: float

class RawContext(BaseSchema):
    """Raw context retrieved from the database."""
    document_id: str
    query: str | None = None
    chunks: List[ChunkSearchResult]
    images: List[ImageSearchResult]

class Passage(BaseSchema):
    """A passage of text from a single page."""
    page_number: int
    text: str
    chunks: List[ChunkReference]
    images: List[ImageReference] = []

class RetrievalContext(BaseSchema):
    """Retrieval context for a document."""
    document_id: str
    query: str | None = None
    passages: List[Passage]
