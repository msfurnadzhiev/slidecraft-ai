"""Schemas for assembled retrieval context (document-ordered passages + images)."""

from typing import List
from uuid import UUID

from src.schemas.base import BaseSchema
from src.schemas.document.chunk import ChunkSearchResult
from src.schemas.document.image import ImageSearchResult

# Default retrieval limits and similarity thresholds
_DEFAULT_CHUNK_LIMIT = 50
_DEFAULT_IMAGE_LIMIT = 10
_DEFAULT_SIMILARITY_THRESHOLD = 0.70
_DEFAULT_IMAGE_THRESHOLD = 0.60

class ContextRetrievalOptions(BaseSchema):
    """Options for retrieving context."""
    query: str | None = None
    chunk_limit: int = _DEFAULT_CHUNK_LIMIT
    chunk_threshold: float = _DEFAULT_SIMILARITY_THRESHOLD
    image_limit: int = _DEFAULT_IMAGE_LIMIT
    image_threshold: float = _DEFAULT_IMAGE_THRESHOLD

class RetrievedContext(BaseSchema):
    """Raw context retrieved from the database."""
    document_id: UUID
    total_pages: int
    options: ContextRetrievalOptions | None = None
    chunks: List[ChunkSearchResult]
    images: List[ImageSearchResult]

class ContextAnswer(BaseSchema):
    """Answer to a question about document context."""
    document_id: UUID
    question: str
    answer: str
