"""Schemas for assembled retrieval context (document-ordered passages + images)."""

from typing import List

from app.schemas.base import BaseSchema


class PassageChunk(BaseSchema):
    """A single chunk in a passage."""
    chunk_id: str
    chunk_index: int
    score: float

class PassageImage(BaseSchema):
    """A single image in a passage."""
    image_id: str
    score: float

class Passage(BaseSchema):
    """A single coherent text block from one page, with chunk and image IDs. One score for the passage."""

    page_number: int
    text: str
    chunks: List[PassageChunk]
    images: List[PassageImage] = []


class RetrievalContext(BaseSchema):
    """Ordered, deduplicated context assembled from search results."""

    document_id: str
    query: str
    passages: List[Passage]
