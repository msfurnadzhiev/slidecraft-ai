"""Schemas for assembled retrieval context (document-ordered passages + images)."""

from typing import List

from app.schemas.base import BaseSchema


class ImageRef(BaseSchema):
    """Lightweight image reference attached to a passage."""

    image_id: str
    storage_path: str
    file_name: str


class Passage(BaseSchema):
    """A single coherent text block from one page, with associated images."""

    page_number: int
    text: str
    chunk_ids: List[str]
    score: float


class RetrievalContext(BaseSchema):
    """Ordered, deduplicated context assembled from search results."""

    document_id: str
    query: str
    passages: List[Passage]
