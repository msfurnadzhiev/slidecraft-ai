from enum import Enum
from uuid import UUID

from src.schemas.base import BaseSchema

class TextContent(BaseSchema):
    """Text content of a slide."""
    text: str
    chunk_id: UUID
    score: float

class ImageContent(BaseSchema):
    """Image content of a slide."""
    image_url: str
    image_id: UUID
    score: float

class SlideType(str, Enum):
    TITLE = "title"
    CONTENT = "content"
    IMAGE = "image"
    DATA = "data"
    CLOSING = "closing"

class SlideStructure(BaseSchema):
    """Content the metadata and planning hints for a single slide."""

    slide_number: int
    slide_type: SlideType
    title: str
    description: str

class SlideContent(SlideStructure):
    """"""
    content: list[TextContent] | None = None
    images: list[ImageContent] | None = None