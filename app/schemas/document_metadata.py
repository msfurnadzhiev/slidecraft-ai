from typing import Optional

from app.schemas.base import BaseSchema


class DocumentMetadataItem(BaseSchema):
    """Single key-value metadata item for API response."""

    key: str
    value: Optional[str] = None
