"""Schemas for document-related operations."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from app.schemas.base import BaseSchema


class PageContent(BaseSchema):
    """Schema for a page extracted from a document."""

    page_number: int
    text: str
    char_count: int

class ImageContent(BaseSchema):
    """Schema for a image extracted from a document."""

    page_number: int
    image_bytes: bytes
    image_mime_type: str
    file_name: str


class DocumentContent(BaseSchema):
    """In-memory representation of a loaded document with its page text and images."""

    document_id: UUID
    file_name: str
    total_pages: int
    pages: List[PageContent] = []
    images: List[ImageContent] = []
    metadata: Optional[Dict[str, Any]] = None


class DocumentBase(BaseSchema):
    """Base schema for document."""
    
    document_id: UUID
    file_name: str
    total_pages: int
    metadata: Optional[Dict[str, Any]] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""

    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    
    pass


class DocumentIngestResponse(BaseSchema):
    """Schema for document ingestion response."""

    document_id: UUID
    file_name: str
    metadata: Optional[Dict[str, Any]] = None
    total_pages: int
    chunks: int
    images: int
