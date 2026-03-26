"""Schemas for document-related operations."""

from typing import Any, Dict, List, Optional
from uuid import UUID

from src.schemas.base import BaseSchema

from src.schemas.document.chunk import ChunkRawContent
from src.schemas.document.image import ImageRawContent

class DocumentRawContent(BaseSchema):
    """Raw content of a loaded document with its page text and images."""

    document_id: UUID
    file_name: str
    total_pages: int
    pages: List[ChunkRawContent] = []
    images: List[ImageRawContent] = []
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
