from typing import Any, Dict, List, Optional

from app.schemas.base import BaseSchema


class PageContent(BaseSchema):
    """Schema for a page content."""

    page_number: int
    text: str
    char_count: int


class DocumentBase(BaseSchema):
    """Base schema for document."""
    
    document_id: str
    file_name: str
    total_pages: int
    storage_path: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentCreate(DocumentBase):
    """Schema for creating a document."""

    pass


class DocumentResponse(DocumentBase):
    """Schema for document response."""
    
    pass


class DocumentContent(DocumentBase):
    """Schema for a document content with pages."""
    
    pass


class DocumentIngestResponse(BaseSchema):
    """Schema for document ingestion response."""

    document_id: str
    file_name: str
    metadata: Optional[Dict[str, Any]] = None
    total_pages: int
    storage_path: str
    chunks: int
    images: int
