"""API dependency definitions for FastAPI endpoints.

These can be used directly in endpoint function parameters with FastAPI's `Depends`.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.bootstrap import (
    get_content_analyzer,
    get_context_retriever,
    get_db,
    get_document_service,
    get_search_service,
)
from app.services.context_analyzer import ContentAnalyzer as ContentAnalyzerClass
from app.services.data_ingestion.document_service import (
    DocumentService as DocumentServiceClass,
)
from app.services.retrieval_context import ContextRetriever as ContextRetrieverClass

Database = Annotated[Session, Depends(get_db)]
DocumentService = Annotated[DocumentServiceClass, Depends(get_document_service)]
SearchService = Annotated[ContextRetrieverClass, Depends(get_search_service)]
ContextRetriever = Annotated[ContextRetrieverClass, Depends(get_context_retriever)]
ContentAnalyzer = Annotated[ContentAnalyzerClass, Depends(get_content_analyzer)]