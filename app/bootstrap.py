"""
Application bootstrap: startup, shared instances, and dependency wiring.

Owns DB initialization and construction of services so the API layer
does not depend on app.db or ingestion/storage modules directly.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.init import init_db
from app.db.session import get_db as _get_db
from app.services.data_ingestion import (
    PDFLoader,
    Embedder,
)
from app.services.data_ingestion.summarizer import Summarizer
from app.services.context_analyzer import ContentAnalyzer as ContentAnalyzerClass
from app.services.retrieval_context import ContextRetriever as ContextRetrieverClass
from app.services.retrieval_context import TextSearch as TextSearchClass
from app.services.retrieval_context import ImageSearch as ImageSearchClass
from app.services.data_ingestion.document_service import (
    DocumentService as DocumentServiceClass,
)
from app.storage import LocalImageStorage

embedder = Embedder.get_instance()
summarizer = Summarizer.get_instance()
file_loader = PDFLoader.get_instance()
image_storage = LocalImageStorage.get_instance()


def startup() -> None:
    """Run application startup (DB extension and table creation). Call from FastAPI lifespan."""
    init_db()


def get_db():
    """Yield a request-scoped database session. Re-exported for API dependency injection."""
    yield from _get_db()


def get_document_service(db: Session = Depends(get_db)) -> DocumentServiceClass:
    """Build DocumentService with injected session and shared instances."""
    return DocumentServiceClass(
        db=db,
        file_loader=file_loader,
        embedder=embedder,
        summarizer=summarizer,
        image_storage=image_storage,
    )


def get_context_retriever(db: Session = Depends(get_db)) -> ContextRetrieverClass:
    """Build ContextRetriever with both embedders for chunk + image search."""
    text_search = TextSearchClass(
        db=db,
    )
    image_search = ImageSearchClass(db=db)

    return ContextRetrieverClass(
        text_search=text_search,
        image_search=image_search,
        embedder=embedder,
    )


def get_search_service(
    context_retriever: ContextRetrieverClass = Depends(get_context_retriever),
) -> ContextRetrieverClass:
    """Backwards-compatible alias for search dependency."""
    return context_retriever



def get_content_analyzer() -> ContentAnalyzerClass:
    """Build ContentAnalyzer for extracting themes and categorising passages."""
    return ContentAnalyzerClass()
