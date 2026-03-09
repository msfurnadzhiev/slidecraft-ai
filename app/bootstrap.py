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
    PDFImageExtractor,
    PDFLoader,
    TextChunker,
    TextEmbedder,
    ImageEmbedder,
)
from app.services.context_analyzer import ContentAnalyzer as ContentAnalyzerClass
from app.services.retrieval_context import ContextAssembler as ContextAssemblerClass
from app.services.retrieval_context import ContextRetriever as ContextRetrieverClass
from app.services.retrieval_context import TextSearch as TextSearchClass
from app.services.retrieval_context import ImageSearch as ImageSearchClass
from app.services.data_ingestion.document_service import (
    DocumentService as DocumentServiceClass,
)
from app.storage import LocalFileStorage, LocalImageStorage

text_chunker = TextChunker.get_instance()
text_embedder = TextEmbedder.get_instance()
image_embedder = ImageEmbedder.get_instance()
file_loader = PDFLoader.get_instance()
file_storage = LocalFileStorage.get_instance()
image_extractor = PDFImageExtractor.get_instance()
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
        text_chunker=text_chunker,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
        file_storage=file_storage,
        image_extractor=image_extractor,
        image_storage=image_storage,
    )


def get_context_retriever(db: Session = Depends(get_db)) -> ContextRetrieverClass:
    """Build ContextRetriever with both embedders for chunk + image search."""
    text_search = TextSearchClass(db=db, file_loader=file_loader, file_storage=file_storage)
    image_search = ImageSearchClass(db=db)

    return ContextRetrieverClass(
        text_search=text_search,
        image_search=image_search,
        text_embedder=text_embedder,
        image_embedder=image_embedder,
    )


def get_search_service(
    context_retriever: ContextRetrieverClass = Depends(get_context_retriever),
) -> ContextRetrieverClass:
    """Backwards-compatible alias for search dependency."""
    return context_retriever


def get_context_assembler() -> ContextAssemblerClass:
    """Build ContextAssembler for transforming search results into ordered context."""
    return ContextAssemblerClass()


def get_content_analyzer() -> ContentAnalyzerClass:
    """Build ContentAnalyzer for extracting themes and categorising passages."""
    return ContentAnalyzerClass()
