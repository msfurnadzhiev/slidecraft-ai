"""
Application bootstrap: startup, shared instances, and dependency wiring.

Owns DB initialization and construction of services so the API layer
does not depend on app.db or ingestion/storage modules directly.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.init import init_db
from app.db.session import get_db as _get_db
from app.ingestion import (
    PDFImageExtractor,
    PDFLoader,
    TextChunker,
    EmbeddingGenerator,
)
from app.services.context_assembler import ContextAssembler as ContextAssemblerClass
from app.services.document_service import DocumentService as DocumentServiceClass
from app.services.search_service import SearchService as SearchServiceClass
from app.storage import LocalFileStorage, LocalImageStorage

# Shared instances (singletons) used by DocumentService
text_chunker = TextChunker.get_instance()
embedding_generator = EmbeddingGenerator.get_instance()
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
        embedding_generator=embedding_generator,
        file_storage=file_storage,
        image_extractor=image_extractor,
        image_storage=image_storage,
    )


def get_search_service(db: Session = Depends(get_db)) -> SearchServiceClass:
    """Build SearchService for internal semantic search (e.g. agent or other services)."""
    return SearchServiceClass(
        db=db,
        embedding_generator=embedding_generator,
        file_loader=file_loader,
        file_storage=file_storage,
    )


def get_context_assembler() -> ContextAssemblerClass:
    """Build ContextAssembler for transforming search results into ordered context."""
    return ContextAssemblerClass()
