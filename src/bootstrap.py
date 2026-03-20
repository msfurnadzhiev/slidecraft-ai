"""
Application bootstrap: startup, shared instances, and dependency wiring.

Owns DB initialization and construction of services so the API layer
does not depend on src.db or ingestion/storage modules directly.
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from src.db.init import init_db
from src.db.session import get_db as _get_db
from src.infrastructure.loaders import PDFLoader
from src.infrastructure.embeddings import TextEmbedder
from src.infrastructure.storage import LocalImageStorage
from src.agents.content_summarizer_agent import ContentSummarizerAgent
from src.agents.image_describer_agent import ImageDescriberAgent
from src.services.ingestion.document import DocumentService as DocumentServiceClass
from src.agents import PresentationStructureAgent as PresentationStructureAgentClass
from src.agents import ContentGeneratorAgent as ContentGeneratorAgentClass
from src.services.retrieval.semantic_search import SemanticSearchSevice

embedder = TextEmbedder.get_instance()
summarizer = ContentSummarizerAgent.get_instance()
image_describer = ImageDescriberAgent.get_instance()
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
        image_describer=image_describer,
        image_storage=image_storage,
    )


def get_presentation_structure_agent() -> PresentationStructureAgentClass:
    """Build PresentationStructureAgent for suggesting presentation outlines."""
    return PresentationStructureAgentClass()


def get_semantic_search_service(
    db: Session = Depends(get_db),
) -> SemanticSearchSevice:
    """Build SemanticSearchSevice for semantic search over content and images."""
    search_service = SemanticSearchSevice(db=db)
    search_service.db = db
    return search_service


def get_content_generator_agent(
    db: Session = Depends(get_db),
) -> ContentGeneratorAgentClass:
    """Build ContentGeneratorAgent for generating slide content."""
    search_service = get_semantic_search_service(db=db)
    return ContentGeneratorAgentClass(search_service=search_service)
