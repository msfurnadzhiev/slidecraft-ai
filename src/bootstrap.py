"""Application bootstrap: startup, shared instances, and dependency wiring."""

from fastapi import Depends
from sqlalchemy.orm import Session

from src.db.init import init_db
from src.db.session import get_db as _get_db
from src.infrastructure.loaders import PDFLoader, PPTLoader
from src.infrastructure.embeddings import TextEmbedder
from src.infrastructure.storage import LocalImageStorage, LocalTemplateStorage, LocalPresentationStorage
from src.agents.content_summarizer_agent import ContentSummarizerAgent
from src.agents.image_describer_agent import ImageDescriberAgent
from src.services.ingestion.document import DocumentRepository as DocumentRepositoryClass
from src.services.ingestion.template import TemplateRepository as TemplateRepositoryClass
from src.services.presentation.builder import PresentationBuilderService as PresentationBuilderServiceClass
from src.services.presentation.pptx_editor import PptxEditor
from src.services.presentation.content_writer import ContentWriter
from src.pipeline.document.pipeline import DocumentIngestionPipeline as DocumentIngestionPipelineClass
from src.pipeline.template.pipeline import TemplateIngestionPipeline as TemplateIngestionPipelineClass
from src.pipeline.presentation.workflow import PresentationWorkflow as PresentationWorkflowClass
from src.agents import PresentationStructureAgent as PresentationStructureAgentClass
from src.agents import ContentGeneratorAgent as ContentGeneratorAgentClass
from src.agents import SlideBuilderAgent as SlideBuilderAgentClass
from src.agents import QualityValidatorAgent as QualityValidatorAgentClass
from src.services.retrieval.semantic_search import SemanticSearchSevice

# ---------------------------------------------------------------------------
# Module-level singletons (stateless, safe to share across requests)
# ---------------------------------------------------------------------------

embedder = TextEmbedder.get_instance()
summarizer = ContentSummarizerAgent.get_instance()
image_describer = ImageDescriberAgent.get_instance()
file_loader = PDFLoader.get_instance()
image_storage = LocalImageStorage.get_instance()
template_storage = LocalTemplateStorage.get_instance()
presentation_storage = LocalPresentationStorage.get_instance()
ppt_loader = PPTLoader()


# ---------------------------------------------------------------------------
# Startup and database initialization
# ---------------------------------------------------------------------------

def startup() -> None:
    """Run application startup (DB extension and table creation). Call from FastAPI lifespan."""
    init_db()


def get_db():
    """Yield a request-scoped database session. Re-exported for API dependency injection."""
    yield from _get_db()


# ---------------------------------------------------------------------------
# Repository factories (CRUD + file cleanup, no pipeline logic)
# ---------------------------------------------------------------------------

def get_document_repository(db: Session = Depends(get_db)) -> DocumentRepositoryClass:
    """Build DocumentRepository with injected session and image storage."""
    return DocumentRepositoryClass(db=db, image_storage=image_storage)


def get_template_repository(db: Session = Depends(get_db)) -> TemplateRepositoryClass:
    """Build TemplateRepository with injected session and template storage."""
    return TemplateRepositoryClass(db=db, template_storage=template_storage)


# ---------------------------------------------------------------------------
# Ingestion pipeline factories (full ingest flow, session-scoped)
# ---------------------------------------------------------------------------

def get_document_ingestion_pipeline(db: Session = Depends(get_db)) -> DocumentIngestionPipelineClass:
    """Build DocumentIngestionPipeline with all required loaders, agents, and storage."""
    return DocumentIngestionPipelineClass(
        db=db,
        file_loader=file_loader,
        embedder=embedder,
        summarizer=summarizer,
        image_describer=image_describer,
        image_storage=image_storage,
    )


def get_template_ingestion_pipeline(db: Session = Depends(get_db)) -> TemplateIngestionPipelineClass:
    """Build TemplateIngestionPipeline with injected session, PPTLoader, and template storage."""
    return TemplateIngestionPipelineClass(
        db=db,
        ppt_loader=ppt_loader,
        template_storage=template_storage,
    )


# ---------------------------------------------------------------------------
# Presentation service and agent factories
# ---------------------------------------------------------------------------

def get_presentation_builder_service() -> PresentationBuilderServiceClass:
    """Build PresentationBuilderService with shared storage instances."""
    editor = PptxEditor(presentation_storage=presentation_storage)
    writer = ContentWriter(image_storage=image_storage)
    return PresentationBuilderServiceClass(
        template_storage=template_storage,
        editor=editor,
        writer=writer,
    )


def get_slide_builder_agent() -> SlideBuilderAgentClass:
    """Build SlideBuilderAgent for selecting the best layout per slide."""
    return SlideBuilderAgentClass()


def get_presentation_structure_agent() -> PresentationStructureAgentClass:
    """Build PresentationStructureAgent for suggesting presentation outlines."""
    return PresentationStructureAgentClass()


def get_semantic_search_service(
    db: Session = Depends(get_db),
) -> SemanticSearchSevice:
    """Build SemanticSearchSevice for semantic search over content and images."""
    return SemanticSearchSevice(db=db)


def get_content_generator_agent(
    db: Session = Depends(get_db),
) -> ContentGeneratorAgentClass:
    """Build ContentGeneratorAgent for generating slide content."""
    search_service = get_semantic_search_service(db=db)
    return ContentGeneratorAgentClass(search_service=search_service)


def get_quality_validator_agent() -> QualityValidatorAgentClass:
    """Build QualityValidatorAgent for validating slide assignments."""
    return QualityValidatorAgentClass()


def get_presentation_workflow(
    db: Session = Depends(get_db),
) -> PresentationWorkflowClass:
    """Build PresentationWorkflow with all required agent and service dependencies."""
    return PresentationWorkflowClass(
        structure_agent=get_presentation_structure_agent(),
        content_agent=get_content_generator_agent(db=db),
        builder_agent=get_slide_builder_agent(),
        validator_agent=get_quality_validator_agent(),
        builder_service=get_presentation_builder_service(),
    )
