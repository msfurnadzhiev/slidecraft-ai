"""API dependency definitions for FastAPI endpoints.

These can be used directly in endpoint function parameters with FastAPI's `Depends`.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from src.bootstrap import (
    get_db,
    get_document_repository,
    get_template_repository,
    get_document_ingestion_pipeline,
    get_template_ingestion_pipeline,
    get_presentation_builder_service,
    get_slide_builder_agent,
    get_presentation_structure_agent,
    get_content_generator_agent,
    get_quality_validator_agent,
    get_presentation_workflow,
)
from src.agents import PresentationStructureAgent as PresentationStructureAgentClass
from src.agents import ContentGeneratorAgent as ContentGeneratorAgentClass
from src.agents import SlideBuilderAgent as SlideBuilderAgentClass
from src.agents import QualityValidatorAgent as QualityValidatorAgentClass
from src.services.ingestion.document import DocumentRepository as DocumentRepositoryClass
from src.services.ingestion.template import TemplateRepository as TemplateRepositoryClass
from src.services.presentation.builder import PresentationBuilderService as PresentationBuilderServiceClass
from src.pipeline.document.pipeline import DocumentIngestionPipeline as DocumentIngestionPipelineClass
from src.pipeline.template.pipeline import TemplateIngestionPipeline as TemplateIngestionPipelineClass
from src.pipeline.presentation.workflow import PresentationWorkflow as PresentationWorkflowClass

Database = Annotated[Session, Depends(get_db)]

DocumentRepository = Annotated[DocumentRepositoryClass, Depends(get_document_repository)]
TemplateRepository = Annotated[TemplateRepositoryClass, Depends(get_template_repository)]

DocumentIngestionPipeline = Annotated[
    DocumentIngestionPipelineClass, Depends(get_document_ingestion_pipeline)
]
TemplateIngestionPipeline = Annotated[
    TemplateIngestionPipelineClass, Depends(get_template_ingestion_pipeline)
]

PresentationBuilderService = Annotated[
    PresentationBuilderServiceClass, Depends(get_presentation_builder_service)
]
SlideBuilderAgent = Annotated[SlideBuilderAgentClass, Depends(get_slide_builder_agent)]
PresentationStructureAgent = Annotated[
    PresentationStructureAgentClass, Depends(get_presentation_structure_agent)
]
ContentGeneratorAgent = Annotated[
    ContentGeneratorAgentClass, Depends(get_content_generator_agent)
]
QualityValidatorAgent = Annotated[
    QualityValidatorAgentClass, Depends(get_quality_validator_agent)
]
PresentationWorkflow = Annotated[
    PresentationWorkflowClass, Depends(get_presentation_workflow)
]
