"""API dependency definitions for FastAPI endpoints.

These can be used directly in endpoint function parameters with FastAPI's `Depends`.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from src.bootstrap import (
    get_db,
    get_document_service,
    get_presentation_structure_agent,
    get_content_generator_agent,
)
from src.agents import PresentationStructureAgent as PresentationStructureAgentClass
from src.agents import ContentGeneratorAgent as ContentGeneratorAgentClass
from src.services.ingestion.document import DocumentService as DocumentServiceClass

Database = Annotated[Session, Depends(get_db)]
DocumentService = Annotated[DocumentServiceClass, Depends(get_document_service)]
PresentationStructureAgent = Annotated[
    PresentationStructureAgentClass, Depends(get_presentation_structure_agent)
]
ContentGeneratorAgent = Annotated[
    ContentGeneratorAgentClass,
    Depends(get_content_generator_agent),
]
