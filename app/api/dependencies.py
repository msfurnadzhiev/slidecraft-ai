"""API dependency definitions: database session and services."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.bootstrap import get_db, get_document_service, get_search_service, get_context_assembler
from app.services.context_assembler import ContextAssembler as ContextAssemblerClass
from app.services.document_service import DocumentService as DocumentServiceClass
from app.services.search_service import SearchService as SearchServiceClass

Database = Annotated[Session, Depends(get_db)]
DocumentService = Annotated[DocumentServiceClass, Depends(get_document_service)]
SearchService = Annotated[SearchServiceClass, Depends(get_search_service)]
ContextAssembler = Annotated[ContextAssemblerClass, Depends(get_context_assembler)]
