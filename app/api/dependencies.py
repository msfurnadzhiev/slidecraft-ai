"""API dependency definitions: database session and services."""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.bootstrap import get_db, get_document_service
from app.services.document_service import DocumentService as DocumentServiceClass

Database = Annotated[Session, Depends(get_db)]
DocumentService = Annotated[DocumentServiceClass, Depends(get_document_service)]
