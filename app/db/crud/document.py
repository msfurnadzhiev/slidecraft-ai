from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.models import Document
from app.schemas.document import DocumentCreate


def create_document(db: Session, document: DocumentCreate) -> Document:
    """Stage a new document in the session."""
    data = document.model_dump()
    data["metadata_"] = data.pop("metadata", None)
    db_document = Document(**data)
    db.add(db_document)
    return db_document


def get_document(db: Session, document_id: str) -> Optional[Document]:
    """Get a document by ID."""
    return db.query(Document).filter(Document.document_id == document_id).first()


def get_all_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    """Get all documents with pagination."""
    return db.query(Document).offset(skip).limit(limit).all()


def delete_document(db: Session, document_id: str) -> bool:
    """Stage a document deletion in the session."""
    document = get_document(db, document_id)
    if document:
        db.delete(document)
        return True
    return False
