"""CRUD operations for Document objects in the database."""

from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.models import Document
from app.schemas.document import DocumentCreate


def create_document(db: Session, document: DocumentCreate) -> Document:
    """
    Stage a new document in the session.

    Args:
        db: SQLAlchemy session.
        document: DocumentCreate schema.

    Returns:
        Document instance (not yet committed).
    """
    data = document.model_dump()
    data["metadata_"] = data.pop("metadata", None)
    db_document = Document(**data)
    db.add(db_document)
    return db_document


def get_document(db: Session, document_id: UUID) -> Optional[Document]:
    """
    Get a document by its ID.

    Args:
        db: SQLAlchemy session.
        document_id: Document primary key.

    Returns:
        Document instance if found, else None.
    """
    stmt = select(Document).where(Document.document_id == document_id)
    result = db.execute(stmt).scalars().first()
    return result


def get_all_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    """
    Get all documents with pagination.

    Args:
        db: SQLAlchemy session.
        skip: Number of rows to skip.
        limit: Maximum number of rows to return.

    Returns:
        List of Document instances.
    """
    stmt = select(Document).offset(skip).limit(limit)
    result = db.execute(stmt).scalars().all()
    return result


def delete_document(db: Session, document_id: UUID) -> bool:
    """
    Stage a document deletion in the session.

    Args:
        db: SQLAlchemy session.
        document_id: Document primary key.

    Returns:
        True if document was found and staged for deletion, False otherwise.
    """
    document = get_document(db, document_id)
    if document:
        db.delete(document)
        return True
    return False
