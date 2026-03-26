"""Repository for document read and delete operations."""

import logging
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.crud import document_crud
from src.infrastructure.storage.image_storage import ImageStorage
from src.schemas.document import DocumentResponse

log = logging.getLogger(__name__)


class DocumentRepository:
    """Thin facade over document CRUD and associated file cleanup."""

    def __init__(self, db: Session, image_storage: ImageStorage) -> None:
        self.db = db
        self.image_storage = image_storage

    def get_document(self, document_id: UUID) -> Optional[DocumentResponse]:
        """Retrieve a document by its ID."""
        document = document_crud.get_document(self.db, document_id)
        if not document:
            return None
        return DocumentResponse(
            document_id=document.document_id,
            file_name=document.file_name,
            total_pages=document.total_pages,
            metadata=document.metadata_,
        )

    def list_documents(self, skip: int = 0, limit: int = 100) -> List[DocumentResponse]:
        """List all documents with optional pagination."""
        documents = document_crud.get_all_documents(self.db, skip=skip, limit=limit)
        return [
            DocumentResponse(
                document_id=doc.document_id,
                file_name=doc.file_name,
                total_pages=doc.total_pages,
            )
            for doc in documents
        ]

    def delete_document(self, document_id: UUID) -> bool:
        """Delete a document and all associated chunks, embeddings, and image files."""
        document = document_crud.get_document(self.db, document_id)
        if document is None:
            return False

        images = document_crud.get_images_by_document(self.db, document_id)
        image_names = [img.file_name for img in images]

        document_crud.delete_document(self.db, document_id)
        self.db.flush()

        doc_id_str = str(document_id)
        for image_name in image_names:
            self.image_storage.delete(doc_id_str, image_name)

        return True
