"""Service layer for document ingestion and retrieval."""

from typing import List, Optional
from uuid import UUID, uuid4

import tiktoken
from sqlalchemy.orm import Session

from app.db.crud import document_crud, chunk_crud, image_crud
from app.services.data_ingestion import (
    FileLoader,
    Embedder,
)
from app.services.data_ingestion.summarizer import Summarizer
from app.schemas.document import (
    DocumentContent,
    DocumentIngestResponse,
    DocumentResponse,
    DocumentCreate,
)
from app.schemas.chunk import ChunkCreate
from app.schemas.image import ImageCreate

from app.storage import ImageStorage


class DocumentService:
    """Service for managing document ingestion and retrieval."""

    def __init__(
        self,
        db: Session,
        file_loader: FileLoader,
        embedder: Embedder,
        summarizer: Summarizer,
        image_storage: ImageStorage,
    ):
        """Initialize the document service."""
        self.db = db
        self.loader = file_loader
        self.embedder = embedder
        self.summarizer = summarizer
        self.image_storage = image_storage
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def ingest_document(self, file_path: str, original_filename: str) -> DocumentIngestResponse:
        """Ingest a document (PDF) and store extracted content in the database.
        
        Args:
            file_path: Path to the document file.
            original_filename: Original filename of the document.

        Returns:
            DocumentIngestResponse: Response containing the document ingestion details.
        """
        document_content = self.loader.load_file(file_path)
    
        # Create document in the database
        document_create = DocumentCreate(
            document_id=document_content.document_id,
            file_name=original_filename,
            total_pages=document_content.total_pages,
            metadata=document_content.metadata,
        )

        db_document = document_crud.create_document(self.db, document_create)

        # Process text and image content
        num_chunks_created = self._chunks_processing(document_content)
        num_images_created = self._images_processing(document_content)

        return DocumentIngestResponse(
            document_id=db_document.document_id,
            file_name=db_document.file_name,
            metadata=db_document.metadata_,
            total_pages=db_document.total_pages,
            chunks=num_chunks_created,
            images=num_images_created,
        )

    def _chunks_processing(self, document_content: DocumentContent) -> int:
        """Process chunks of a document and store in the database."""

        # TODO: re-enable summarization when rate limits are not an issue
        # page_texts = [p.text for p in pages]
        # summaries = self.summarizer.summarize_texts(page_texts)

        chunks_creates: List[ChunkCreate] = []

        for page in document_content.pages:
            chunk_id = uuid4()
            token_count = len(self.encoding.encode(page.text))

            content_vector = self.embedder.generate_embedding(page.text)

            chunk = ChunkCreate(
                chunk_id=chunk_id,
                document_id=document_content.document_id,
                page_number=page.page_number,
                token_count=token_count,
                content=page.text,
                content_vector=content_vector,
                summary="",
                summary_vector=[0.0] * 384,
            )
            chunks_creates.append(chunk)

        db_chunks = chunk_crud.create_chunks(self.db, chunks_creates)

        return len(db_chunks)

    def _images_processing(self, document_content: DocumentContent) -> int:
        """Process images of a document and store in the database.

        NOTE: Image description generation is disabled for now.
        Images are stored but without LLM-generated descriptions.
        """

        image_creates: List[ImageCreate] = []

        for raw_image in document_content.images:

            image_id = uuid4()

            # TODO: re-enable when vision model is available
            # description = self.summarizer.describe_image(
            #     raw_image.image_bytes, mime_type=raw_image.image_mime_type,
            # )
            # description_vector = self.embedder.generate_embedding(description)

            storage_path = self.image_storage.save_bytes(
                document_id=str(document_content.document_id),
                filename=raw_image.file_name,
                data=raw_image.image_bytes,
            )

            image = ImageCreate(
                image_id=image_id,
                document_id=document_content.document_id,
                page_number=raw_image.page_number,
                file_name=raw_image.file_name,
                storage_path=storage_path,
                description="",
                description_vector=[0.0] * 384,
            )

            image_creates.append(image)

        db_images = image_crud.create_images(self.db, image_creates)

        return len(db_images)

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
        """Delete a document along with all associated chunks, embeddings, and images."""
        document = document_crud.get_document(self.db, document_id)
        if document is None:
            return False

        # Gather associated files
        images = image_crud.get_images_by_document(self.db, document_id)
        image_names = [img.file_name for img in images]

        # Delete database entries
        document_crud.delete_document(self.db, document_id)
        self.db.flush()

        # Delete stored files
        doc_id_str = str(document_id)
        for image_name in image_names:
            self.image_storage.delete(doc_id_str, image_name)

        return True
