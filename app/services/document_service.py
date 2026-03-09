"""Service layer for document ingestion and retrieval."""

from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.crud import document_crud, chunk_crud, image_crud
from app.ingestion import FileLoader, TextChunker, PDFImageExtractor, ImageExtractionResult
from app.ingestion.embeddings import TextEmbedder, ImageEmbedder
from app.schemas.document import (
    DocumentContent,
    DocumentIngestResponse,
    DocumentResponse,
    DocumentCreate,
)
from app.schemas.image import ImageResponse
from app.storage import FileStorage, ImageStorage


class DocumentService:
    """Service for managing document ingestion and retrieval."""

    def __init__(
        self,
        db: Session,
        file_loader: FileLoader,
        text_chunker: TextChunker,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
        file_storage: FileStorage,
        image_extractor: PDFImageExtractor,
        image_storage: ImageStorage,
    ):
        """Initialize the document service."""
        self.db = db
        self.loader = file_loader
        self.chunker = text_chunker
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.storage = file_storage
        self.image_extractor = image_extractor
        self.image_storage = image_storage

    def ingest_document(self, file_path: str, original_filename: str) -> DocumentIngestResponse:
        """Ingest a document (PDF) and store it with chunks, embeddings, and images.
        
        Args:
            file_path: Path to the document file.
            original_filename: Original filename of the document.

        Returns:
            DocumentIngestResponse: Response containing the document ingestion details.
        """
        document_content = self.loader.load_file(file_path)
        metadata = self.loader.extract_metadata(file_path)
        document_id = document_content.document_id

        # Compute storage path
        doc_storage_path = self.storage.relative_path(document_id, original_filename)

        # Stage document in the database
        document_create = DocumentCreate(
            document_id=document_id,
            file_name=original_filename,
            total_pages=document_content.total_pages,
            storage_path=doc_storage_path,
            metadata=metadata,
        )
        db_document = document_crud.create_document(self.db, document_create)

        # Process text and image content
        num_chunks_created = self._extract_chunks(document_id, document_content)
        num_images_created = self._extract_images(file_path, document_id)

        # Save PDF file to storage
        self.storage.save(
            source_path=file_path,
            document_id=document_id,
            filename=original_filename,
        )

        return DocumentIngestResponse(
            document_id=db_document.document_id,
            file_name=db_document.file_name,
            metadata=db_document.metadata_,
            total_pages=db_document.total_pages,
            storage_path=db_document.storage_path,
            chunks=num_chunks_created,
            images=num_images_created,
        )

    def _extract_chunks(self, document_id: str, document_content: DocumentContent) -> int:
        """Chunk document text, generate embeddings, and store in the database."""
        chunk_creates = self.chunker.chunk_document(document_content)
        if not chunk_creates:
            return 0

        self.text_embedder.embed_chunks(chunk_creates)
        chunk_crud.create_chunks(self.db, chunk_creates)
        return len(chunk_creates)

    def _extract_images(self, file_path: str, document_id: str) -> int:
        """Extract images, generate CLIP embeddings, and store in the database."""
        extraction: ImageExtractionResult = self.image_extractor.extract_images(
            file_path, document_id, self.image_storage
        )
        if not extraction.images:
            return 0

        self.image_embedder.embed_images(extraction.images, extraction.image_bytes)
        image_crud.create_images(self.db, extraction.images)
        return len(extraction.images)

    def get_document(self, document_id: str) -> Optional[DocumentResponse]:
        """Retrieve a document by its ID."""
        document = document_crud.get_document(self.db, document_id)
        if not document:
            return None
        return DocumentResponse(
            document_id=document.document_id,
            file_name=document.file_name,
            total_pages=document.total_pages,
            storage_path=document.storage_path,
            metadata=document.metadata_,
        )

    def get_document_images(self, document_id: str) -> List[ImageResponse]:
        """Retrieve all images associated with a document."""
        if not document_crud.get_document(self.db, document_id):
            return []
        images = image_crud.get_images_by_document(self.db, document_id)
        return [
            ImageResponse(
                image_id=img.image_id,
                document_id=img.document_id,
                storage_path=img.storage_path,
                page_number=img.page_number,
                file_name=img.file_name,
            )
            for img in images
        ]

    def list_documents(self, skip: int = 0, limit: int = 100) -> List[DocumentResponse]:
        """List all documents with optional pagination."""
        documents = document_crud.get_all_documents(self.db, skip=skip, limit=limit)
        return [
            DocumentResponse(
                document_id=doc.document_id,
                file_name=doc.file_name,
                total_pages=doc.total_pages,
                storage_path=doc.storage_path,
            )
            for doc in documents
        ]

    def delete_document(self, document_id: str) -> bool:
        """Delete a document along with all associated chunks, embeddings, and stored files."""
        document = document_crud.get_document(self.db, document_id)
        if document is None:
            return False

        # Gather associated files
        file_name = document.file_name
        images = image_crud.get_images_by_document(self.db, document_id)
        image_names = [img.file_name for img in images]

        # Delete database entries
        document_crud.delete_document(self.db, document_id)
        self.db.flush()

        # Delete stored files
        for image_name in image_names:
            self.image_storage.delete(document_id, image_name)
        self.storage.delete(document_id, file_name)

        return True
