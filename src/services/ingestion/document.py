"""Service layer for document ingestion and retrieval."""

import logging
from typing import List, Optional
from uuid import UUID, uuid4

import tiktoken
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

from src.db.crud import document_crud, chunk_crud, image_crud
from src.infrastructure.loaders import FileLoader
from src.infrastructure.embeddings import TextEmbedder
from src.agents.content_summarizer_agent import ContentSummarizerAgent
from src.agents.image_describer_agent import ImageDescriberAgent
from src.schemas.document.document import (
    DocumentContent,
    DocumentIngestResponse,
    DocumentResponse,
    DocumentCreate,
)
from src.schemas.document.chunk import ChunkCreate
from src.schemas.document.image import ImageCreate

from src.infrastructure.storage.image_storage import ImageStorage


class DocumentService:
    """Service for managing document ingestion and retrieval."""

    def __init__(
        self,
        db: Session,
        file_loader: FileLoader,
        embedder: TextEmbedder,
        summarizer: ContentSummarizerAgent,
        image_describer: ImageDescriberAgent,
        image_storage: ImageStorage,
    ):
        """Initialize the document service."""
        self.db = db
        self.loader = file_loader
        self.embedder = embedder
        self.summarizer = summarizer
        self.image_describer = image_describer
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
        log.info("Ingesting document: %s (path=%s).", original_filename, file_path)

        document_content = self.loader.load_file(file_path)
        log.info(
            "Loaded document %s – %d page(s), %d image(s).",
            document_content.document_id,
            document_content.total_pages,
            len(document_content.images),
        )

        document_create = DocumentCreate(
            document_id=document_content.document_id,
            file_name=original_filename,
            total_pages=document_content.total_pages,
            metadata=document_content.metadata,
        )

        db_document = document_crud.create_document(self.db, document_create)
        log.info("Document record created: %s.", db_document.document_id)

        num_chunks_created = self._chunks_processing(document_content)
        num_images_created = self._images_processing(document_content)

        log.info(
            "Ingestion complete for %s – %d chunk(s), %d image(s).",
            db_document.document_id, num_chunks_created, num_images_created,
        )

        return DocumentIngestResponse(
            document_id=db_document.document_id,
            file_name=db_document.file_name,
            metadata=db_document.metadata_,
            total_pages=db_document.total_pages,
            chunks=num_chunks_created,
            images=num_images_created,
        )

    def _chunks_processing(self, document_content: DocumentContent) -> int:
        """Process chunks of a document: summarize, embed, and store in the database."""

        pages = document_content.pages
        stripped_texts = [p.text.strip() for p in pages]
        token_counts = [len(self.encoding.encode(t)) for t in stripped_texts]

        log.info(
            "Processing %d page(s) for document %s (total tokens: %d).",
            len(pages), document_content.document_id, sum(token_counts),
        )

        # --- summarization ---
        page_data = list(zip(stripped_texts, token_counts))
        summaries = self.summarizer.summarize_pages(page_data)

        # --- embedding generation ---
        # Store None for empty content/summary. Zero vectors cause pgvector cosine_distance
        # to return NaN, which filters out all results in search_similar (NaN <= threshold is False).
        log.info("Generating embeddings for %d page(s).", len(pages))
        content_vectors = []
        summary_vectors = []

        for idx, text in enumerate(stripped_texts):
            if text:
                content_vectors.append(self.embedder.generate_embedding(text))
            else:
                content_vectors.append(None)

            summary_text = summaries[idx]
            if summary_text:
                summary_vectors.append(self.embedder.generate_embedding(summary_text))
            else:
                summary_vectors.append(None)

        log.info("Embedding generation complete for document %s.", document_content.document_id)

        # --- chunk creation ---
        chunks_creates: List[ChunkCreate] = []

        for idx, page in enumerate(pages):
            chunk = ChunkCreate(
                chunk_id=uuid4(),
                document_id=document_content.document_id,
                page_number=page.page_number,
                token_count=token_counts[idx],
                content=page.text,
                content_vector=content_vectors[idx],
                summary=summaries[idx],
                summary_vector=summary_vectors[idx],
            )
            chunks_creates.append(chunk)

        db_chunks = chunk_crud.create_chunks(self.db, chunks_creates)
        log.info("Stored %d chunk(s) for document %s.", len(db_chunks), document_content.document_id)

        return len(db_chunks)

    def _images_processing(self, document_content: DocumentContent) -> int:
        """Process images of a document: describe, embed, store in the database."""

        images = document_content.images
        if not images:
            log.info("No images to process for document %s.", document_content.document_id)
            return 0

        log.info("Processing %d image(s) for document %s.", len(images), document_content.document_id)

        image_pairs = [(img.image_bytes, img.image_mime_type) for img in images]
        descriptions = self.image_describer.describe_images(image_pairs)

        # Store None for empty descriptions. Zero vectors cause NaN distance, filtering out all results.
        description_vectors = []
        for idx, desc in enumerate(descriptions):
            if desc:
                description_vectors.append(self.embedder.generate_embedding(desc))
            else:
                description_vectors.append(None)

        image_creates: List[ImageCreate] = []

        for idx, raw_image in enumerate(images):
            storage_path = self.image_storage.save_bytes(
                document_id=str(document_content.document_id),
                filename=raw_image.file_name,
                data=raw_image.image_bytes,
            )
            log.debug(
                "Stored image %s (page %d) at %s.",
                raw_image.file_name, raw_image.page_number, storage_path,
            )

            image = ImageCreate(
                image_id=uuid4(),
                document_id=document_content.document_id,
                page_number=raw_image.page_number,
                file_name=raw_image.file_name,
                storage_path=storage_path,
                description=descriptions[idx],
                description_vector=description_vectors[idx],
            )

            image_creates.append(image)

        db_images = image_crud.create_images(self.db, image_creates)
        log.info("Stored %d image(s) for document %s.", len(db_images), document_content.document_id)

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
