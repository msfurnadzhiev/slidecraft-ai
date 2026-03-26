"""Document ingestion pipeline: load → chunk → embed → persist."""

import logging
from typing import List, Optional, Tuple
from uuid import uuid4

import tiktoken
from sqlalchemy.orm import Session

from src.db.crud import document_crud
from src.infrastructure.loaders.pdf_loader import PDFLoader
from src.infrastructure.embeddings.text_embedder import TextEmbedder
from src.infrastructure.storage.image_storage import ImageStorage
from src.agents.content_summarizer_agent import ContentSummarizerAgent
from src.agents.image_describer_agent import ImageDescriberAgent
from src.schemas.document import (
    DocumentRawContent,
    DocumentCreate,
    DocumentIngestResponse,
    ChunkCreate,
    ImageCreate,
)

log = logging.getLogger(__name__)


class DocumentIngestionPipeline:
    """Orchestrates the full document ingestion flow."""

    def __init__(
        self,
        db: Session,
        file_loader: PDFLoader,
        embedder: TextEmbedder,
        summarizer: ContentSummarizerAgent,
        image_describer: ImageDescriberAgent,
        image_storage: ImageStorage,
    ) -> None:
        """Initialize the document ingestion pipeline."""
        self.db = db
        self.loader = file_loader
        self.embedder = embedder
        self.summarizer = summarizer
        self.image_describer = image_describer
        self.image_storage = image_storage
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def ingest(self, file_path: str, original_filename: str) -> DocumentIngestResponse:
        """Run the full ingestion pipeline for a single PDF file.

        Args:
            file_path: Absolute path to the PDF on disk.
            original_filename: Original filename supplied by the client.

        Returns:
            DocumentIngestResponse summarising what was persisted.
        """
        log.info("Ingesting document: %s (path=%s).", original_filename, file_path)

        document_content = self.loader.load_file(file_path)

        # Create the Document record
        document_create = DocumentCreate(
            document_id=document_content.document_id,
            file_name=original_filename,
            total_pages=document_content.total_pages,
            metadata=document_content.metadata,
        )
        db_document = document_crud.create_document(self.db, document_create)

        # Process chunks and images
        num_chunks = self._process_chunks(document_content)
        num_images = self._process_images(document_content)

        log.info(
            "Ingestion complete for %s - %d chunk(s), %d image(s).",
            db_document.document_id,
            num_chunks,
            num_images,
        )

        return DocumentIngestResponse(
            document_id=db_document.document_id,
            file_name=db_document.file_name,
            metadata=db_document.metadata_,
            total_pages=db_document.total_pages,
            chunks=num_chunks,
            images=num_images,
        )

    # ---------------------------------------------------------------------------
    # Chunk processing
    # ---------------------------------------------------------------------------

    def _process_chunks(self, document_content: DocumentRawContent) -> int:
        """Summarise, embed, and persist page chunks. Returns the count stored."""
        pages = document_content.pages
        texts = [p.text.strip() for p in pages]
        token_counts = [self._count_tokens(t) for t in texts]

        log.info(
            "Processing %d page(s) for document %s (total tokens: %d).",
            len(pages),
            document_content.document_id,
            sum(token_counts),
        )

        page_data = list(zip(texts, token_counts))
        summaries = self.summarizer.summarize_pages(page_data)

        content_vectors = self._embed_batch(texts)
        summary_vectors = self._embed_batch(summaries)

        chunks_creates: List[ChunkCreate] = [
            ChunkCreate(
                chunk_id=uuid4(),
                document_id=document_content.document_id,
                page_number=page.page_number,
                token_count=token_counts[idx],
                content=page.text,
                content_vector=content_vectors[idx],
                summary=summaries[idx],
                summary_vector=summary_vectors[idx],
            )
            for idx, page in enumerate(pages)
        ]

        db_chunks = document_crud.create_chunks(self.db, chunks_creates)

        return len(db_chunks)


    # ---------------------------------------------------------------------------
    # Image processing
    # ---------------------------------------------------------------------------

    def _process_images(self, content: DocumentRawContent) -> int:
        """Describe, embed, store files, and persist image records. Returns count stored."""
        images = content.images

        if not images:
            log.info("No images to process for document %s.", content.document_id)
            return 0

        log.info(
            "Processing %d image(s) for document %s.",
            len(images),
            content.document_id,
        )

        descriptions = self._describe_images(images)
        vectors = self._embed_batch(descriptions)

        image_creates: List[ImageCreate] = []

        for idx, raw_image in enumerate(images):
            description = descriptions[idx]
            vector = vectors[idx]

            if not self._is_valid_image(description, vector, raw_image):
                continue

            storage_path = self._store_image(content.document_id, raw_image)

            image_creates.append(
                ImageCreate(
                    image_id=uuid4(),
                    document_id=content.document_id,
                    page_number=raw_image.page_number,
                    file_name=raw_image.file_name,
                    storage_path=storage_path,
                    description=description,
                    description_vector=vector,
                )
            )

        db_images = document_crud.create_images(self.db, image_creates)

        return len(db_images)


    # ---------------------------------------------------------------------------
    # Helper functions
    # ---------------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self.encoding.encode(text)) if text else 0

    def _embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed a batch of text strings."""
        return self.embedder.embed_texts(texts)

    def _describe_images(self, images) -> List[Optional[str]]:
        """Describe a batch of images."""
        pairs: List[Tuple[bytes, str]] = [
            (img.image_bytes, img.image_mime_type) for img in images
        ]
        return self.image_describer.describe_images(pairs)

    def _is_valid_image(self, description, vector, raw_image) -> bool:
        """Check if an image is valid."""
        if not description or vector is None:
            log.warning(
                "Skipping image '%s' (page %d) — missing description or embedding",
                raw_image.file_name,
                raw_image.page_number,
            )
            return False
        return True

    def _store_image(self, document_id, raw_image) -> str:
        """Store an image in the image storage."""
        path = self.image_storage.save_bytes(
            document_id=str(document_id),
            filename=raw_image.file_name,
            data=raw_image.image_bytes,
        )

        log.debug(
            "Stored image %s (page %d) at %s",
            raw_image.file_name,
            raw_image.page_number,
            path,
        )

        return path
