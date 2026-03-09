"""Module for reading and processing PDF files.

Provides file loaders for extracting document content, text chunks, and metadata.
Includes a singleton FileLoader base class and a PDFLoader implementation.
"""

import os
import uuid
from typing import TYPE_CHECKING, Dict, List

import fitz

from app.schemas.document import DocumentContent, PageContent
from app.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from app.db.models import Chunk

# List of metadata keys to extract from PDF
DOCUMENT_METADATA_KEYS = (
    "title",
    "author",
    "subject",
    "keywords",
    "creator",
    "producer",
    "creationDate",
    "modDate",
    "format",
    "encryption",
)

# Maps chunk_id (string) to extracted text (string)
ChunkTextMap: type = Dict[str, str]

# Maps page number to a list of chunks on that page
PageChunksMap: type = Dict[int, List["Chunk"]]

# Maps metadata key (string) to value (string)
DocumentMetadata: type = Dict[str, str]


class FileLoader(metaclass=SingletonMeta):
    """Abstract base class for file loaders."""

    def load_file(self, file_path: str) -> DocumentContent:
        """
        Load a file and return a DocumentContent object.

        Args:
            file_path: Path to the file to load.

        Returns:
            DocumentContent object representing the loaded document.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_chunks_text(
        self, file_path: str, chunks: List["Chunk"]
    ) -> ChunkTextMap:
        """
        Re-extract text for a batch of chunks from a stored file using char offsets.

        Args:
            file_path: Path to the file.
            chunks: List of Chunk objects with start and end character offsets.

        Returns:
            ChunkTextMap mapping chunk_id -> extracted text.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class PDFLoader(FileLoader):
    """Load a PDF file and extract document content, text chunks, and metadata."""

    def load_file(self, file_path: str) -> DocumentContent:
        """
        Load a PDF file and return a DocumentContent object.

        Args:
            file_path: Path to the PDF file.

        Returns:
            DocumentContent with page text and metadata.
        """
        filename = os.path.basename(file_path)
        document_id = str(uuid.uuid4())
        pages: List[PageContent] = []

        doc = fitz.open(file_path)
        try:
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                text = page.get_text("text").strip()
                pages.append(
                    PageContent(
                        page_number=page_index + 1,
                        text=text,
                        char_count=len(text),
                    )
                )
        finally:
            doc.close()

        return DocumentContent(
            document_id=document_id,
            file_name=filename,
            total_pages=len(pages),
            pages=pages,
        )

    def extract_chunks_text(
        self, file_path: str, chunks: List["Chunk"]
    ) -> Dict[str, str]:
        """Extract text for multiple chunks from a PDF using stored char offsets.

        Args:
            file_path: Path to the PDF file.
            chunks: List of Chunk objects.

        Returns:
            ChunkTextMap mapping chunk_id -> extracted text.
        """
        if not chunks:
            return {}

        result: ChunkTextMap = {}
        by_page: PageChunksMap = {}
        for chunk in chunks:
            by_page.setdefault(chunk.page_number, []).append(chunk)

        doc = fitz.open(file_path)
        try:
            for page_number, page_chunks in by_page.items():
                page_index = page_number - 1
                if page_index < 0 or page_index >= len(doc):
                    continue
                page = doc.load_page(page_index)
                page_text = page.get_text("text").strip()
                for chunk in page_chunks:
                    start = max(chunk.start_char_offset, 0)
                    end = min(chunk.end_char_offset, len(page_text))
                    result[chunk.chunk_id] = page_text[start:end]
        finally:
            doc.close()

        return result

    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract embedded metadata from a PDF.

        Only returns keys with non-empty string values.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary of metadata key-value pairs.
        """
        result: DocumentMetadata = {}
        doc = fitz.open(file_path)
        try:
            meta = doc.metadata or {}
            for key in DOCUMENT_METADATA_KEYS:
                value = meta.get(key)
                if value is None or not str(value).strip():
                    continue
                result[key] = str(value).strip()
            return result
        finally:
            doc.close()
