"""Module for reading and processing PDF files."""

import fitz
import os
import uuid
from typing import TYPE_CHECKING, Dict, List

from app.schemas.document import DocumentContent, PageContent
from app.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from app.db.models import Chunk

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

DocumentMetadata = Dict[str, str]


class FileLoader(metaclass=SingletonMeta):
    """Load a file and return a DocumentContent object."""
    
    def __init__(self):
        pass
    
    def load_file(self, file_path: str) -> DocumentContent:
        """Load a file and return a DocumentContent object."""
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_chunks_text(
        self, file_path: str, chunks: List["Chunk"]
    ) -> Dict[str, str]:
        """Re-extract text for a batch of chunks from a stored file using char offsets.

        Returns a dict mapping chunk_id -> extracted text.
        """
        raise NotImplementedError("Subclasses must implement this method.")

class PDFLoader(FileLoader):
    """Load a PDF file and return a DocumentContent object."""
    
    def __init__(self):
        pass
    
    def load_file(self, file_path: str) -> DocumentContent:
        """Load a PDF file and return a DocumentContent object."""
        filename = os.path.basename(file_path)
        document_id = str(uuid.uuid4())
        pages = []

        doc = fitz.open(file_path)
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            text = page.get_text("text").strip()
            pages.append(PageContent(
                page_number=page_index + 1,
                text=text,
                char_count=len(text)
            ))
        doc.close()

        return DocumentContent(
            document_id=document_id,
            file_name=filename,
            total_pages=len(pages),
            pages=pages
        )

    def extract_chunks_text(
        self, file_path: str, chunks: List["Chunk"]
    ) -> Dict[str, str]:
        """Open PDF once, group chunks by page, slice text by char offsets.

        Returns a dict mapping chunk_id -> extracted text.
        """
        if not chunks:
            return {}

        result: Dict[str, str] = {}
        by_page: Dict[int, List["Chunk"]] = {}
        for ch in chunks:
            by_page.setdefault(ch.page_number, []).append(ch)

        doc = fitz.open(file_path)
        try:
            for page_number, page_chunks in by_page.items():
                page_index = page_number - 1
                if page_index < 0 or page_index >= len(doc):
                    continue
                page = doc.load_page(page_index)
                page_text = page.get_text("text").strip()
                for ch in page_chunks:
                    start = max(ch.start_char_offset, 0)
                    end = min(ch.end_char_offset, len(page_text))
                    result[ch.chunk_id] = page_text[start:end]
        finally:
            doc.close()

        return result

    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract embedded metadata from a PDF and return a key-value dict.

        Only keys with non-empty string values are included.
        """
        result: DocumentMetadata = {}
        doc = fitz.open(file_path)
        try:
            meta = doc.metadata or {}
            for key in DOCUMENT_METADATA_KEYS:
                if key not in meta:
                    continue
                value = meta[key]
                if value is None or not str(value).strip():
                    continue
                result[key] = str(value).strip()
            return result
        finally:
            doc.close()