"""Module for reading and processing PDF files."""

import fitz
import os
import uuid
from typing import Dict

from app.schemas.document import DocumentContent, PageContent
from app.utils.singleton import SingletonMeta

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