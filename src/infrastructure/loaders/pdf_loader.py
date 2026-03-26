"""Module for reading and processing PDF files."""

import os
from uuid import uuid4
from typing import List, Dict, TYPE_CHECKING

import fitz

from src.schemas.document import (
    DocumentRawContent,
    ImageRawContent,
    ChunkRawContent,
)
from src.utils.singleton import SingletonMeta

if TYPE_CHECKING:
    from src.db.models import Chunk

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

class PDFLoader(metaclass=SingletonMeta):
    """Load a PDF file and extract document content, text chunks, and metadata."""

    def load_file(self, file_path: str) -> DocumentRawContent:
        """Load a PDF file and return a DocumentRawContent object.

        Args:
            file_path: Path to the PDF file.

        Returns:
            DocumentRawContent with page text and metadata.
        """
        filename = os.path.basename(file_path)
        document_id = uuid4()
        
        pages = self.extract_pages(file_path)
        images = self.extract_images(file_path)
        metadata = self.extract_metadata(file_path)
        
        return DocumentRawContent(
            document_id=document_id,
            file_name=filename,
            total_pages=len(pages),
            pages=pages,
            images=images,
            metadata=metadata,
        )

    def extract_pages(self, file_path: str) -> List[ChunkRawContent]:
        """Extract pages from a PDF file."""
        pages: List[ChunkRawContent] = []
        doc = fitz.open(file_path)
        try:
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                text = page.get_text("text").strip()
                pages.append(
                    ChunkRawContent(
                        page_number=page_index + 1,
                        text=text,
                        char_count=len(text),
                    )
                )
        finally:
            doc.close()
        return pages

    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract embedded metadata from a PDF.
        
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

    def extract_images(self, file_path: str) -> List[ImageRawContent]:
        """Extract images from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            List of ImageRawContent objects.
        """
        images: List[ImageRawContent] = []
        doc = fitz.open(file_path)

        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                for idx, image in enumerate(image_list):
                    xref = image[0]
                    try:
                        base = doc.extract_image(xref)
                    except Exception:
                        continue

                    raw_bytes: bytes = base["image"]
                    page_number = page_index + 1
                    ext = base.get("ext", "png")
                    mime_type = f"image/{ext.lower()}"
                    file_name = f"image_{page_number}_{idx}.{ext.lower()}"

                    images.append(
                        ImageRawContent(
                            page_number=page_number,
                            image_bytes=raw_bytes,
                            image_mime_type=mime_type,
                            file_name=file_name,
                        )
                    )
        finally:
            doc.close()

        return images
