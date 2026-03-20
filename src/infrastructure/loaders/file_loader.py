"""Module for reading and processing PDF files.

Provides file loaders for extracting document content, text chunks, and metadata.
Includes a singleton FileLoader base class and a PDFLoader implementation.
"""

import os
from typing import TYPE_CHECKING, Dict, List


from src.schemas.document.document import DocumentContent, PageContent, ImageContent
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

    def extract_pages(self, file_path: str) -> List[PageContent]:
        """
        Extract pages from a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of PageContent objects.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """
        Extract embedded metadata from a file.

        Args:
            file_path: Path to the file.

        Returns:
            Dictionary of metadata key-value pairs.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def extract_images(self, file_path: str) -> List[ImageContent]:
        """
        Extract images from a file.

        Args:
            file_path: Path to the file.

        Returns:
            List of ImageCreate objects.
        """
        raise NotImplementedError("Subclasses must implement this method.")


