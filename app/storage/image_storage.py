"""Image storage for files extracted from PDFs (e.g. per-page or embedded images)."""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Final

from app.utils.singleton import SingletonABCMeta


class ImageStorage(ABC, metaclass=SingletonABCMeta):
    """Abstract base class for image storage backends."""

    @abstractmethod
    def save(self, source_path: str, document_id: str, filename: str) -> str:
        """Persist an image file and return its relative storage path."""
        raise NotImplementedError

    @abstractmethod
    def save_bytes(self, document_id: str, filename: str, data: bytes) -> str:
        """Persist image bytes and return relative storage path."""
        raise NotImplementedError

    @abstractmethod
    def get_path(self, document_id: str, filename: str) -> str:
        """Return the absolute path for a stored image."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: str, filename: str) -> bool:
        """Delete a stored image if it exists."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, document_id: str, filename: str) -> bool:
        """Check whether a stored image exists."""
        raise NotImplementedError


class LocalImageStorage(ImageStorage):
    """Store images on the local filesystem under a fixed base directory (Docker volume)."""

    _DEFAULT_BASE_DIR: Final[str] = "/data/images"

    def __init__(self) -> None:
        self.base_dir = self._DEFAULT_BASE_DIR
        self._ensure_base_dir()

    def _ensure_base_dir(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)

    def _relative_path(self, document_id: str, filename: str) -> str:
        """Return a portable relative path (forward slashes) for DB storage."""
        return f"{document_id}/{filename}"

    def _build_absolute_path(self, document_id: str, filename: str) -> str:
        return os.path.join(self.base_dir, document_id, filename)

    def save(self, source_path: str, document_id: str, filename: str) -> str:
        """Copy the image into the storage directory (volume).

        Returns a relative path of the form `{document_id}/{filename}` for storage in the database.
        """
        destination_path = self._build_absolute_path(document_id, filename)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        shutil.copy2(source_path, destination_path)

        return self._relative_path(document_id, filename)

    def save_bytes(self, document_id: str, filename: str, data: bytes) -> str:
        """Write image bytes to the volume (e.g. from PDF image extraction)."""
        destination_path = self._build_absolute_path(document_id, filename)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        with open(destination_path, "wb") as f:
            f.write(data)

        return self._relative_path(document_id, filename)

    def get_path(self, document_id: str, filename: str) -> str:
        """Return the absolute path for the given document and image filename."""
        return self._build_absolute_path(document_id, filename)

    def get_absolute_path(self, storage_path: str) -> str:
        """Resolve a relative storage_path from the DB to an absolute path on the volume."""
        normalized = storage_path.replace("\\", "/").strip("/")
        return os.path.join(self.base_dir, normalized)

    def delete(self, document_id: str, filename: str) -> bool:
        """Delete the stored image if it exists. Removes empty document subdirectory."""
        path = self._build_absolute_path(document_id, filename)
        if not os.path.isfile(path):
            return False

        try:
            os.remove(path)
            doc_dir = os.path.dirname(path)
            if os.path.isdir(doc_dir) and not os.listdir(doc_dir):
                os.rmdir(doc_dir)
            return True
        except OSError:
            return False

    def exists(self, document_id: str, filename: str) -> bool:
        """Check whether the stored image exists."""
        path = self._build_absolute_path(document_id, filename)
        return os.path.exists(path)
