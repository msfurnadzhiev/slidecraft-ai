"""File storage abstractions for persisting original documents."""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Final

from app.utils.singleton import SingletonABCMeta


class FileStorage(ABC, metaclass=SingletonABCMeta):
    """Abstract base class for file storage backends."""

    @abstractmethod
    def relative_path(self, document_id: str, filename: str) -> str:
        """Return the relative storage path for a document file (without writing)."""
        raise NotImplementedError

    @abstractmethod
    def save(self, source_path: str, document_id: str, filename: str) -> str:
        """Persist a file and return its relative storage path."""
        raise NotImplementedError

    @abstractmethod
    def save_bytes(self, document_id: str, filename: str, data: bytes) -> str:
        """Persist bytes and return relative storage path."""
        raise NotImplementedError

    @abstractmethod
    def get_path(self, document_id: str, filename: str) -> str:
        """Return the absolute path for a stored file."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, document_id: str, filename: str) -> bool:
        """Delete a stored file if it exists."""
        raise NotImplementedError

    @abstractmethod
    def exists(self, document_id: str, filename: str) -> bool:
        """Check whether a stored file exists."""
        raise NotImplementedError


class LocalFileStorage(FileStorage):
    """Store files on the local filesystem under a base directory."""

    _DEFAULT_BASE_DIR: Final[str] = "/data/documents"

    def __init__(self) -> None:
        base_dir_from_env = os.getenv("DOCUMENT_STORAGE_PATH")
        self.base_dir = (base_dir_from_env or self._DEFAULT_BASE_DIR).rstrip("/")
        self._ensure_base_dir()

    def _ensure_base_dir(self) -> None:
        os.makedirs(self.base_dir, exist_ok=True)

    def _relative_path(self, document_id: str, filename: str) -> str:
        """Return a portable relative path (forward slashes) for DB storage."""
        return f"{document_id}/{filename}"

    def _build_absolute_path(self, document_id: str, filename: str) -> str:
        return os.path.join(self.base_dir, document_id, filename)

    def relative_path(self, document_id: str, filename: str) -> str:
        """Return the relative path for DB storage (same format as save)."""
        return self._relative_path(document_id, filename)

    def save(self, source_path: str, document_id: str, filename: str) -> str:
        """Copy the source file into the storage directory (volume).

        Returns a relative path of the form `{document_id}/{filename}` (always
        forward slashes) for storage in the database.
        """
        destination_path = self._build_absolute_path(document_id, filename)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        shutil.copy2(source_path, destination_path)

        return self._relative_path(document_id, filename)

    def save_bytes(self, document_id: str, filename: str, data: bytes) -> str:
        destination_path = self._build_absolute_path(document_id, filename)
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            f.write(data)
        return self._relative_path(document_id, filename)

    def get_path(self, document_id: str, filename: str) -> str:
        """Return the absolute path for the given document and filename."""
        return self._build_absolute_path(document_id, filename)

    def get_absolute_path(self, storage_path: str) -> str:
        """Resolve a relative storage_path from the DB to an absolute path on the volume."""
        # Normalize in case DB has backslashes from another environment
        normalized = storage_path.replace("\\", "/").strip("/")
        return os.path.join(self.base_dir, normalized)

    def delete(self, document_id: str, filename: str) -> bool:
        """Delete the stored file if it exists.

        Removes the file from the volume and the document subdirectory if empty.
        Returns True if a file was deleted, False otherwise.
        """
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
        """Check whether the stored file exists."""
        path = self._build_absolute_path(document_id, filename)
        return os.path.exists(path)

