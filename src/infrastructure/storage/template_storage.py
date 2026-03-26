"""Template file storage for persisted .pptx files (e.g. Docker volume)."""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Final

from src.utils.singleton import SingletonABCMeta


class TemplateStorage(ABC, metaclass=SingletonABCMeta):
    """Abstract base class for .pptx template file storage backends."""

    @abstractmethod
    def save(self, source_path: str, storage_name: str) -> str:
        """Copy *source_path* into the storage directory under *storage_name*.

        Args:
            source_path: Absolute path to the source file (e.g. a temp file).
            storage_name: Target filename within the storage root
                          (e.g. ``"<uuid>.pptx"``).

        Returns:
            Relative storage path suitable for persisting in the database.
        """
        raise NotImplementedError

    @abstractmethod
    def get_absolute_path(self, storage_path: str) -> str:
        """Resolve a relative *storage_path* from the DB to an absolute path.

        Args:
            storage_path: Relative path as returned by :meth:`save`.

        Returns:
            Absolute path to the file on disk.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, storage_path: str) -> bool:
        """Remove the stored file.

        Args:
            storage_path: Relative path as stored in the database.

        Returns:
            True if the file existed and was deleted, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, storage_path: str) -> bool:
        """Check whether the stored file exists on disk.

        Args:
            storage_path: Relative path as stored in the database.
        """
        raise NotImplementedError


class LocalTemplateStorage(TemplateStorage):
    """Store .pptx template files on the local filesystem (Docker volume)."""

    _DEFAULT_BASE_DIR: Final[str] = "/data/templates"

    def __init__(self) -> None:
        base_dir = os.getenv("TEMPLATE_STORAGE_PATH", self._DEFAULT_BASE_DIR)
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, source_path: str, storage_name: str) -> str:
        """Copy *source_path* into the templates volume as *storage_name*.

        Returns:
            Relative path (just the filename) for DB storage.
        """
        destination = os.path.join(self.base_dir, storage_name)
        shutil.copy2(source_path, destination)
        return storage_name

    def get_absolute_path(self, storage_path: str) -> str:
        """Return the absolute path for the given relative *storage_path*."""
        return os.path.join(self.base_dir, storage_path)

    def delete(self, storage_path: str) -> bool:
        """Delete the stored template file."""
        path = os.path.join(self.base_dir, storage_path)
        if not os.path.isfile(path):
            return False
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    def exists(self, storage_path: str) -> bool:
        """Check whether the template file exists on disk."""
        return os.path.isfile(os.path.join(self.base_dir, storage_path))
