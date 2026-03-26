"""Storage for generated .pptx presentation files."""

import os
import shutil
from abc import ABC, abstractmethod
from typing import Final

from src.utils.singleton import SingletonABCMeta


class PresentationStorage(ABC, metaclass=SingletonABCMeta):
    """Abstract base class for generated presentation file storage backends."""

    @abstractmethod
    def save(self, source_path: str, storage_name: str) -> str:
        """Copy *source_path* into the storage directory under *storage_name*.

        Args:
            source_path: Absolute path to the source file.
            storage_name: Target filename within the storage root (e.g. ``"<uuid>.pptx"``).

        Returns:
            Relative storage path suitable for persisting in the database or returning to callers.
        """
        raise NotImplementedError

    @abstractmethod
    def get_absolute_path(self, storage_path: str) -> str:
        """Resolve a relative *storage_path* to an absolute filesystem path.

        Args:
            storage_path: Relative path as returned by :meth:`save`.

        Returns:
            Absolute path to the file on disk.
        """
        raise NotImplementedError

    @abstractmethod
    def delete(self, storage_path: str) -> bool:
        """Remove a stored presentation file.

        Args:
            storage_path: Relative path as returned by :meth:`save`.

        Returns:
            True if the file existed and was deleted, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def exists(self, storage_path: str) -> bool:
        """Check whether a stored presentation file exists on disk."""
        raise NotImplementedError


class LocalPresentationStorage(PresentationStorage):
    """Store generated .pptx files on the local filesystem (Docker volume)."""

    _DEFAULT_BASE_DIR: Final[str] = "/data/presentations"

    def __init__(self) -> None:
        self.base_dir = os.getenv("PRESENTATION_STORAGE_PATH", self._DEFAULT_BASE_DIR)
        os.makedirs(self.base_dir, exist_ok=True)

    def save(self, source_path: str, storage_name: str) -> str:
        """Copy *source_path* into the presentations volume as *storage_name*.

        Returns:
            Relative path (just the filename) for downstream use.
        """
        destination = os.path.join(self.base_dir, storage_name)
        shutil.copy2(source_path, destination)
        return storage_name

    def get_absolute_path(self, storage_path: str) -> str:
        """Return the absolute path for the given relative *storage_path*."""
        return os.path.join(self.base_dir, storage_path)

    def delete(self, storage_path: str) -> bool:
        """Delete a stored presentation file."""
        path = os.path.join(self.base_dir, storage_path)
        if not os.path.isfile(path):
            return False
        try:
            os.remove(path)
            return True
        except OSError:
            return False

    def exists(self, storage_path: str) -> bool:
        """Check whether the presentation file exists on disk."""
        return os.path.isfile(os.path.join(self.base_dir, storage_path))
