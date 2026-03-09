"""The module provides a thread-safe Singleton metaclass."""

from abc import ABCMeta
from threading import Lock
from typing import ClassVar


class SingletonMeta(type):
    """Thread-safe Singleton metaclass.

    Ensures only one instance of each class exists.
    """

    _instances: ClassVar[dict[type, object]] = {}
    _lock: ClassVar[Lock] = Lock()

    def __call__(cls, *args, **kwargs):
        """Create a new instance of the class.

        Double-checked locking pattern for thread safety.
        """
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]

    def get_instance(cls, *args, **kwargs):
        """Return the singleton instance, creating it if necessary."""
        return cls(*args, **kwargs)


class SingletonABCMeta(SingletonMeta, ABCMeta):
    """Metaclass combining Singleton and ABC. Use for abstract base classes that must be singletons."""

    pass
