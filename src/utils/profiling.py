import logging
import time
from functools import wraps

log = logging.getLogger(__name__)

def trace_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            name = func.__qualname__
            log.debug(f"[TRACE] {name} executed in {elapsed:.6f}s")

    return wrapper