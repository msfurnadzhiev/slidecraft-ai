import logging
import random
import re
import time
from typing import Any, Callable, TypeVar

_429_MAX_RETRIES = 5

T = TypeVar("T") # Type variable for the return type of the function

log = logging.getLogger(__name__)

class RetryPolicy:
    """Generic retry policy for LLM calls."""

    def __init__(
        self,
        max_retries: int = _429_MAX_RETRIES,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.jitter = jitter

    def should_retry(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            re.search(pattern, text)
            for pattern in (
                r"\b429\b",
                r"rate limit",
                r"resource exhausted",
                r"too many requests",
            )
        )

    def parse_retry_delay(self, error_text: str) -> float | None:
        """Extract server-provided delay if available."""
        text = error_text.lower()

        match = re.search(r"(\d+(?:\.\d+)?)\s*(ms|s|seconds?)", text)
        if not match:
            return None

        value = float(match.group(1))
        unit = match.group(2)

        if unit.startswith("ms"):
            value /= 1000.0

        return value + 2.0

    def compute_delay(self, attempt: int, exc: Exception) -> float:
        """Compute retry delay using server hint or exponential backoff."""
        server_delay = self.parse_retry_delay(str(exc))

        if server_delay is not None:
            delay = server_delay
        else:
            delay = self.base_delay * (2 ** (attempt - 1))

        if self.jitter:
            delay += random.uniform(0, 1)

        return min(delay, self.max_delay)

    def execute(
        self,
        fn: Callable[..., T],
        *args: Any,
        on_retry: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> T:
        """Execute function with retry logic.

        Args:
            fn: The function to execute.
            *args: Positional arguments forwarded to ``fn``.
            on_retry: Optional callback invoked after each wait period, before
                the next attempt.  Use this to reset external state (e.g. a
                rate-limiter window) that may have drifted due to the 429 error.
            **kwargs: Keyword arguments forwarded to ``fn``.
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                if not self.should_retry(exc) or attempt == self.max_retries:
                    raise

                delay = self.compute_delay(attempt, exc)

                log.warning(
                    "Retry in %.1fs (attempt %d/%d): %s",
                    delay,
                    attempt,
                    self.max_retries,
                    exc,
                )

                time.sleep(delay)

                if on_retry is not None:
                    on_retry()

        raise RuntimeError("Unreachable")
