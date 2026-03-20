"""RPM + TPM rate limiter for LLM calls."""

import logging
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.agents.core.agent_models import ModelRateLimits

WINDOW_SECONDS = 60.0

log = logging.getLogger(__name__)

class RateLimiter:
    """Fixed-window rate limiter for requests per minute and tokens per minute."""

    def __init__(self, rate_limits: "ModelRateLimits") -> None:
        self.rate_limits = rate_limits
        self._window_start = time.time()
        self._request_count = 0
        self._token_count = 0
        self._lock = threading.Lock()

    def _reset_if_expired(self) -> None:
        now = time.time()

        if now - self._window_start >= WINDOW_SECONDS:
            self._window_start = now
            self._request_count = 0
            self._token_count = 0

    def acquire(self, estimated_tokens: int) -> None:
        """Block until request fits within RPM and TPM limits.

        Also available as ``wait_if_needed`` for backward compatibility.
        """

        while True:
            with self._lock:
                self._reset_if_expired()

                if (
                    self._request_count < self.rate_limits.rpm
                    and self._token_count + estimated_tokens <= self.rate_limits.tpm
                ):
                    return

                sleep_sec = WINDOW_SECONDS - (
                    time.time() - self._window_start
                ) + 1

                log.debug(
                    "Rate limit reached (requests=%d/%d, tokens=%d/%d). Waiting %.0fs.",
                    self._request_count,
                    self.rate_limits.rpm,
                    self._token_count,
                    self.rate_limits.tpm,
                    sleep_sec,
                )

            time.sleep(max(sleep_sec, 1))

    def record(self, tokens_used: int) -> None:
        """Record completed request."""

        with self._lock:
            self._reset_if_expired()

            self._request_count += 1
            self._token_count += tokens_used

    def force_reset(self) -> None:
        """Force reset after a 429 wait."""

        with self._lock:
            self._window_start = time.time()
            self._request_count = 0
            self._token_count = 0

    wait_if_needed = acquire

    @contextmanager
    def limit(self, estimated_tokens: int):
        """
        Context manager wrapper.

        Usage:
            with limiter.limit(estimated_tokens):
                call_llm()
        """

        self.acquire(estimated_tokens)

        try:
            yield
        finally:
            self.record(estimated_tokens)