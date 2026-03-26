"""Gemini-based text embedding using the Google Generative AI API."""

import logging
import os
import random
import re
import threading
import time
from collections import deque
from typing import List, Optional, Tuple

import tiktoken
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.utils.singleton import SingletonMeta
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)

_EMBEDDING_MODEL = "models/gemini-embedding-001"
_EMBEDDING_DIMENSIONS = 768
_MAX_RETRIES = 6

# API quota defaults (free-tier limits; override via env vars)
_DEFAULT_RPM = 100
_DEFAULT_TPM = 30_000
_DEFAULT_RPD = 1_000

# Batch sizing — stay under the TPM limit with a small headroom buffer
_MAX_TOKENS_PER_BATCH = 28_000
_MAX_TEXTS_PER_BATCH = 250

# Warn when this fraction of the daily RPD budget has been consumed
_RPD_WARN_THRESHOLD = 0.80

EmbeddingVector = List[float]

_RETRY_DELAY_RE = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


class EmbeddingRateLimiter:
    """Thread-safe sliding-window rate limiter that enforces RPM, TPM, and RPD.

    Each call to ``acquire(token_count)`` records a request and its token cost
    in three independent sliding windows (1-minute for RPM/TPM, 24-hour for RPD)
    and blocks until all three limits allow the request to proceed.
    """

    def __init__(self, rpm: int, tpm: int, rpd: int) -> None:
        self._rpm = rpm
        self._tpm = tpm
        self._rpd = rpd
        self._lock = threading.Lock()
        self._min_requests: deque[float] = deque()
        self._min_tokens: deque[Tuple[float, int]] = deque()
        self._day_requests: deque[float] = deque()

    def acquire(self, token_count: int) -> None:
        """Block until a request costing ``token_count`` tokens can proceed."""
        while True:
            with self._lock:
                now = time.monotonic()
                self._evict(now)

                rpm_ok = len(self._min_requests) < self._rpm
                tpm_ok = self._tokens_in_window() + token_count <= self._tpm
                rpd_ok = len(self._day_requests) < self._rpd

                if rpm_ok and tpm_ok and rpd_ok:
                    self._min_requests.append(now)
                    self._min_tokens.append((now, token_count))
                    self._day_requests.append(now)

                    rpd_used = len(self._day_requests) / self._rpd
                    if rpd_used >= _RPD_WARN_THRESHOLD:
                        log.warning(
                            "Embedding RPD budget %.0f%% consumed (%d/%d requests today).",
                            rpd_used * 100,
                            len(self._day_requests),
                            self._rpd,
                        )
                    return

                wait = self._compute_wait(now, token_count)

            log.debug(
                "Embedding rate limit (%s) — sleeping %.2fs",
                self._limit_reason(token_count),
                wait,
            )
            time.sleep(max(wait, 0.1))

    # -------------------------------------------------------------------------

    def _evict(self, now: float) -> None:
        while self._min_requests and now - self._min_requests[0] >= 60.0:
            self._min_requests.popleft()
        while self._min_tokens and now - self._min_tokens[0][0] >= 60.0:
            self._min_tokens.popleft()
        while self._day_requests and now - self._day_requests[0] >= 86_400.0:
            self._day_requests.popleft()

    def _tokens_in_window(self) -> int:
        return sum(t for _, t in self._min_tokens)

    def _compute_wait(self, now: float, token_count: int) -> float:
        waits: List[float] = []
        if len(self._min_requests) >= self._rpm and self._min_requests:
            waits.append(60.0 - (now - self._min_requests[0]) + 0.1)
        if self._tokens_in_window() + token_count > self._tpm and self._min_tokens:
            waits.append(60.0 - (now - self._min_tokens[0][0]) + 0.1)
        if len(self._day_requests) >= self._rpd and self._day_requests:
            waits.append(86_400.0 - (now - self._day_requests[0]) + 0.1)
        return max(waits) if waits else 0.1

    def _limit_reason(self, token_count: int) -> str:
        reasons = []
        if len(self._min_requests) >= self._rpm:
            reasons.append("RPM")
        if self._tokens_in_window() + token_count > self._tpm:
            reasons.append("TPM")
        if len(self._day_requests) >= self._rpd:
            reasons.append("RPD")
        return "+".join(reasons) or "unknown"


class TextEmbedder(metaclass=SingletonMeta):
    """Singleton wrapper around Gemini embeddings API.

    ``embed_texts()`` groups texts into token-aware batches (up to
    ``_MAX_TOKENS_PER_BATCH`` tokens each) so that one API call embeds as many
    texts as possible, minimising RPD consumption while staying within the
    30 K TPM and 100 RPM per-minute limits.
    """

    def __init__(self) -> None:
        api_key = self._require_api_key()

        self._client = GoogleGenerativeAIEmbeddings(
            model=_EMBEDDING_MODEL,
            google_api_key=api_key,
            output_dimensionality=_EMBEDDING_DIMENSIONS,
        )

        rpm = int(os.getenv("EMBEDDING_RPM_LIMIT", str(_DEFAULT_RPM)))
        tpm = int(os.getenv("EMBEDDING_TPM_LIMIT", str(_DEFAULT_TPM)))
        rpd = int(os.getenv("EMBEDDING_RPD_LIMIT", str(_DEFAULT_RPD)))

        self._rate_limiter = EmbeddingRateLimiter(rpm=rpm, tpm=tpm, rpd=rpd)
        self._encoding = tiktoken.get_encoding("cl100k_base")

        log.info(
            "TextEmbedder initialised (model=%s, dim=%d, rpm=%d, tpm=%d, rpd=%d)",
            _EMBEDDING_MODEL,
            _EMBEDDING_DIMENSIONS,
            rpm,
            tpm,
            rpd,
        )

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    @trace_runtime
    def generate_embedding(self, text: str) -> EmbeddingVector:
        """Embed a single text (used for search queries)."""
        log.debug("Embedding request (len=%d chars)", len(text))
        token_count = self._count_tokens(text)

        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limiter.acquire(token_count)
                return self._client.embed_query(text)

            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt == _MAX_RETRIES - 1:
                    raise

                delay = self._compute_retry_delay(exc, attempt)
                log.warning(
                    "429 rate limit (attempt %d/%d) — retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Embedding failed after max retries")

    def embed_texts(self, texts: List[str]) -> List[Optional[EmbeddingVector]]:
        """Embed a list of texts using token-aware batching.

        Non-empty texts are packed into batches of up to ``_MAX_TOKENS_PER_BATCH``
        tokens (and ``_MAX_TEXTS_PER_BATCH`` count). The rate limiter is acquired
        once per batch, consuming one RPM and one RPD unit per batch call.
        Empty / whitespace-only texts receive ``None`` in the output.
        """
        results: List[Optional[EmbeddingVector]] = [None] * len(texts)

        indexed: List[Tuple[int, str, int]] = [
            (i, t, self._count_tokens(t))
            for i, t in enumerate(texts)
            if t and t.strip()
        ]

        if not indexed:
            return results

        batches = self._build_batches(indexed)
        total_tokens = sum(tc for _, _, tc in indexed)

        log.info(
            "Embedding %d text(s) in %d batch(es) (~%d tokens total).",
            len(indexed),
            len(batches),
            total_tokens,
        )

        for batch_no, batch in enumerate(batches, 1):
            indices = [i for i, _, _ in batch]
            batch_texts = [t for _, t, _ in batch]
            batch_tokens = sum(tc for _, _, tc in batch)

            log.debug(
                "Embedding batch %d/%d — %d text(s), ~%d tokens.",
                batch_no,
                len(batches),
                len(batch_texts),
                batch_tokens,
            )

            vectors = self._embed_batch_with_retry(
                batch_no, len(batches), batch_texts, batch_tokens
            )

            for idx, vec in zip(indices, vectors):
                results[idx] = vec

        return results

    # -------------------------------------------------------------------------
    # Batch building
    # -------------------------------------------------------------------------

    def _build_batches(
        self,
        indexed: List[Tuple[int, str, int]],
    ) -> List[List[Tuple[int, str, int]]]:
        """Greedily pack items into batches within token and count limits."""
        batches: List[List[Tuple[int, str, int]]] = []
        current: List[Tuple[int, str, int]] = []
        current_tokens = 0

        for item in indexed:
            _, _, tc = item
            if current and (
                current_tokens + tc > _MAX_TOKENS_PER_BATCH
                or len(current) >= _MAX_TEXTS_PER_BATCH
            ):
                batches.append(current)
                current = []
                current_tokens = 0

            current.append(item)
            current_tokens += tc

        if current:
            batches.append(current)

        return batches

    # -------------------------------------------------------------------------
    # API call with retry
    # -------------------------------------------------------------------------

    def _embed_batch_with_retry(
        self,
        batch_no: int,
        total_batches: int,
        texts: List[str],
        token_count: int,
    ) -> List[EmbeddingVector]:
        for attempt in range(_MAX_RETRIES):
            try:
                self._rate_limiter.acquire(token_count)
                return self._client.embed_documents(texts)

            except Exception as exc:
                if not self._is_rate_limit_error(exc) or attempt == _MAX_RETRIES - 1:
                    raise

                delay = self._compute_retry_delay(exc, attempt)
                log.warning(
                    "Batch %d/%d: 429 rate limit (attempt %d/%d) — retrying in %.1fs",
                    batch_no,
                    total_batches,
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"Embedding batch {batch_no}/{total_batches} failed after max retries"
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        return len(self._encoding.encode(text)) if text else 0

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        msg = str(exc)
        return "429" in msg or "RESOURCE_EXHAUSTED" in msg

    def _compute_retry_delay(self, exc: Exception, attempt: int) -> float:
        msg = str(exc)
        match = _RETRY_DELAY_RE.search(msg)
        if match:
            return float(match.group(1)) + random.uniform(1.0, 3.0)
        return (2 ** attempt) * 5 + random.uniform(0.0, 5.0)

    @staticmethod
    def _require_api_key() -> str:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY environment variable is required.")
        return api_key
