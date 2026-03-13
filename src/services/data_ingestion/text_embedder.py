"""Embedding service – converts a single text string into a vector."""

import json
import logging
import os
from typing import List
from urllib import error as url_error
from urllib import request as url_request

from src.utils.singleton import SingletonMeta
from src.utils.profiling import trace_runtime

_DEFAULT_TEXT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_EMBEDDINGS_API_URL = "http://localhost:8080/v1"

log = logging.getLogger(__name__)

def _embeddings_api_url() -> str:
    return os.getenv("EMBEDDINGS_API_URL", _DEFAULT_EMBEDDINGS_API_URL)

EmbeddingVector: type = List[float]

class TextEmbedder(metaclass=SingletonMeta):
    """Thin wrapper around an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        text_model_name: str = _DEFAULT_TEXT_MODEL,
        api_url: str | None = None,
    ):
        self.model_name = text_model_name
        self.api_url = api_url if api_url is not None else _embeddings_api_url()
        log.info("TextEmbedder initialised (model=%s, api=%s).", self.model_name, self.api_url)

    @trace_runtime
    def generate_embedding(self, text: str) -> EmbeddingVector:
        """Encode a single text string into a vector."""
        text_preview = text[:80].replace("\n", " ")
        log.debug("Requesting embedding for text (len=%d): '%s…'", len(text), text_preview)

        payload = {
            "model": self.model_name,
            "input": [text],
        }
        req = url_request.Request(
            url=f"{self.api_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with url_request.urlopen(req, timeout=30) as response:
                raw = response.read().decode("utf-8")
        except url_error.URLError as exc:
            log.error("Embeddings API unreachable at %s: %s", self.api_url, exc)
            raise RuntimeError(
                f"Failed to reach embeddings API at {self.api_url}: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error("Embeddings API returned invalid JSON.")
            raise RuntimeError("Embeddings API returned invalid JSON.") from exc

        if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            vec = parsed["data"][0]["embedding"]
            log.debug("Embedding received (dim=%d).", len(vec))
            return vec

        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict) and "embedding" in parsed[0]:
                vec = parsed[0]["embedding"]
                log.debug("Embedding received (dim=%d).", len(vec))
                return vec
            if parsed and isinstance(parsed[0], list):
                log.debug("Embedding received (dim=%d).", len(parsed[0]))
                return parsed[0]

        log.error("Unexpected embeddings API response format: %s", str(parsed)[:200])
        raise RuntimeError("Unexpected embeddings API response format.")
