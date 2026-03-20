"""Module for embedding text into a vector.

Provides a singleton TextEmbedder class for generating embeddings from text.
"""

import json
import logging
import os
from typing import List
from urllib import error as url_error
from urllib import request as url_request

from src.utils.singleton import SingletonMeta
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)

# Default text model and embeddings API URL
_DEFAULT_TEXT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_EMBEDDINGS_API_URL = "http://localhost:8080/v1"

def _text_model_name() -> str:
    return os.getenv("TEXT_MODEL_NAME", _DEFAULT_TEXT_MODEL)

def _embeddings_api_url() -> str:
    return os.getenv("EMBEDDINGS_API_URL", _DEFAULT_EMBEDDINGS_API_URL)

# Type alias for embedding vectors
EmbeddingVector: type = List[float]

class TextEmbedder(metaclass=SingletonMeta):
    """Thin wrapper around an OpenAI-compatible embeddings endpoint."""

    def __init__(self):
        """Initialize the TextEmbedder with the default model and API URL."""
        self.model_name = _text_model_name()
        self.api_url = _embeddings_api_url()
        log.info("TextEmbedder initialised (model=%s, api=%s).", self.model_name, self.api_url)

    def _build_request(self, text: str) -> url_request.Request:
        """Build a POST request to the embeddings API."""
        payload = {"model": self.model_name, "input": [text]}
        return url_request.Request(
            url=f"{self.api_url}/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _fetch_embedding_response(self, req: url_request.Request) -> str:
        """Send the request to the embeddings API and return the raw response."""
        try:
            with url_request.urlopen(req, timeout=30) as response:
                return response.read().decode("utf-8")
        except url_error.URLError as exc:
            raise RuntimeError(f"Failed to reach embeddings API at {self.api_url}: {exc}") from exc

    def _parse_embedding_response(self, raw: str) -> EmbeddingVector:
        """Parse the raw JSON response from the embeddings API into a vector."""
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Embeddings API returned invalid JSON.") from exc

        # Handle dict-based response
        if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            vec = parsed["data"][0]["embedding"]
            return vec

        # Handle list-based response
        if isinstance(parsed, list) and parsed:
            if isinstance(parsed[0], dict) and "embedding" in parsed[0]:
                vec = parsed[0]["embedding"]
                return vec
            if isinstance(parsed[0], list):
                return parsed[0]

        raise RuntimeError("Unexpected embeddings API response format.")

    @trace_runtime
    def generate_embedding(self, text: str) -> EmbeddingVector:
        """Encode a single text string into a vector."""
        text_preview = text[:80].replace("\n", " ")

        log.debug("Requesting embedding for text (len=%d): '%s…'", len(text), text_preview)

        req = self._build_request(text)
        raw_response = self._fetch_embedding_response(req)
        return self._parse_embedding_response(raw_response)