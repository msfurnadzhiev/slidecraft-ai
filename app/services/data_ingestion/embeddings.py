"""Embedding service – converts a single text string into a vector."""

import json
import os
from typing import List
from urllib import error as url_error
from urllib import request as url_request

from app.utils.singleton import SingletonMeta

_DEFAULT_TEXT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_EMBEDDINGS_API_URL = "http://localhost:8080/v1"


def _embeddings_api_url() -> str:
    return os.getenv("EMBEDDINGS_API_URL", _DEFAULT_EMBEDDINGS_API_URL)

EmbeddingVector: type = List[float]


class Embedder(metaclass=SingletonMeta):
    """Thin wrapper around an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        text_model_name: str = _DEFAULT_TEXT_MODEL,
        api_url: str | None = None,
    ):
        self.model_name = text_model_name
        self.api_url = api_url if api_url is not None else _embeddings_api_url()

    def generate_embedding(self, text: str) -> EmbeddingVector:
        """Encode a single text string into a vector."""
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
            raise RuntimeError(
                f"Failed to reach embeddings API at {self.api_url}: {exc}"
            ) from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Embeddings API returned invalid JSON.") from exc

        if isinstance(parsed, dict) and isinstance(parsed.get("data"), list):
            return parsed["data"][0]["embedding"]

        if isinstance(parsed, list):
            if parsed and isinstance(parsed[0], dict) and "embedding" in parsed[0]:
                return parsed[0]["embedding"]
            if parsed and isinstance(parsed[0], list):
                return parsed[0]

        raise RuntimeError("Unexpected embeddings API response format.")
