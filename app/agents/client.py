"""Google GenAI client wrapper with built-in retry for rate limits."""

import logging
import os
import time
from typing import Any

from app.utils.singleton import SingletonMeta

from google import genai
from google.genai import errors as genai_errors

_DEFAULT_MODEL = "gemini-2.0-flash"
_MAX_RETRIES = 5
_BASE_DELAY = 25.0

log = logging.getLogger(__name__)


class GenAIClient(metaclass=SingletonMeta):
    """Build and expose Google GenAI client with automatic 429 retry."""

    def __init__(self):
        self.model_name = os.getenv("GENAI_MODEL_NAME", _DEFAULT_MODEL)
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GENAI_API_KEY environment variable is required. "
            )
        self.client = genai.Client(api_key=api_key)

    def generate_content(self, contents: Any) -> str:
        """Call generate_content with automatic retry on 429 rate-limit errors."""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                )
                return (response.text or "").strip()
            except genai_errors.ClientError as exc:
                if exc.code != 429 or attempt == _MAX_RETRIES:
                    raise
                delay = _BASE_DELAY * attempt
                log.warning(
                    "Rate limited (429), retrying in %.0fs (attempt %d/%d)",
                    delay, attempt, _MAX_RETRIES,
                )
                time.sleep(delay)

        raise RuntimeError("Unreachable")
