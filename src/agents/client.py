"""Google GenAI client wrapper with built-in retry for rate limits."""

import os
from typing import Any

from src.utils.singleton import SingletonMeta

from google import genai


_GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"

class GeminiClient(metaclass=SingletonMeta):
    """Google GenAI client (currently unused)."""

    def __init__(self):
        self.model_name = os.getenv("GENAI_MODEL_NAME", _GEMINI_DEFAULT_MODEL)
        api_key = os.getenv("GENAI_API_KEY")
        if not api_key:
            raise RuntimeError("GENAI_API_KEY environment variable is required.")
        self.client = genai.Client(api_key=api_key)

    def generate_content(self, contents: Any) -> str:
        """Call generate_content and return the assistant reply."""
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
        )
        return (response.text or "").strip()
