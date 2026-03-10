"""LLM-based text summarization for ingestion enrichment."""

import re
from typing import List

from app.agents.client import GenAIClient
from app.utils.singleton import SingletonMeta

_BATCH_SIZE = 5


class Summarizer(metaclass=SingletonMeta):
    """Generate concise chunk summaries via Google GenAI in batches."""

    def __init__(self):
        self.genai = GenAIClient.get_instance()

    def summarize_texts(self, texts: List[str]) -> List[str]:
        """Summarize a list of texts in batches, returning one summary per text."""
        results: List[str] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            results.extend(self._summarize_batch(batch))
        return results

    def _summarize_batch(self, texts: List[str]) -> List[str]:
        if not texts:
            return []

        if len(texts) == 1:
            return [self._summarize_single(texts[0])]

        numbered = "\n\n".join(
            f"[{idx + 1}]\n{text}" for idx, text in enumerate(texts)
        )
        prompt = (
            f"Summarize each of the following {len(texts)} numbered texts "
            "in at most 2 concise sentences each. "
            "Preserve key facts, numbers, and terminology. "
            "Return plain text only, prefixed with the matching number "
            "in the format [N] summary.\n\n"
            f"{numbered}"
        )

        raw = self.genai.generate_content(prompt)
        return self._parse_numbered_response(raw, len(texts))

    def _summarize_single(self, text: str) -> str:
        if not text.strip():
            return ""

        prompt = (
            "Summarize the following text in at most 2 concise sentences. "
            "Preserve key facts, numbers, and terminology. "
            "Return plain text only.\n\n"
            f"Text:\n{text}"
        )
        return self.genai.generate_content(prompt)

    @staticmethod
    def _parse_numbered_response(raw: str, expected_count: int) -> List[str]:
        """Parse '[N] summary' lines from the LLM response."""
        pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\n\[|\Z)", re.DOTALL)
        matches = {int(m.group(1)): m.group(2).strip() for m in pattern.finditer(raw)}

        results: List[str] = []
        for idx in range(1, expected_count + 1):
            results.append(matches.get(idx, ""))
        return results

    # def describe_image(self, image_bytes: bytes, mime_type: str = "image/png") -> str:
    #     """Describe image content in one concise sentence (disabled)."""
    #     pass
