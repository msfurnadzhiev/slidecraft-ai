"""Parsing utilities for content-generation output."""

import json
import re

from src.schemas.presentation.slide import SlideContent, SlideStructure

_CODE_BLOCK_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _normalize_raw_output(raw: str | list) -> str:
    if isinstance(raw, list):
        parts: list[str] = []
        for c in raw:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict) and "text" in c:
                parts.append(c["text"])
        return "".join(parts)
    return raw if isinstance(raw, str) else str(raw)


def _strip_code_fence(raw: str) -> str:
    text = _normalize_raw_output(raw)
    match = _CODE_BLOCK_PATTERN.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_slide_content(raw_output: str | list, slide_structure: SlideStructure) -> SlideContent:
    """Parse and validate LLM output as SlideContent."""
    cleaned = _strip_code_fence(raw_output)
    if not cleaned:
        raise ValueError("Content agent returned empty output.")

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Content agent did not return valid JSON: {exc}") from exc

    payload["slide_number"] = slide_structure.slide_number
    payload["slide_type"] = slide_structure.slide_type.value
    payload["title"] = payload.get("title") or slide_structure.title
    payload["description"] = payload.get("description") or slide_structure.description

    return SlideContent.model_validate(payload)
