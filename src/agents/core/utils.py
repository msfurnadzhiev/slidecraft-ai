"""Constants for the LLM agents."""

from typing import Any

CHARS_PER_TOKEN = 4
TOKEN_SAFETY_MULTIPLIER = 1.2


def estimate_tokens(contents: Any) -> int:
    """Rough token count: ~4 chars per token, with safety margin."""
    return max(1, int(len(str(contents)) / CHARS_PER_TOKEN * TOKEN_SAFETY_MULTIPLIER))
