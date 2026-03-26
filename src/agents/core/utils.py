"""Shared utilities and constants for LLM agents."""

import logging
import re
from typing import Any, List

CHARS_PER_TOKEN = 4
TOKEN_SAFETY_MULTIPLIER = 1.2

log = logging.getLogger(__name__)


def estimate_tokens(contents: Any) -> int:
    """Rough token count: ~4 chars per token, with safety margin."""
    return max(1, int(len(str(contents)) / CHARS_PER_TOKEN * TOKEN_SAFETY_MULTIPLIER))


def parse_numbered_response(raw: str, expected_count: int) -> List[str]:
    """Parse '[N] ...' numbered LLM responses into a list of strings.

    Returns a list of length *expected_count*.  Items the model omitted or
    mis-formatted are returned as empty strings.
    """
    pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)", re.DOTALL)

    matches = {
        int(m.group(1)): m.group(2).strip()
        for m in pattern.finditer(raw)
    }

    if not matches:
        log.warning("No numbered items parsed from LLM response")

    return [matches.get(i, "") for i in range(1, expected_count + 1)]
