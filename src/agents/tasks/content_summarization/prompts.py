"""Prompts for content summarization."""

from typing import List


def build_single_prompt(text: str) -> str:
    """Return a prompt string for a single-text summary."""
    return (
        "Summarize the following text in at most 2 concise sentences. "
        "Preserve key facts, numbers, and terminology. "
        "Return plain text only.\n\n"
        f"Text:\n{text}"
    )


def build_numbered_prompt(texts: List[str]) -> str:
    """Build a single prompt for multiple texts with numbering."""
    numbered = "\n\n".join(f"[{i + 1}]\n{text}" for i, text in enumerate(texts))
    return (
        f"Summarize each of the following {len(texts)} numbered texts "
        "in at most 2 concise sentences each. "
        "Preserve key facts, numbers, and terminology. "
        "Return plain text only, prefixed with the matching number "
        "in the format [N] summary.\n\n"
        f"{numbered}"
    )
