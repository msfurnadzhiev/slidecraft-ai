"""Prompts for image description generation."""

import base64
from typing import List, Tuple

from langchain_core.messages import HumanMessage

_SINGLE_INSTRUCTION = (
    "Describe this image in 1-2 concise sentences. "
    "Focus on the visual content, any text visible, "
    "and the overall purpose of the image. "
    "Return plain text only."
)

_BATCH_INSTRUCTION = (
    "Describe each of the {n} numbered images above "
    "in 1-2 concise sentences each. "
    "Focus on the visual content, any text visible, "
    "and the overall purpose of the image. "
    "Return plain text only, prefixed with the matching number "
    "in the format [N] description."
)


def _image_part(image_bytes: bytes, mime_type: str) -> dict:
    """Build a LangChain inline-image content block."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64}"},
    }


def build_single_prompt(image_info: Tuple[bytes, str]) -> list:
    """Return a single-message prompt for one image."""
    image_bytes, mime_type = image_info
    return [
        HumanMessage(content=[
            _image_part(image_bytes, mime_type),
            {"type": "text", "text": _SINGLE_INSTRUCTION},
        ])
    ]


def build_numbered_prompt(images: List[Tuple[bytes, str]]) -> list:
    """Return a single-message prompt for a batch of numbered images."""
    content: list = []
    for idx, (image_bytes, mime_type) in enumerate(images, 1):
        content.append({"type": "text", "text": f"[{idx}]"})
        content.append(_image_part(image_bytes, mime_type))

    content.append({
        "type": "text",
        "text": _BATCH_INSTRUCTION.format(n=len(images)),
    })

    return [HumanMessage(content=content)]
