"""Prompts for image description generation."""

from typing import List, Tuple

from google.genai import types as genai_types


def build_single_prompt(image_info: Tuple[bytes, str]) -> list:
    """Return the prompt contents for a single image."""
    image_bytes, mime_type = image_info
    return [
        genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        "Describe this image in 1-2 concise sentences. "
        "Focus on the visual content, any text visible, "
        "and the overall purpose of the image. "
        "Return plain text only.",
    ]


def build_numbered_prompt(images: List[Tuple[bytes, str]]) -> list:
    """Return the prompt contents for multiple images with numbering."""
    contents: list = []
    for idx, (image_bytes, mime_type) in enumerate(images, 1):
        contents.append(f"[{idx}]")
        contents.append(genai_types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

    contents.append(
        f"Describe each of the {len(images)} numbered images above "
        "in 1-2 concise sentences each. "
        "Focus on the visual content, any text visible, "
        "and the overall purpose of the image. "
        "Return plain text only, prefixed with the matching number "
        "in the format [N] description."
    )
    return contents
