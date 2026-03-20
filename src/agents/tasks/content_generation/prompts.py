"""Prompts for content-generation workflow."""

from typing import List

from src.schemas.document.chunk import ChunkSearchResult
from src.schemas.document.image import ImageSearchResult
from src.schemas.presentation.slide import SlideStructure


SYNTHESIS_SYSTEM_PROMPT = (
    "You are a presentation content synthesizer working with a large document (potentially hundreds of pages). "
    "You receive a numbered list of pre-retrieved passages and images. "
    "Your job is to convert ALL of them into rich, detailed slide content.\n\n"
    "MANDATORY rules — no exceptions:\n"
    "1. Every single passage in the list MUST appear as exactly one TextContent entry in content[]. "
    "   Do NOT skip, merge, or omit any passage, regardless of how relevant it seems.\n"
    "2. Each TextContent.text must be a detailed paragraph of 3–6 sentences that rewrites and expands "
    "   the passage in the context of the slide topic. Do not copy verbatim.\n"
    "3. Every retrieved image MUST appear as an ImageContent entry in images[]. "
    "   Copy image_url, image_id, and score exactly as shown.\n"
    "4. Exception: TITLE slides set content to null (no text) but still include all images.\n"
    "5. Use only information from the given passages — do not invent facts.\n"
    "6. Return a single valid SlideContent JSON object."
)


def build_synthesis_prompt(
    slide_structure: SlideStructure,
    chunks: List[ChunkSearchResult],
    images: List[ImageSearchResult],
) -> str:
    """Embed all retrieved chunks and images directly into the synthesis prompt."""
    chunks_section = "\n\n".join(
        f"[{i + 1}] chunk_id={c.chunk_id}  page={c.page_number}  score={c.score:.3f}\n{c.content}"
        for i, c in enumerate(chunks)
    ) or "(no passages retrieved)"

    images_section = "\n\n".join(
        f"[{i + 1}] image_id={img.image_id}  page={img.page_number}  score={img.score:.3f}\n"
        f"url: {img.storage_path}\ndescription: {img.description or '(none)'}"
        for i, img in enumerate(images)
    ) or "(no images retrieved)"

    return (
        f"Slide metadata:\n"
        f"  slide_number: {slide_structure.slide_number}\n"
        f"  slide_type: {slide_structure.slide_type.value}\n"
        f"  title: {slide_structure.title}\n"
        f"  description: {slide_structure.description}\n\n"
        f"Retrieved passages — {len(chunks)} total. ALL must appear in content[]:\n\n"
        f"{chunks_section}\n\n"
        f"Retrieved images — {len(images)} total. ALL must appear in images[]:\n\n"
        f"{images_section}\n\n"
        "Produce a SlideContent object:\n"
        "- content[]: one entry per passage, in order. "
        "  text = detailed paragraph, chunk_id = the chunk_id shown, score = the score shown.\n"
        "- images[]: one entry per image. "
        "  image_url = url shown, image_id = image_id shown, score = score shown."
    )
