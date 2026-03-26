"""Prompts for the slide-builder agent."""

from typing import List, Optional, Set

from src.schemas.presentation.slide import SlideContent, SlideType
from src.schemas.template import SlideLayoutResponse
from src.agents.tasks.slide_builder.utils import (
    has_image_placeholder,
    count_content_placeholders,
    split_placeholders,
    format_placeholder_list,
    format_auto_filled,
)


AGENT_SYSTEM_PROMPT = (
    "You are an expert presentation designer. Your job is to:\n"
    "  1. Select the best slide layout for the given slide.\n"
    "  2. Write concise, structured content for each fillable placeholder.\n\n"

    "Workflow — follow this order strictly:\n"
    "  Step 1: Call select_layout exactly once with the layout_index that best fits "
    "the slide type and source material. The tool will confirm your selection and list "
    "the placeholder_idx values you must fill.\n"
    "  Step 2: Call fill_placeholder once for EACH placeholder listed by select_layout. "
    "Do NOT skip any placeholder. Do NOT fill title placeholders — they are auto-filled.\n\n"

    "Recovery:\n"
    "  - If select_layout returns an ERROR, fix the index and call it again.\n"
    "  - If fill_placeholder returns an ERROR, fix the issue and call it again.\n\n"

    "Content rules:\n"
    "  - Use bullet points: 3-6 bullets per placeholder, each 5-10 words, separated by newlines.\n"
    "  - Synthesise from the source material — do NOT copy text verbatim.\n"
    "  - Do NOT repeat or paraphrase the title.\n"
    "  - Keep content concise, structured, and presentation-ready.\n"
)


def build_layout_description(
    layout: SlideLayoutResponse,
    used_layout_indices: Optional[Set[int]],
) -> str:
    auto_filled, fillable = split_placeholders(layout)

    has_image = has_image_placeholder(layout)
    content_count = count_content_placeholders(layout)

    used_marker = (
        " [ALREADY USED]"
        if used_layout_indices and layout.layout_index in used_layout_indices
        else ""
    )
    image_marker = " [IMAGE PLACEHOLDER]" if has_image else ""

    return (
        f"  layout_index={layout.layout_index}  "
        f"name={layout.name or '—'}  "
        f"content_slots={content_count}"
        f"{image_marker}{used_marker}\n"
        f"    auto-filled (title): {format_auto_filled(auto_filled)}\n"
        f"    yours to fill:\n{format_placeholder_list(fillable)}"
    )


def build_layouts_block(
    layouts: List[SlideLayoutResponse],
    used_layout_indices: Optional[Set[int]],
):
    lines = []
    image_capable = []

    for layout in layouts:
        if has_image_placeholder(layout):
            image_capable.append(layout.layout_index)
        lines.append(build_layout_description(layout, used_layout_indices))

    return "\n\n".join(lines) or "(no layouts available)", image_capable


def build_source_block(slide: SlideContent) -> str:
    if not slide.content:
        return "(no source material)"

    return "\n\n".join(
        f"[{i + 1}] {chunk.text}"
        for i, chunk in enumerate(slide.content)
    )


def build_variety_block(used_layout_indices: Optional[Set[int]]) -> str:
    if not used_layout_indices:
        return ""

    return (
        "## Layout usage so far\n\n"
        f"Already used layout indices: {sorted(used_layout_indices)}\n"
        "Prefer unused layouts for visual variety. "
        "Reuse only if no good alternative exists.\n\n"
    )


def build_content_guidance(
    slide: SlideContent,
    image_capable_indices: List[int],
) -> str:
    """Return slide-type-specific filling instructions."""
    slide_type = slide.slide_type

    if slide_type == SlideType.TITLE:
        return (
            "This is a TITLE slide.\n"
            f'- Title is already set: "{slide.title}"\n'
            "- If a subtitle placeholder exists, write one short tagline (≤15 words).\n"
            "- Leave other placeholders empty.\n"
        )

    if slide_type == SlideType.CLOSING:
        return (
            "This is a CLOSING slide.\n"
            f'- Title is already set: "{slide.title}"\n'
            "- Fill content with 3-5 concise takeaway bullets (≤10 words each).\n"
        )

    if slide_type == SlideType.IMAGE:
        if image_capable_indices:
            return (
                "This is an IMAGE slide.\n"
                f"- You MUST choose from image-capable layouts: {image_capable_indices}\n"
                f'- Title is already set: "{slide.title}"\n'
                "- Add 2-4 short supporting bullets.\n"
            )
        return (
            "This is an IMAGE slide but no image layouts are available.\n"
            "- Use a content layout and write 3-5 bullets.\n"
        )

    image_instruction = ""
    if slide.images and image_capable_indices:
        image_instruction = (
            f"This slide has images — choose from image-capable layouts: {image_capable_indices}\n"
        )

    return (
        "This is a CONTENT slide.\n"
        f"{image_instruction}"
        f'- Title is already set: "{slide.title}"\n'
        "- Fill ALL placeholders.\n"
        "- 3-5 short bullets per placeholder.\n"
        "- Split content logically across placeholders.\n"
    )

def build_revision_block(revision_feedback: Optional[str]) -> str:
    """Return a quality-revision section when feedback from a previous attempt exists."""
    if not revision_feedback:
        return ""
    return (
        "## Quality revision required\n\n"
        "Your previous attempt was rejected by the quality validator. "
        "You MUST address all of the issues listed below before selecting a layout.\n\n"
        f"{revision_feedback}\n\n"
    )


def build_agent_input(
    slide: SlideContent,
    layouts: List[SlideLayoutResponse],
    used_layout_indices: Optional[Set[int]] = None,
    revision_feedback: Optional[str] = None,
) -> str:
    """Build the human-turn message that starts an agent run for one slide.

    Args:
        slide: Slide content to assign a layout and fill placeholders for.
        layouts: All layouts available in the target template.
        used_layout_indices: Layout indices already chosen for earlier slides.
        revision_feedback: Actionable feedback from the QualityValidatorAgent
            that should be addressed in this revision attempt.  ``None`` on the
            first attempt.
    """
    layouts_block, image_capable = build_layouts_block(layouts, used_layout_indices)
    source_block = build_source_block(slide)
    variety_block = build_variety_block(used_layout_indices)
    content_guidance = build_content_guidance(slide, image_capable)
    revision_block = build_revision_block(revision_feedback)

    return (
        f"{revision_block}"
        f"{variety_block}"
        "## Available layouts\n\n"
        f"{layouts_block}\n\n"
        "## Slide input\n\n"
        f"slide_number : {slide.slide_number}\n"
        f"slide_type   : {slide.slide_type.value}\n"
        f"title        : {slide.title}\n"
        f"description  : {slide.description}\n"
        f"has_images   : {bool(slide.images)}\n\n"
        "## Source material\n\n"
        f"{source_block}\n\n"
        "## Content instructions\n\n"
        f"{content_guidance}\n"
        "Now call select_layout, then fill_placeholder for each listed placeholder."
    )
