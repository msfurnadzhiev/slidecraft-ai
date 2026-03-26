"""Layout inspection utilities for the slide-builder task."""

from typing import List, Tuple

from src.schemas.template import SlideLayoutResponse

# Placeholder roles filled automatically from slide metadata — the LLM must not touch them
AUTO_FILLED_ROLES: frozenset = frozenset({"title"})

# Placeholder roles that carry image or media content
IMAGE_ROLES: frozenset = frozenset({"image", "picture", "media"})


def has_image_placeholder(layout: SlideLayoutResponse) -> bool:
    """Return True if the layout contains an image/media placeholder."""
    return any(el.role in IMAGE_ROLES for el in layout.elements)


def count_content_placeholders(layout: SlideLayoutResponse) -> int:
    """Return the number of placeholders the LLM is expected to fill."""
    return sum(1 for el in layout.elements if el.role not in AUTO_FILLED_ROLES)


def split_placeholders(
    layout: SlideLayoutResponse,
) -> Tuple[List, List]:
    """Split layout elements into (auto-filled, fillable) lists."""
    auto_filled = [el for el in layout.elements if el.role in AUTO_FILLED_ROLES]
    fillable = [el for el in layout.elements if el.role not in AUTO_FILLED_ROLES]
    return auto_filled, fillable


def format_placeholder_list(placeholders) -> str:
    if not placeholders:
        return "(none)"

    return "\n".join(
        f"    placeholder_idx={el.placeholder_idx}  role={el.role or 'content'}"
        f"  w={el.width:.2f}  h={el.height:.2f}"
        for el in placeholders
    )


def format_auto_filled(placeholders) -> str:
    if not placeholders:
        return "none"

    return ", ".join(
        f"idx={el.placeholder_idx}(role={el.role})"
        for el in placeholders
    )
