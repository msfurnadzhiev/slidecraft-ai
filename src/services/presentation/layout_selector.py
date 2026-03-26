"""Utilities for mapping slide types to template layout indices.

The logic here is intentionally stateless: given a list of
SlideLayoutResponse objects (parsed from a template) and a SlideType,
it returns the best-matching layout index to use when building a slide.
"""

from typing import Dict, List

from src.schemas.presentation.slide import SlideType
from src.schemas.template import SlideLayoutResponse

# Maps each SlideType to an ordered list of preferred layout roles.
# The first role that has at least one matching layout in the template wins.
SLIDE_TYPE_LAYOUT_PREFERENCES: Dict[SlideType, List[str]] = {
    SlideType.TITLE: ["title", "subtitle"],
    SlideType.CONTENT: ["content"],
    SlideType.IMAGE: ["image", "content"],
    SlideType.DATA: ["table", "chart", "content"],
    SlideType.CLOSING: ["title", "subtitle", "content"],
}

# When a layout has multiple roles, the highest-priority role in this tuple
# is used to classify it for the layout map.
LAYOUT_ROLE_PRIORITY: tuple = (
    "image", "chart", "table", "diagram", "media",
    "content", "subtitle", "title",
)


def build_layout_map(
    layouts: List[SlideLayoutResponse],
) -> Dict[str, List[int]]:
    """Build a mapping from dominant role name to a list of layout indices.

    Each layout is assigned a single *dominant* role by finding the first
    matching entry in :data:`LAYOUT_ROLE_PRIORITY`.  Layouts whose elements
    carry no recognised role are silently skipped.

    Args:
        layouts: Layout metadata returned by the template inspection service.

    Returns:
        A dict where keys are role names and values are ordered lists of
        layout indices that carry that role.
    """
    result: Dict[str, List[int]] = {}

    for layout in layouts:
        roles = {el.role for el in layout.elements if el.role}

        for role in LAYOUT_ROLE_PRIORITY:
            if role in roles:
                result.setdefault(role, []).append(layout.layout_index)
                break

    return result


def select_layout_index(
    slide_type: SlideType,
    layout_map: Dict[str, List[int]],
) -> int:
    """Return the best layout index for slide_type given layout_map.

    Iterates through the preferred roles for slide_type and returns 
    the first layout index found in layout_map. Falls back to index 0 
    when no preference matches. This corresponds to the very first 
    layout in the template.

    Args:
        slide_type: The logical type of the slide being built.
        layout_map: Role → layout indices mapping built by :func:`build_layout_map`.

    Returns:
        A zero-based layout index suitable for
    """
    for preferred_role in SLIDE_TYPE_LAYOUT_PREFERENCES.get(slide_type, []):
        indices = layout_map.get(preferred_role)
        if indices:
            return indices[0]
    return 0
