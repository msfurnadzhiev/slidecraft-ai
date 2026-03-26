"""Utility functions for the content generation task."""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from src.db.crud import document_crud
from src.schemas.presentation.slide import TextContent, ImageContent

log = logging.getLogger(__name__)

def validate_chunks(
    content: Optional[List[TextContent]],
    db: Session,
) -> Optional[List[TextContent]]:
    """Filter out TextContent entries with unknown chunk IDs."""
    if not content:
        return None

    known_ids = {
        chunk.chunk_id
        for chunk in document_crud.get_chunks_by_ids(
            db, [entry.chunk_id for entry in content]
        )
    }

    return _filter_valid(
        items=content,
        is_valid=lambda e: e.chunk_id in known_ids,
        label="chunk_id",
        get_id=lambda e: e.chunk_id,
    )

def validate_images(
    images: Optional[List[ImageContent]],
    db: Session,
) -> Optional[List[ImageContent]]:
    """Filter out ImageContent entries with unknown image IDs."""
    if not images:
        log.debug("validate_images: agent produced no images (images field is null/empty)")
        return None

    log.info("validate_images: agent produced %d image(s) — verifying against DB", len(images))

    known_ids = {
        img.image_id
        for img in document_crud.get_images_by_ids(
            db, [img.image_id for img in images]
        )
    }

    result = _filter_valid(
        items=images,
        is_valid=lambda e: e.image_id in known_ids,
        label="image_id",
        get_id=lambda e: e.image_id,
    )

    log.info(
        "validate_images: %d/%d image(s) passed DB validation",
        len(result) if result else 0,
        len(images),
    )
    return result

def _filter_valid(
    items: List,
    is_valid,
    label: str,
    get_id,
) -> Optional[List]:
    """Generic validator that filters invalid items and logs warnings."""
    validated = []

    for item in items:
        if not is_valid(item):
            log.warning(
                "Unknown %s=%s selected by agent — dropped",
                label,
                get_id(item),
            )
            continue
        validated.append(item)

    return validated or None