"""CRUD operations for Image objects in the database."""

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from src.db.models import Image
from src.schemas.document.image import ImageCreate


def create_images(db: Session, images: List[ImageCreate]) -> List[Image]:
    """
    Stage multiple images in the database session.

    Args:
        db: SQLAlchemy session.
        images: List of ImageCreate schemas.

    Returns:
        List of Image instances (not yet committed).
    """
    db_images: List[Image] = [Image(**img.model_dump()) for img in images]
    db.add_all(db_images)
    return db_images


def get_images_by_document(db: Session, document_id: UUID) -> List[Image]:
    """
    Get all images for a specific document.

    Args:
        db: SQLAlchemy session.
        document_id: Document ID.

    Returns:
        List of Image instances.
    """
    stmt = select(Image).where(Image.document_id == document_id)
    result = db.execute(stmt).scalars().all()
    return result


def search_similar(
    db: Session,
    query_vector: List[float],
    document_id: UUID,
    limit: int = 10,
    max_distance: float = 0.5,
) -> List[Tuple["Image", float]]:
    """
    Hybrid image similarity search:
    - Stage 1: ANN search using description_vector (HNSW index)
    - Stage 2: Optionally filter by max_distance and return top results

    Args:
        db: SQLAlchemy Session.
        query_vector: Query embedding vector.
        document_id: The document ID to restrict search.
        limit: Maximum number of results.
        max_distance: Maximum distance threshold (smaller = closer match).

    Returns:
        List of tuples: (Image, distance)
    """
    # Constants for similarity search
    _CANDIDATE_MULTIPLIER = 5

    # Stage 1: retrieve top candidates using HNSW index
    candidate_limit = limit * _CANDIDATE_MULTIPLIER

    stmt = (
        select(
            Image,
            Image.description_vector.cosine_distance(query_vector).label("distance"),
        )
        .where(Image.document_id == document_id)
        .where(Image.description_vector.isnot(None))
        .order_by(Image.description_vector.cosine_distance(query_vector))
        .limit(candidate_limit)
    )

    candidates = db.execute(stmt).all()

    # Stage 2: Optionally filter by max_distance and return top results
    results: List[Tuple["Image", float]] = []

    for row in candidates:
        image = row.Image
        distance = float(row.distance)

        if distance <= max_distance:
            results.append((image, distance))

    results.sort(key=lambda x: x[1])

    return results[:limit]
