"""CRUD operations for Image objects in the database."""

from typing import List, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.models import Image
from app.schemas.image import ImageCreate


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


def get_images_by_document(db: Session, document_id: str) -> List[Image]:
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
    document_id: str,
    limit: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> List[Tuple[Image, float]]:
    """
    Return images most similar to query_vector using cosine distance.

    Args:
        db: SQLAlchemy session.
        query_vector: Query embedding vector.
        document_id: Document ID to filter images.
        limit: Maximum number of results.
        max_distance: Optional maximum cosine distance filter.

    Returns:
        List of tuples: (Image, distance)
    """
    distance_expr = Image.vector.cosine_distance(query_vector)

    stmt = (
        select(Image, distance_expr.label("distance"))
        .where(Image.document_id == document_id)
        .where(Image.vector.isnot(None))
    )

    if max_distance is not None:
        stmt = stmt.where(distance_expr <= max_distance)

    stmt = stmt.order_by(distance_expr)

    if limit is not None:
        stmt = stmt.limit(limit)

    results = db.execute(stmt).all()

    return [(row.Image, float(row.distance)) for row in results]