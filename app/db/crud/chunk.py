"""CRUD operations for Chunk objects in the database."""

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from app.db.models import Chunk
from app.schemas.chunk import ChunkCreate


def create_chunks(db: Session, chunks: List[ChunkCreate]) -> List[Chunk]:
    """
    Create multiple chunks in the database.

    Args:
        db: SQLAlchemy Session.
        chunks: List of ChunkCreate schemas.

    Returns:
        List of created Chunk instances (not yet committed).
    """
    db_chunks: List[Chunk] = [Chunk(**chunk.model_dump()) for chunk in chunks]
    db.add_all(db_chunks)
    return db_chunks


def get_chunks_by_document(db: Session, document_id: UUID) -> List[Chunk]:
    """
    Return all chunks for a document, ordered by page and chunk index.

    Args:
        db: SQLAlchemy Session.
        document_id: The document to fetch chunks for.

    Returns:
        List of Chunk instances in document order.
    """
    stmt = (
        select(Chunk)
        .where(Chunk.document_id == document_id)
        .order_by(Chunk.page_number)
    )
    return list(db.execute(stmt).scalars().all())


def search_similar(
    db: Session,
    query_vector: List[float],
    document_id: UUID,
    limit: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> List[Tuple[Chunk, float]]:
    """
    Return chunks most similar to query_vector using cosine distance.

    Args:
        db: SQLAlchemy Session.
        query_vector: Query embedding vector.
        document_id: The document ID to restrict search.
        limit: Maximum number of results.
        max_distance: Optional maximum cosine distance filter.

    Returns:
        List of tuples: (Chunk, distance)
    """
    distance_expr = Chunk.content_vector.cosine_distance(query_vector)

    stmt = (
        select(Chunk, distance_expr.label("distance"))
        .where(Chunk.document_id == document_id)
        .where(Chunk.content_vector.isnot(None))
    )

    if max_distance is not None:
        stmt = stmt.where(distance_expr <= max_distance)

    stmt = stmt.order_by(distance_expr)

    if limit is not None:
        stmt = stmt.limit(limit)

    results = db.execute(stmt).all()

    return [(row.Chunk, float(row.distance)) for row in results]
