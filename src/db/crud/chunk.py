"""CRUD operations for Chunk objects in the database."""

from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import case, select

from src.db.models import Chunk
from src.schemas.document.chunk import ChunkCreate

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
    limit: int = 10,
    max_distance: float = 0.5,
) -> List[Tuple["Chunk", float]]:
    """Hybrid similarity search using summary_vector and content_vector.

    The search is performed in two stages:
    1. ANN search using summary_vector (HNSW index)
    2. Rerank candidates using 40% summary + 60% content

    Args:
        db: SQLAlchemy Session.
        query_vector: Query embedding vector.
        document_id: The document ID to restrict search.
        limit: Maximum number of results.
        max_distance: Maximum distance threshold (smaller = closer match).

    Returns:
        List of tuples: (Chunk, distance)
    """

    # Constants for similarity search
    _CANDIDATE_MULTIPLIER = 5
    _WEIGHT_SUMMARY = 0.4
    _WEIGHT_CONTENT = 0.6
    
    # ANN search using summary_vector (HNSW index)
    candidate_limit = limit * _CANDIDATE_MULTIPLIER

    stmt = (
        select(
            Chunk,
            Chunk.summary_vector.cosine_distance(query_vector).label("summary_dist"),
            Chunk.content_vector.cosine_distance(query_vector).label("content_dist"),
        )
        .where(Chunk.document_id == document_id)
        .where(Chunk.summary_vector.isnot(None))
        .where(Chunk.content_vector.isnot(None))
        .order_by(Chunk.summary_vector.cosine_distance(query_vector))
        .limit(candidate_limit)
    )

    candidates = db.execute(stmt).all()

    # Rerank candidates using 40% summary + 60% content
    results: List[Tuple["Chunk", float]] = []

    for row in candidates:
        chunk = row.Chunk
        summary_dist = float(row.summary_dist)
        content_dist = float(row.content_dist)

        distance = _WEIGHT_SUMMARY * summary_dist + _WEIGHT_CONTENT * content_dist

        if distance <= max_distance:
            results.append((chunk, distance))

    results.sort(key=lambda x: x[1])

    return results[:limit]