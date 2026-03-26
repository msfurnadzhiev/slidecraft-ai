"""CRUD operations for Document, Chunk, and Image objects in the database."""

import logging
from typing import List, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session
from sqlalchemy import select

from src.db.models import Document, Chunk, Image
from src.schemas.document import DocumentCreate, ChunkCreate, ImageCreate

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document CRUD operations
# ---------------------------------------------------------------------------


def create_document(db: Session, document: DocumentCreate) -> Document:
    """
    Stage a new document in the session.

    Args:
        db: SQLAlchemy session.
        document: DocumentCreate schema.

    Returns:
        Document instance (not yet committed).
    """
    data = document.model_dump()
    data["metadata_"] = data.pop("metadata", None)
    db_document = Document(**data)
    db.add(db_document)
    return db_document


def get_document(db: Session, document_id: UUID) -> Optional[Document]:
    """
    Get a document by its ID.

    Args:
        db: SQLAlchemy session.
        document_id: Document primary key.

    Returns:
        Document instance if found, else None.
    """
    stmt = select(Document).where(Document.document_id == document_id)
    result = db.execute(stmt).scalars().first()
    return result


def get_all_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    """
    Get all documents with pagination.

    Args:
        db: SQLAlchemy session.
        skip: Number of rows to skip.
        limit: Maximum number of rows to return.

    Returns:
        List of Document instances.
    """
    stmt = select(Document).offset(skip).limit(limit)
    result = db.execute(stmt).scalars().all()
    return result


def delete_document(db: Session, document_id: UUID) -> bool:
    """
    Stage a document deletion in the session.

    Args:
        db: SQLAlchemy session.
        document_id: Document primary key.

    Returns:
        True if document was found and staged for deletion, False otherwise.
    """
    document = get_document(db, document_id)
    if document:
        db.delete(document)
        return True
    return False



# ---------------------------------------------------------------------------
# Chunk CRUD operations
# ---------------------------------------------------------------------------


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


def get_chunks_by_ids(db: Session, chunk_ids: List[UUID]) -> List[Chunk]:
    """
    Fetch chunks by a list of primary-key IDs.

    Args:
        db: SQLAlchemy session.
        chunk_ids: Chunk primary keys to look up.

    Returns:
        List of Chunk instances that exist in the database.
    """
    if not chunk_ids:
        return []
    stmt = select(Chunk).where(Chunk.chunk_id.in_(chunk_ids))
    return list(db.execute(stmt).scalars().all())


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


def chunk_similarity_search(
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


# ---------------------------------------------------------------------------
# Image CRUD operations
# ---------------------------------------------------------------------------


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


def get_images_by_ids(db: Session, image_ids: List[UUID]) -> List[Image]:
    """
    Fetch images by a list of primary-key IDs.

    Args:
        db: SQLAlchemy session.
        image_ids: Image primary keys to look up.

    Returns:
        List of Image instances that exist in the database.
    """
    if not image_ids:
        return []
    stmt = select(Image).where(Image.image_id.in_(image_ids))
    return list(db.execute(stmt).scalars().all())


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


def image_similarity_search(
    db: Session,
    query_vector: List[float],
    document_id: UUID,
    limit: int = 10,
    max_distance: float = 0.5,
) -> List[Tuple["Image", float]]:
    """Image similarity search using description_vector.

    The search is performed in two stages:
    1. ANN search using description_vector (HNSW index)
    2. Rerank candidates using cosine distance

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

    # ANN search using description_vector (HNSW index)
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

    # Rerank candidates using cosine distance
    results: List[Tuple["Image", float]] = []

    for row in candidates:
        image = row.Image
        distance = float(row.distance)

        if distance <= max_distance:
            results.append((image, distance))

    results.sort(key=lambda x: x[1])

    return results[:limit]
