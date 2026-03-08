"""CRUD operations for image embeddings (CLIP ViT-B/32, 512-dim)."""

from typing import List, Optional, Tuple

from sqlalchemy.orm import Session

from app.db.models import ImageEmbedding
from app.schemas.embedding import ImageEmbeddingCreate


def create_embeddings(
    db: Session, embeddings: List[ImageEmbeddingCreate],
) -> List[ImageEmbedding]:
    """Stage multiple image embeddings in the session."""
    db_rows = [ImageEmbedding(**emb.model_dump()) for emb in embeddings]
    db.add_all(db_rows)
    return db_rows


def search_similar(
    db: Session,
    query_vector: List[float],
    document_id: str,
    limit: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> List[Tuple[ImageEmbedding, float]]:
    """Return image embeddings most similar to query_vector (cosine distance)."""
    distance = ImageEmbedding.vector.cosine_distance(query_vector)
    q = db.query(ImageEmbedding, distance).filter(
        ImageEmbedding.document_id == document_id,
    )
    if max_distance is not None:
        q = q.filter(distance <= max_distance)
    q = q.order_by(distance)
    if limit is not None:
        q = q.limit(limit)
    rows = q.all()
    return [(row[0], float(row[1])) for row in rows]


def delete_by_document(db: Session, document_id: str) -> int:
    """Stage deletion of all image embeddings for a document."""
    return (
        db.query(ImageEmbedding)
        .filter(ImageEmbedding.document_id == document_id)
        .delete(synchronize_session=False)
    )
