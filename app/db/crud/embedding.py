from sqlalchemy.orm import Session
from typing import List, Optional, Tuple

from app.db.models import Embedding
from app.schemas.embedding import EmbeddingCreate


def create_embeddings(db: Session, embeddings: List[EmbeddingCreate]) -> List[Embedding]:
    """Stage multiple embeddings in the session (unified: chunk, image)."""
    db_embeddings = []
    for emb in embeddings:
        db_embeddings.append(Embedding(**emb.model_dump()))
    db.add_all(db_embeddings)
    return db_embeddings


def get_embedding_by_object(
    db: Session, object_type: str, object_id: str
) -> Optional[Embedding]:
    """Get the embedding for an object by type and id."""
    return (
        db.query(Embedding)
        .filter(
            Embedding.object_type == object_type,
            Embedding.object_id == object_id,
        )
        .first()
    )


def get_embeddings_by_document(db: Session, document_id: str) -> List[Embedding]:
    """Get all embeddings for a document."""
    return db.query(Embedding).filter(Embedding.document_id == document_id).all()


def delete_embedding_by_object(
    db: Session, object_type: str, object_id: str
) -> bool:
    """Stage an embedding deletion by object type and id."""
    emb = get_embedding_by_object(db, object_type, object_id)
    if emb:
        db.delete(emb)
        return True
    return False


def delete_embeddings_by_document(db: Session, document_id: str) -> int:
    """Stage deletion of all embeddings for a document. Returns count of rows."""
    return (
        db.query(Embedding)
        .filter(Embedding.document_id == document_id)
        .delete(synchronize_session=False)
    )


def update_embedding_vector(
    db: Session, object_type: str, object_id: str, vector: List[float]
) -> Optional[Embedding]:
    """Stage an embedding vector update."""
    emb = get_embedding_by_object(db, object_type, object_id)
    if emb:
        emb.vector = vector
    return emb


def search_similar_embeddings(
    db: Session,
    query_vector: List[float],
    document_id: str,
    object_type: str,
    limit: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> List[Tuple[Embedding, float]]:
    """Return embeddings most similar to query_vector (cosine distance), scoped to document.

    pgvector cosine_distance is in [0, 2]; 0 = identical.

    Args:
        db: The database session
        query_vector: The query vector
        document_id: The document ID
        object_type: The object type
        limit: The limit on the number of results
        max_distance: The maximum distance for the results (0..1, e.g. 0.25 = 75% similarity)

    Returns:
        A list of tuples containing the embedding and the distance
    
    """
    distance = Embedding.vector.cosine_distance(query_vector)
    q = db.query(Embedding, distance).filter(
        Embedding.document_id == document_id,
        Embedding.object_type == object_type,
    )
    if max_distance is not None:
        q = q.filter(distance <= max_distance)
    
    q = q.order_by(distance)

    if limit is not None:
        q = q.limit(limit)
    
    rows = q.all()
    return [(row[0], float(row[1])) for row in rows]
