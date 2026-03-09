from sqlalchemy.orm import Session
from typing import List, Optional

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
