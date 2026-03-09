from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.models import Chunk
from app.schemas.chunk import ChunkCreate



def create_chunks(db: Session, chunks: List[ChunkCreate]) -> List[Chunk]:
    """Create multiple chunks in the database."""
    db_chunks = [
        Chunk(**chunk.model_dump(exclude={"text"})) for chunk in chunks
    ]
    db.add_all(db_chunks)
    return db_chunks


def get_chunk(db: Session, chunk_id: str) -> Optional[Chunk]:
    """Get a chunk by ID."""
    return db.query(Chunk).filter(Chunk.chunk_id == chunk_id).first()


def get_chunks_by_document(db: Session, document_id: str) -> List[Chunk]:
    """Get all chunks for a document."""
    return db.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.page_number, Chunk.chunk_index).all()


def get_chunks_by_page(db: Session, document_id: str, page_number: int) -> List[Chunk]:
    """Get all chunks for a specific page."""
    return db.query(Chunk).filter(
        Chunk.document_id == document_id,
        Chunk.page_number == page_number
    ).order_by(Chunk.chunk_index).all()


def delete_chunk(db: Session, chunk_id: str) -> bool:
    """Stage a chunk deletion in the session."""
    chunk = get_chunk(db, chunk_id)
    if chunk:
        db.delete(chunk)
        return True
    return False


def delete_chunks_by_document(db: Session, document_id: str) -> int:
    """Stage deletion of all chunks for a document. Returns count of matched rows."""
    return db.query(Chunk).filter(Chunk.document_id == document_id).delete()


def update_chunk(
    db: Session,
    chunk_id: str,
    token_count: Optional[int] = None,
) -> Optional[Chunk]:
    """Stage a chunk update in the session."""
    chunk = get_chunk(db, chunk_id)
    if chunk and token_count is not None:
        chunk.token_count = token_count
    return chunk
