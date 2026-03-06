import uuid

from sqlalchemy import Column, String, ForeignKey, Integer, UniqueConstraint
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.models import Base as BaseModel


class Embedding(BaseModel):
    """Unified embeddings for chunks and images. Use object_type for filtering."""

    __tablename__ = "embeddings"
    __table_args__ = (UniqueConstraint("object_type", "object_id", name="uq_embeddings_object"),)

    embedding_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    object_id = Column(String, nullable=False)
    object_type = Column(String, nullable=False)  # 'chunk' | 'image'
    vector = Column(Vector(384), nullable=False)
    page_number = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="embeddings")
