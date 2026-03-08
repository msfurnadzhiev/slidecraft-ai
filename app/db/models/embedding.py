import uuid

from sqlalchemy import Column, String, ForeignKey, Integer
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from app.db.models import Base as BaseModel


class ChunkEmbedding(BaseModel):
    """Text chunk embedding (sentence-transformers, 384-dim)."""

    __tablename__ = "chunk_embeddings"

    embedding_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    chunk_id = Column(
        String,
        ForeignKey("chunks.chunk_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    vector = Column(Vector(384), nullable=False)
    page_number = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="chunk_embeddings")
    chunk = relationship("Chunk", back_populates="embedding")


class ImageEmbedding(BaseModel):
    """Image embedding (CLIP ViT-B/32, 512-dim)."""

    __tablename__ = "image_embeddings"

    embedding_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )
    image_id = Column(
        String,
        ForeignKey("images.image_id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    vector = Column(Vector(512), nullable=False)
    page_number = Column(Integer, nullable=True)

    document = relationship("Document", back_populates="image_embeddings")
    image = relationship("Image", back_populates="embedding")
