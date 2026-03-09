"""This module defines the Chunk SQLAlchemy model."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, Index, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.db.models import Base as BaseModel

if TYPE_CHECKING:
    from app.db.models.document import Document


class Chunk(BaseModel):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String, primary_key=True)

    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    start_char_offset: Mapped[int] = mapped_column(Integer, nullable=False)
    end_char_offset: Mapped[int] = mapped_column(Integer, nullable=False)

    vector: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "page_number",
            "chunk_index",
            name="unique_chunks_doc_page_chunk",
        ),
        Index(
            "chunks_vector_idx",
            "vector",
            postgresql_using="hnsw",
            postgresql_ops={"vector": "vector_cosine_ops"},
        ),
    )
