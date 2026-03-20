"""This module defines the Chunk SQLAlchemy model."""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Integer, Index, text, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from src.db.models import Base as BaseModel

if TYPE_CHECKING:
    from src.db.models.document import Document


class Chunk(BaseModel):
    __tablename__ = "chunks"

    chunk_id: Mapped[UUID] = mapped_column(primary_key=True)

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)

    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_vector: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_vector: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "page_number",
            name="unique_document_page",
        ),
        Index(
            "chunks_content_vector_idx",
            "content_vector",
            postgresql_using="hnsw",
            postgresql_ops={"content_vector": "vector_cosine_ops"},
            postgresql_where=text("content_vector IS NOT NULL")
        ),
        Index(
            "chunks_summary_vector_idx",
            "summary_vector",
            postgresql_using="hnsw",
            postgresql_ops={"summary_vector": "vector_cosine_ops"},
            postgresql_where=text("summary_vector IS NOT NULL")
        ),
    )
