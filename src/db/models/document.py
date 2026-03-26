"""This module defines the Document, Chunk, and Image SQLAlchemy models."""

from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, Index, Integer, String, text, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.db.models import Base as BaseModel


class Document(BaseModel):
    __tablename__ = "documents"

    document_id: Mapped[UUID] = mapped_column(primary_key=True)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    total_pages: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_: Mapped[dict | None] = mapped_column("metadata", JSON, nullable=True)

    # Relationships
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    images: Mapped[list["Image"]] = relationship(
        "Image",
        back_populates="document",
        cascade="all, delete-orphan",
    )


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
    content_vector: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)

    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_vector: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)

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


class Image(BaseModel):
    __tablename__ = "images"

    image_id: Mapped[UUID] = mapped_column(primary_key=True)

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)

    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    description_vector: Mapped[list[float] | None] = mapped_column(Vector(768), nullable=True)

    # Relationships
    document: Mapped["Document"] = relationship(back_populates="images")

    __table_args__ = (
        Index(
            "images_description_vector_idx",
            "description_vector",
            postgresql_using="hnsw",
            postgresql_ops={"description_vector": "vector_cosine_ops"},
        ),
    )
