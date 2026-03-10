"""This module defines the Image SQLAlchemy model."""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, Integer, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.db.models import Base as BaseModel

if TYPE_CHECKING:
    from app.db.models.document import Document


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
    description_vector: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)

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
