"""This module defines the Document SQLAlchemy model."""

from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import Integer, String
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.models import Base as BaseModel

if TYPE_CHECKING:
    from app.db.models.chunk import Chunk
    from app.db.models.image import Image


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
