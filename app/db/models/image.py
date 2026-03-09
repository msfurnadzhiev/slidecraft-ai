"""This module defines the Image SQLAlchemy model."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector

from app.db.models import Base as BaseModel

if TYPE_CHECKING:
    from app.db.models.document import Document


class Image(BaseModel):
    __tablename__ = "images"

    image_id: Mapped[str] = mapped_column(String, primary_key=True)

    document_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    storage_path: Mapped[str] = mapped_column(String, nullable=False)
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    file_name: Mapped[str] = mapped_column(String, nullable=False)
    vector: Mapped[list[float] | None] = mapped_column(Vector(512), nullable=True)

    document: Mapped["Document"] = relationship(back_populates="images")