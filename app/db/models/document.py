from sqlalchemy import Column, String, Integer
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import relationship

from app.db.models import Base as BaseModel


class Document(BaseModel):
    __tablename__ = "documents"

    document_id = Column(String, primary_key=True)
    file_name = Column(String, nullable=False)
    storage_path = Column(String, nullable=False)
    total_pages = Column(Integer, nullable=False)
    metadata_ = Column("metadata", JSON, nullable=True)

    chunks = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    images = relationship(
        "Image",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    embeddings = relationship(
        "Embedding",
        back_populates="document",
        cascade="all, delete-orphan",
    )
