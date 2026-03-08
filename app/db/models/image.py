from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

from app.db.models import Base as BaseModel


class Image(BaseModel):
    __tablename__ = "images"

    image_id = Column(String, primary_key=True)

    document_id = Column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    )

    storage_path = Column(String, nullable=False)
    page_number = Column(Integer, nullable=False)
    file_name = Column(String, nullable=False)

    document = relationship("Document", back_populates="images")
    embedding = relationship(
        "ImageEmbedding",
        back_populates="image",
        uselist=False,
        cascade="all, delete-orphan",
    )
