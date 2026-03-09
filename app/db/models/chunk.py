from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship

from app.db.models import Base as BaseModel

class Chunk(BaseModel):
    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True)

    document_id = Column(
        String,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False
    )

    page_number = Column(Integer, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    token_count = Column(Integer, nullable=False)
    start_char_offset = Column(Integer, nullable=False)
    end_char_offset = Column(Integer, nullable=False)

    document = relationship("Document", back_populates="chunks")
    embedding = relationship(
        "Embedding",
        primaryjoin="and_(Embedding.object_type=='chunk', Embedding.object_id==Chunk.chunk_id)",
        foreign_keys="[Embedding.object_id]",
        uselist=False,
        viewonly=True,
    )

