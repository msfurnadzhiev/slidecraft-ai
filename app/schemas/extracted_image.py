"""Schema for images extracted from PDFs during ingestion."""

from app.schemas.base import BaseSchema


class ExtractedImage(BaseSchema):
    """One image extracted from a PDF (in-memory; not persisted to DB here)."""

    page_number: int
    image_index: int
    data: bytes
    ext: str
    width: int
    height: int

    def suggested_filename(self) -> str:
        """Suggested filename when saving to image storage (e.g. page_1_0.png)."""
        return f"page_{self.page_number}_{self.image_index}.{self.ext}"
