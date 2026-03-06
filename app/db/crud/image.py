from sqlalchemy.orm import Session
from typing import List, Optional

from app.db.models import Image
from app.schemas.image import ImageCreate


def create_image(db: Session, image: ImageCreate) -> Image:
    """Stage a new image in the session."""
    db_image = Image(**image.model_dump())
    db.add(db_image)
    return db_image


def create_images(db: Session, images: List[ImageCreate]) -> List[Image]:
    """Stage multiple images in the session."""
    db_images = [Image(**img.model_dump()) for img in images]
    db.add_all(db_images)
    return db_images


def get_image(db: Session, image_id: str) -> Optional[Image]:
    """Get an image by ID."""
    return db.query(Image).filter(Image.image_id == image_id).first()


def get_images_by_document(db: Session, document_id: str) -> List[Image]:
    """Get all images for a document, ordered by page and file_name."""
    return (
        db.query(Image)
        .filter(Image.document_id == document_id)
        .order_by(Image.page_number, Image.file_name)
        .all()
    )


def delete_image(db: Session, image_id: str) -> bool:
    """Stage an image deletion in the session (embedding deleted via app or cascade)."""
    image = get_image(db, image_id)
    if image:
        db.delete(image)
        return True
    return False


def delete_images_by_document(db: Session, document_id: str) -> int:
    """Stage deletion of all images for a document. Returns count of deleted rows."""
    return db.query(Image).filter(Image.document_id == document_id).delete(
        synchronize_session=False
    )
