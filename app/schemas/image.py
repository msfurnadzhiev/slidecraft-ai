from app.schemas.base import BaseSchema


class ImageBase(BaseSchema):
    """Base schema for image."""

    document_id: str
    storage_path: str
    page_number: int
    file_name: str


class ImageCreate(ImageBase):
    """Schema for creating an image."""

    image_id: str


class ImageResponse(ImageBase):
    """Schema for image response."""

    image_id: str
