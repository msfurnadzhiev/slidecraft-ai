"""Extract embedded images from PDF files and produce ImageCreate with storage."""

import uuid
from dataclasses import dataclass, field
from typing import Dict, List

import fitz

from app.schemas.image import ImageCreate
from app.storage import ImageStorage
from app.utils.singleton import SingletonMeta


@dataclass
class ImageExtractionResult:
    """Bundle returned by ``extract_images``: metadata for the DB + raw bytes for embedding."""

    images: List[ImageCreate] = field(default_factory=list)
    image_bytes: Dict[str, bytes] = field(default_factory=dict)


class PDFImageExtractor(metaclass=SingletonMeta):
    """Extract images from PDF pages using PyMuPDF."""

    def extract_images(
        self, file_path: str, document_id: str, image_storage: ImageStorage,
    ) -> ImageExtractionResult:
        """Extract all embedded images from a PDF, save to storage, return metadata + bytes.

        The raw bytes are preserved in the result so they can be forwarded to
        the embedding generator without a second read from storage.
        """
        result = ImageExtractionResult()
        doc = fitz.open(file_path)
        try:
            for page_index in range(len(doc)):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                for image_index, img in enumerate(image_list):
                    xref = img[0]
                    try:
                        base = doc.extract_image(xref)
                    except Exception:
                        continue

                    raw_bytes: bytes = base["image"]
                    page_number = page_index + 1
                    ext = base.get("ext", "png")
                    filename = f"page_{page_number}_{image_index}.{ext}"
                    image_id = str(uuid.uuid4())

                    storage_path = image_storage.save_bytes(
                        document_id=document_id,
                        filename=filename,
                        data=raw_bytes,
                    )

                    result.images.append(
                        ImageCreate(
                            image_id=image_id,
                            document_id=document_id,
                            storage_path=storage_path,
                            page_number=page_number,
                            file_name=filename,
                        )
                    )
                    result.image_bytes[image_id] = raw_bytes
        finally:
            doc.close()
        return result
