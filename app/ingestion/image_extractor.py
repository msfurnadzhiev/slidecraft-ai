"""Extract embedded images from PDF files and produce ImageCreate with storage."""

import uuid
from typing import List

import fitz

from app.schemas.image import ImageCreate
from app.storage import ImageStorage
from app.utils.singleton import SingletonMeta


class PDFImageExtractor(metaclass=SingletonMeta):
    """Extract images from PDF pages using PyMuPDF."""

    def extract_images(
        self, file_path: str, document_id: str, image_storage: ImageStorage
    ) -> List[ImageCreate]:
        """Extract all embedded images from a PDF, save to storage, return ImageCreate list.

        Iterates each page and extracts image data for every embedded image.
        Same image on multiple pages is extracted once per page.

        Args:
            file_path: Path to the PDF file.
            document_id: Document ID for storage paths.
            image_storage: Storage backend to persist image bytes.

        Returns:
            List of ImageCreate (image_id, document_id, storage_path, page_number, file_name).
        """
        result: List[ImageCreate] = []
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
                    page_number = page_index + 1
                    ext = base.get("ext", "png")
                    filename = f"page_{page_number}_{image_index}.{ext}"
                    storage_path = image_storage.save_bytes(
                        document_id=document_id,
                        filename=filename,
                        data=base["image"],
                    )
                    result.append(
                        ImageCreate(
                            image_id=str(uuid.uuid4()),
                            document_id=document_id,
                            storage_path=storage_path,
                            page_number=page_number,
                            file_name=filename,
                        )
                    )
        finally:
            doc.close()
        return result
