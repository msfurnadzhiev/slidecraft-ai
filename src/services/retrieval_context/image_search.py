"""Image semantic search for document-scoped RAG.

Uses text embeddings of image descriptions (384-dim vectors) to
perform similarity search over images extracted from a document.
"""

from typing import List, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.crud import image_crud
from src.db.models import Image
from src.schemas.image import ImageSearchResult

# Score = 1.0 - distance (higher = more similar)
_SIMILARITY_OFFSET = 1.0

ImageSimilarityRows = List[Tuple[Image, float]]


class ImageSearch:
    """
    Semantic search over images belonging to a single document.

    This class performs vector similarity search over description vectors.
    """

    def __init__(self, db: Session):
        """Initialize the image search service."""
        self.db = db

    def search(
        self,
        document_id: UUID,
        query_vector: List[float],
        limit: int,
        max_distance: float | None,
    ) -> List[ImageSearchResult]:
        """
        Run semantic similarity search for images within a document.

        Args:
            document_id: ID of the document to search within.
            query_vector: Embedding vector representing the search query.
            limit: Maximum number of results to return.
            max_distance: Optional maximum vector distance threshold for matches.

        Returns:
            A list of ImageSearchResult objects containing image metadata
            and similarity scores.
        """
        image_rows = image_crud.search_similar(
            db=self.db,
            query_vector=query_vector,
            document_id=document_id,
            limit=limit,
            max_distance=max_distance,
        )
        return self._build_results(image_rows)

    def fetch_all(self, document_id: UUID) -> List[ImageSearchResult]:
        """
        Retrieve all images belonging to a document.

        This method bypasses semantic filtering and returns every image
        with a neutral similarity score.

        Args:
            document_id: ID of the document.

        Returns:
            A list of ImageSearchResult objects for all images in the
            document, each assigned a default similarity score of 1.0.
        """
        images = image_crud.get_images_by_document(self.db, document_id)
        image_rows: ImageSimilarityRows = [(img, 0.0) for img in images]
        return self._build_results(image_rows)

    def _build_results(
        self, image_rows: ImageSimilarityRows
    ) -> List[ImageSearchResult]:
        """
        Convert raw similarity query results into image search result objects.

        Args:
            image_rows: List of tuples containing Image ORM objects and
                their vector distance values.

        Returns:
            A list of ImageSearchResult objects ready to be returned by
            the search API.
        """
        if not image_rows:
            return []

        results: List[ImageSearchResult] = []
        for image, distance in image_rows:
            score = round(_SIMILARITY_OFFSET - float(distance), 4)
            results.append(
                ImageSearchResult(
                    image_id=image.image_id,
                    page_number=image.page_number,
                    storage_path=image.storage_path,
                    file_name=image.file_name,
                    score=score,
                )
            )
        return results