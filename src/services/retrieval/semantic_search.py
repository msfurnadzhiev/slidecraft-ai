"""Text (chunk) and image semantic search for document-scoped RAG, using query strings."""

import logging
from typing import List, Tuple, TypeVar, Generic, Callable
from uuid import UUID

from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

from src.db.crud import chunk_crud, image_crud
from src.db.models import Chunk, Image
from src.schemas.document.chunk import ChunkSearchResult
from src.schemas.document.image import ImageSearchResult

from src.utils.singleton import SingletonMeta
from src.infrastructure.embeddings.text_embedder import TextEmbedder

# Default values for semantic search
_SIMILARITY_OFFSET = 1.0
DEFAULT_CHUNK_RESULT_LIMIT = 5
DEFAULT_IMAGE_RESULT_LIMIT = 5
DEFAULT_CHUNK_SIMILARITY_THRESHOLD = 0.7
DEFAULT_IMAGE_SIMILARITY_THRESHOLD = 0.5

# Typing aliases
ChunkSimilarityRows = List[Tuple[Chunk, float]]
ImageSimilarityRows = List[Tuple[Image, float]]

T = TypeVar("T")
R = TypeVar("R")

class SemanticSearchSevice(metaclass=SingletonMeta):
    """
    Semantic search over content belonging to a single document.
    Supports both text chunks and images.
    """

    def __init__(self, db: Session):
        self.db = db
        self.embedder = TextEmbedder()

    def _compute_score(self, distance: float) -> float:
        """Convert distance to similarity score."""
        return round(_SIMILARITY_OFFSET - float(distance), 4)

    def _semantic_search(
        self,
        query: str,
        document_id: UUID,
        crud_search_fn: Callable,
        result_mapper: Callable[[T, float], R],
        result_limit: int,
        similarity_threshold: float,
    ) -> List[R]:
        """
        Generic semantic search method for chunks or images.
        """
        query_vector = self.embedder.generate_embedding(query)
        max_distance = 1.0 - similarity_threshold

        rows: List[Tuple[T, float]] = crud_search_fn(
            db=self.db,
            query_vector=query_vector,
            document_id=document_id,
            limit=result_limit,
            max_distance=max_distance,
        )

        if not rows:
            return []   

        results: List[R] = []
        for item, distance in rows:
            results.append(result_mapper(item, distance))
        return results

    def chunk_semantic_search(
        self,
        document_id: UUID,
        query: str,
        result_limit: int = DEFAULT_CHUNK_RESULT_LIMIT,
        similarity_threshold: float = DEFAULT_CHUNK_SIMILARITY_THRESHOLD,
    ) -> List[ChunkSearchResult]:
        """Run semantic similarity search for text chunks."""
        def mapper(chunk: Chunk, distance: float) -> ChunkSearchResult:
            return ChunkSearchResult(
                chunk_id=chunk.chunk_id,
                page_number=chunk.page_number,
                content=chunk.content,
                summary=chunk.summary,
                score=self._compute_score(distance),
            )

        return self._semantic_search(
            query=query,
            document_id=document_id,
            crud_search_fn=chunk_crud.search_similar,
            result_mapper=mapper,
            result_limit=result_limit,
            similarity_threshold=similarity_threshold,
        )

    def image_semantic_search(
        self,
        document_id: UUID,
        query: str,
        result_limit: int = DEFAULT_IMAGE_RESULT_LIMIT,
        similarity_threshold: float = DEFAULT_IMAGE_SIMILARITY_THRESHOLD,
    ) -> List[ImageSearchResult]:
        """Run semantic similarity search for images."""
        def mapper(image: Image, distance: float) -> ImageSearchResult:
            return ImageSearchResult(
                image_id=image.image_id,
                page_number=image.page_number,
                storage_path=image.storage_path,
                file_name=image.file_name,
                score=self._compute_score(distance),
                description=image.description,
            )

        return self._semantic_search(
            query=query,
            document_id=document_id,
            crud_search_fn=image_crud.search_similar,
            result_mapper=mapper,
            result_limit=result_limit,
            similarity_threshold=similarity_threshold,
        )
