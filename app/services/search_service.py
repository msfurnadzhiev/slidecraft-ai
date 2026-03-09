"""
Internal service for document-scoped semantic search (RAG retrieval).

- Uses sentence-transformers for text-to-text chunk search (384-dim vectors).
- Uses CLIP for text-to-image cross-modal search (512-dim vectors).
- Each model queries its own embedding table; vector spaces are independent.
"""

from typing import List, Tuple, TypeVar

from sqlalchemy.orm import Session

from app.db.crud import document_crud, chunk_crud, image_crud
from app.db.models import Chunk, Image
from app.ingestion.embeddings import TextEmbedder, ImageEmbedder
from app.ingestion import FileLoader
from app.schemas.search import ImageResultItem, SearchResponse, SearchResultItem, SearchRequest
from app.storage import FileStorage

# Default retrieval limits and similarity thresholds
_SIMILARITY_OFFSET = 1.0

# Type variable for either Chunk or Image with its distance
T = TypeVar("T", Chunk, Image)
SimilarityRows = List[Tuple[T, float]]

class SearchService:
    """Semantic search over a single document's chunks and images."""

    def __init__(
        self,
        db: Session,
        text_embedder: TextEmbedder,
        image_embedder: ImageEmbedder,
        file_loader: FileLoader,
        file_storage: FileStorage,
    ):
        """Initialize the search service."""
        self.db = db
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder
        self.loader = file_loader
        self.file_storage = file_storage


    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Perform semantic search for text chunks and images within a single document.

        Text chunks use sentence-transformers (384-dim vectors).
        Images use CLIP (512-dim vectors). Each model queries its own embedding table.

        Args:
            request: SearchRequest containing document_id, query, limits, and thresholds.

        Returns:
            SearchResponse with ranked chunks and images.
        """
        # Resolve document path
        document_id, pdf_path = self._resolve_document(request.document_id)

        # Compute maximum allowed distances for filtering
        chunk_max_distance = _SIMILARITY_OFFSET - request.chunk_threshold
        image_max_distance = (
            _SIMILARITY_OFFSET - request.image_threshold
            if request.image_threshold is not None
            else None
        )

        # Text chunk search
        text_query_vector: List[float] = \
            self.text_embedder.generate_embedding(request.query)
        
        chunk_rows: SimilarityRows = chunk_crud.search_similar(
            db=self.db,
            query_vector=text_query_vector,
            document_id=document_id,
            limit=request.chunk_limit,
            max_distance=chunk_max_distance,
        )

        # Cross-modal image search
        clip_query_vector: List[float] = \
            self.image_embedder.generate_text_query_embedding(request.query)
        
        image_rows: SimilarityRows = image_crud.search_similar(
            db=self.db,
            query_vector=clip_query_vector,
            document_id=document_id,
            limit=request.image_limit,
            max_distance=image_max_distance,
        )

        # Build the search response
        return SearchResponse(
            document_id=document_id,
            query=request.query,
            results=self._build_chunk_results(chunk_rows, pdf_path),
            image_results=self._build_image_results(image_rows),
        )

    def _resolve_document(self, document_id: str) -> Tuple[str, str]:
        """Return document ID and absolute PDF path."""
        document = document_crud.get_document(self.db, document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        path = self.file_storage.get_absolute_path(document.storage_path)
        return document_id, path

    def _build_chunk_results(
        self, chunk_rows: SimilarityRows, pdf_path: str
    ) -> List[SearchResultItem]:
        """Load chunk text from PDF and build SearchResultItem list.

        Args:
            chunk_rows: List of tuples: (Chunk, distance)
            pdf_path: Path to the PDF file for re-extracting chunk text.

        Returns:
            List of SearchResultItem with text populated.
        """
        if not chunk_rows:
            return []

        chunks = [chunk for chunk, _ in chunk_rows]
        chunk_text_map = self.loader.extract_chunks_text(pdf_path, chunks)

        results: List[SearchResultItem] = []
        for chunk, distance in chunk_rows:
            score = round(_SIMILARITY_OFFSET - float(distance), 4)
            text = chunk_text_map.get(chunk.chunk_id, "")
            results.append(
                SearchResultItem(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    text=text,
                    score=score,
                )
            )

        return results

    def _build_image_results(self, image_rows: SimilarityRows) -> List[ImageResultItem]:
        """Build ImageResultItem list from retrieved images.
        
        Args:
            image_rows: List of tuples: (Image, distance)

        Returns:
            List of ImageResultItem
        """
        if not image_rows:
            return []

        results: List[ImageResultItem] = []
        for image, distance in image_rows:
            score = round(_SIMILARITY_OFFSET - float(distance), 4)
            results.append(
                ImageResultItem(
                    image_id=image.image_id,
                    page_number=image.page_number,
                    storage_path=image.storage_path,
                    file_name=image.file_name,
                    score=score,
                )
            )

        return results