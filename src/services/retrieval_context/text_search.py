"""Text (chunk) semantic search for document-scoped RAG."""

from typing import List, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

from src.db.crud import chunk_crud, document_crud
from src.db.models import Chunk
from src.schemas.chunk import ChunkSearchResult

# Score = 1.0 - distance (higher = more similar)
_SIMILARITY_OFFSET = 1.0

ChunkSimilarityRows = List[Tuple[Chunk, float]]


class TextSearch:
    """
    Semantic search over text chunks belonging to a single document.

    This class performs vector similarity search against stored chunk
    embeddings and reconstructs the corresponding chunk text directly
    from the PDF file.
    """

    def __init__(
        self,
        db: Session,
    ):
        """Initialize the text search service."""
        self.db = db

    def search(
        self,
        document_id: UUID,
        query_vector: List[float],
        limit: int,
        max_distance: float,
    ) -> List[ChunkSearchResult]:
        """
        Run semantic similarity search for chunks within a document.

        Args:
            document_id: ID of the document to search within.
            query_vector: Embedding vector representing the search query.
            limit: Maximum number of results to return.
            max_distance: Maximum vector distance threshold for matches.

        Returns:
            A list of ChunkSearchResult objects containing chunk metadata,
            extracted text, and similarity scores.
        """
        chunk_rows = chunk_crud.search_similar(
            db=self.db,
            query_vector=query_vector,
            document_id=document_id,
            limit=limit,
            max_distance=max_distance,
        )
        return self._build_results(document_id, chunk_rows)

    def fetch_all(
        self, document_id: UUID
    ) -> List[ChunkSearchResult]:
        """
        Retrieve all chunks belonging to a document.

        This method bypasses semantic filtering and returns every chunk
        with a neutral similarity score.

        Args:
            document_id: ID of the document.

        Returns:
            A list of SearchResultItem objects for all document chunks,
            each assigned a default similarity score of 1.0.
        """
        chunks = chunk_crud.get_chunks_by_document(self.db, document_id)
        chunk_rows: ChunkSimilarityRows = [(c, 0.0) for c in chunks]
        return self._build_results(document_id, chunk_rows)

    def _build_results(
        self,
        document_id: UUID,
        chunk_rows: ChunkSimilarityRows
    ) -> List[ChunkSearchResult]:
        """
        Convert raw similarity query results into search result objects.

        This method extracts the corresponding chunk text from the PDF
        and combines it with chunk metadata and computed similarity scores.

        Args:
            document_id: ID of the document.
            chunk_rows: List of tuples: (Chunk, vector distance)

        Returns:
            A list of ChunkSearchResult objects.
        """
        if document_crud.get_document(self.db, document_id) is None:
            raise ValueError(f"Document not found: {document_id}")

        if not chunk_rows:
            return []

        results: List[ChunkSearchResult] = []
        for chunk, distance in chunk_rows:
            score = round(_SIMILARITY_OFFSET - float(distance), 4)
            text = chunk.content or ""
            summary = chunk.summary or ""
            results.append(
                ChunkSearchResult(
                    chunk_id=chunk.chunk_id,
                    page_number=chunk.page_number,
                    text=text,
                    summary=summary,
                    score=score,
                )
            )

        return results