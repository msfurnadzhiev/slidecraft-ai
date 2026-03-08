"""Internal service for document-scoped semantic search (RAG retrieval)."""

from sqlalchemy.orm import Session

from app.db.crud import document_crud, chunk_crud, embedding_crud
from app.db.models import Chunk, Embedding
from app.ingestion import EmbeddingGenerator, FileLoader
from app.schemas.search import SearchResponse, SearchResultItem
from app.storage import FileStorage


_LIMIT_CHUNK_RESULTS = 50
_SIMILARITY_THRESHOLD = 0.70
_SIMILARITY_OFFSET = 1.0
_OBJECT_TYPE_CHUNK = "chunk"


class SearchService:
    """Semantic search over a single document's chunks."""

    def __init__(
        self,
        db: Session,
        embedding_generator: EmbeddingGenerator,
        file_loader: FileLoader,
        file_storage: FileStorage,
    ):
        self.db = db
        self.embedder = embedding_generator
        self.loader = file_loader
        self.file_storage = file_storage

    def search(
        self,
        document_id: str,
        query: str,
        object_type: str = _OBJECT_TYPE_CHUNK,
        limit: int = _LIMIT_CHUNK_RESULTS,
        threshold = _SIMILARITY_THRESHOLD,
    ) -> SearchResponse:
        """Run semantic search; return top-k chunks with optional min similarity filter.

        Args:
            document_id: The ID of the document to search
            query: The query to search for
            object_type: The type of object to search for
            limit: The number of results to return
            threshold: The minimum similarity threshold for the results 

        Returns:
            A SearchResponse object containing the search results

        """
        _, pdf_path = self._resolve_document(document_id)
        query_vector = self.embedder.generate_embedding(query)

        # pgvector uses cosine distance
        max_distance = _SIMILARITY_OFFSET - threshold

        embedding_rows = embedding_crud.search_similar_embeddings(
            self.db,
            query_vector,
            document_id,
            object_type=object_type,
            limit=limit,
            max_distance=max_distance,
        )
        if not embedding_rows:
            return SearchResponse(document_id=document_id, query=query, results=[])

        chunk_by_id, text_by_chunk_id = self._load_chunks_and_text(
            embedding_rows, pdf_path
        )
        results = self._build_result_items(
            embedding_rows, chunk_by_id, text_by_chunk_id
        )
        return SearchResponse(document_id=document_id, query=query, results=results)

    def _resolve_document(self, document_id: str) -> tuple[str, str]:
        """Resolve document and return (document_id, absolute pdf_path).
        
        Args:
            document_id: The ID of the document to resolve

        Returns:
            A tuple containing the document_id and the absolute path to the PDF file
    
        """
        document = document_crud.get_document(self.db, document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        path = self.file_storage.get_absolute_path(document.storage_path)
        return document_id, path

    def _load_chunks_and_text(
        self,
        embedding_rows: list[tuple[Embedding, float]],
        pdf_path: str,
    ) -> tuple[dict[str, Chunk], dict[str, str]]:
        """Load chunks and their text from the PDF.
        
        Args:
            embedding_rows: A list of tuples containing the embedding and the distance
            pdf_path: The path to the PDF file

        Returns:
            A tuple containing the chunk_by_id and the text_by_chunk_id

        Raises:
            ValueError: If the document is not found

        """
        chunk_ids = [emb.object_id for emb, _ in embedding_rows]
        chunks = chunk_crud.get_chunks_by_ids(self.db, chunk_ids)
        chunk_by_id = {c.chunk_id: c for c in chunks}
        text_by_chunk_id = self.loader.extract_chunks_text(pdf_path, chunks)
        return chunk_by_id, text_by_chunk_id

    def _build_result_items(
        self,
        embedding_rows: list[tuple[Embedding, float]],
        chunk_by_id: dict[str, Chunk],
        text_by_chunk_id: dict[str, str],
    ) -> list[SearchResultItem]:
        """Convert embedding search results and chunk data into SearchResultItems.
        
        Args:
            embedding_rows: A list of tuples containing the embedding and the distance
            chunk_by_id: A dictionary containing the chunks by their ID
            text_by_chunk_id: A dictionary containing the text by the chunk ID

        Returns:
            A list of SearchResultItem objects

        """
        results: list[SearchResultItem] = []
        for emb, distance in embedding_rows:
            chunk = chunk_by_id.get(emb.object_id)
            if chunk is None:
                continue
            score = round(_SIMILARITY_OFFSET - float(distance), 4)
            text = text_by_chunk_id.get(chunk.chunk_id, "")
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
