"""Internal service for document-scoped semantic search (RAG retrieval)."""

from sqlalchemy.orm import Session

from app.db.crud import document_crud, chunk_crud, embedding_crud, image_crud
from app.db.models import Chunk, Embedding, Image
from app.ingestion import EmbeddingGenerator, FileLoader
from app.schemas.search import ImageResultItem, SearchResponse, SearchResultItem
from app.storage import FileStorage

_LIMIT_CHUNK_RESULTS = 50
_LIMIT_IMAGE_RESULTS = 10
_SIMILARITY_THRESHOLD = 0.70
_SIMILARITY_OFFSET = 1.0
_OBJECT_TYPE_CHUNK = "chunk"
_OBJECT_TYPE_IMAGE = "image"


class SearchService:
    """Semantic search over a single document's chunks and images."""

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
        chunk_limit: int = _LIMIT_CHUNK_RESULTS,
        image_limit: int = _LIMIT_IMAGE_RESULTS,
        threshold: float = _SIMILARITY_THRESHOLD,
    ) -> SearchResponse:
        """Run semantic search for both chunks and images; return combined results.

        The same CLIP query vector is used against chunk embeddings and image
        embeddings so that text and visual results are scored on the same scale.
        """
        _, pdf_path = self._resolve_document(document_id)
        query_vector = self.embedder.generate_embedding(query)
        max_distance = _SIMILARITY_OFFSET - threshold

        chunk_rows = embedding_crud.search_similar_embeddings(
            self.db, query_vector, document_id,
            object_type=_OBJECT_TYPE_CHUNK,
            limit=chunk_limit,
            max_distance=max_distance,
        )
        image_rows = embedding_crud.search_similar_embeddings(
            self.db, query_vector, document_id,
            object_type=_OBJECT_TYPE_IMAGE,
            limit=image_limit,
            max_distance=max_distance,
        )

        results = self._build_chunk_results(chunk_rows, pdf_path)
        image_results = self._build_image_results(image_rows)

        return SearchResponse(
            document_id=document_id,
            query=query,
            results=results,
            image_results=image_results,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_document(self, document_id: str) -> tuple[str, str]:
        """Resolve document and return (document_id, absolute pdf_path)."""
        document = document_crud.get_document(self.db, document_id)
        if document is None:
            raise ValueError(f"Document not found: {document_id}")
        path = self.file_storage.get_absolute_path(document.storage_path)
        return document_id, path

    def _build_chunk_results(
        self,
        embedding_rows: list[tuple[Embedding, float]],
        pdf_path: str,
    ) -> list[SearchResultItem]:
        """Load chunk metadata + text from PDF and build result items."""
        if not embedding_rows:
            return []

        chunk_ids = [emb.object_id for emb, _ in embedding_rows]
        chunks = chunk_crud.get_chunks_by_ids(self.db, chunk_ids)
        chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        text_by_chunk_id = self.loader.extract_chunks_text(pdf_path, chunks)

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

    def _build_image_results(
        self,
        embedding_rows: list[tuple[Embedding, float]],
    ) -> list[ImageResultItem]:
        """Load image metadata and build result items."""
        if not embedding_rows:
            return []

        image_ids = [emb.object_id for emb, _ in embedding_rows]
        images = image_crud.get_images_by_ids(self.db, image_ids)
        image_by_id: dict[str, Image] = {img.image_id: img for img in images}

        results: list[ImageResultItem] = []
        for emb, distance in embedding_rows:
            image = image_by_id.get(emb.object_id)
            if image is None:
                continue
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
