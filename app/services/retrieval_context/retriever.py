"""Retrieve raw context and delegate assembly to ContextAssembler."""

from typing import TYPE_CHECKING

from app.schemas.context import ContextRequest, RawContext

if TYPE_CHECKING:
    from app.services.data_ingestion.embeddings import ImageEmbedder, TextEmbedder
    from app.services.retrieval_context.image_search import ImageSearch
    from app.services.retrieval_context.text_search import TextSearch


class ContextRetriever:
    """Orchestrate retrieval using text + image semantic search services."""

    def __init__(
        self,
        text_search: "TextSearch",
        image_search: "ImageSearch",
        text_embedder: "TextEmbedder",
        image_embedder: "ImageEmbedder",
    ):
        self.text_search = text_search
        self.image_search = image_search
        self.text_embedder = text_embedder
        self.image_embedder = image_embedder

    def retrieve_context(
        self,
        document_id: str,
        context_request: ContextRequest | None = None,
    ) -> RawContext:
        """
        Retrieve chunk and image context for a document.

        If query is omitted, all chunks/images are returned (fetch_all mode).
        """
        if context_request and context_request.query:
            text_vector = self.text_embedder.generate_embedding(context_request.query)
            image_vector = self.image_embedder.generate_text_query_embedding(
                context_request.query
            )

            max_distance_chunk = 1.0 - context_request.chunk_threshold
            max_distance_image = 1.0 - context_request.image_threshold

            chunks = self.text_search.search(
                document_id=document_id,
                query_vector=text_vector,
                limit=context_request.chunk_limit,
                max_distance=max_distance_chunk,
            )
            images = self.image_search.search(
                document_id=document_id,
                query_vector=image_vector,
                limit=context_request.image_limit,
                max_distance=max_distance_image,
            )
        else:
            chunks = self.text_search.fetch_all(document_id)
            images = self.image_search.fetch_all(document_id)

        return RawContext(
            document_id=document_id,
            query=context_request.query if context_request else None,
            chunks=chunks,
            images=images,
        )
