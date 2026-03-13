"""Retrieve retrieved context."""

from typing import TYPE_CHECKING
from uuid import UUID

from src.schemas.context import ContextRetrievalOptions, RetrievedContext

if TYPE_CHECKING:
    from src.services.data_ingestion.text_embedder import TextEmbedder
    from src.services.retrieval_context.image_search import ImageSearch
    from src.services.retrieval_context.text_search import TextSearch


class ContextRetriever:
    """Orchestrate retrieval using text + image semantic search services."""

    def __init__(
        self,
        text_search: "TextSearch",
        image_search: "ImageSearch",
        embedder: "TextEmbedder",
    ):
        self.text_search = text_search
        self.image_search = image_search
        self.embedder = embedder

    def retrieve_context(
        self,
        document_id: UUID,
        options: ContextRetrievalOptions | None = None,
    ) -> RetrievedContext:
        """
        Retrieve chunk and image context for a document.

        If query is omitted, all chunks/images are returned (fetch_all mode).
        """
        if options and options.query:
            text_vector = self.embedder.generate_embedding(options.query)
            image_vector = self.embedder.generate_embedding(options.query)

            max_distance_chunk = 1.0 - options.chunk_threshold
            max_distance_image = 1.0 - options.image_threshold

            chunks = self.text_search.search(
                document_id=document_id,
                query_vector=text_vector,
                limit=options.chunk_limit,
                max_distance=max_distance_chunk,
            )
            # images = self.image_search.search(
            #     document_id=document_id,
            #     query_vector=image_vector,
            #     limit=options.image_limit,
            #     max_distance=max_distance_image,
            # )
        else:
            chunks = self.text_search.fetch_all(document_id)
            # images = self.image_search.fetch_all(document_id)

        return RetrievedContext(
            document_id=document_id,
            options=options if options else None,
            chunks=chunks,
            # images=images,
        )
