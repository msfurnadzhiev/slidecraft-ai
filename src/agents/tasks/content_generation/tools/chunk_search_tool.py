"""Chunk search tool for agent workflows."""

from typing import List, Optional
from uuid import UUID

from langchain_core.tools import BaseTool, tool

from src.schemas.document import ChunkSearchResult
from src.services.retrieval.semantic_search import SemanticSearchSevice

def format_chunks(
    chunks: List[ChunkSearchResult],
    include_index: bool = False,
) -> str:
    """
    Convert chunk search results into a readable string for LLM consumption.

    Args:
        chunks: Retrieved chunk results
        include_index: Whether to prefix each chunk with an index

    Returns:
        Formatted string representation of chunks
    """
    if not chunks:
        return "(no matching chunks)"

    return "\n\n".join(
        _format_single_chunk(chunk, idx if include_index else None)
        for idx, chunk in enumerate(chunks, 1)
    )


def _format_single_chunk(
    chunk: ChunkSearchResult,
    index: Optional[int] = None,
) -> str:
    """Format a single chunk into a structured text block."""
    prefix = f"[{index}] " if index else ""

    return (
        f"{prefix}chunk_id={chunk.chunk_id}, "
        f"page={chunk.page_number}, "
        f"score={chunk.score:.3f}\n"
        f"content: {chunk.content}"
    )


def build_chunk_search_tool(
    document_id: UUID,
    search_service: SemanticSearchSevice,
    result_limit: int,
) -> BaseTool:
    """Create a document-scoped chunk search tool."""

    @tool("search_relevant_chunks")
    def search_relevant_chunks(query: str) -> str:
        """Find semantically relevant document text chunks for a query."""
        if not query.strip():
            return "(empty query)"

        chunks = search_service.chunk_semantic_search(
            document_id=document_id,
            query=query,
            result_limit=result_limit,
        )
        return format_chunks(
            chunks,
            include_index=True,
        )

    return search_relevant_chunks
