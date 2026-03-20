"""Chunk search tool for agent workflows."""

from typing import List
from uuid import UUID

from langchain_core.tools import BaseTool, tool

from src.schemas.document.chunk import ChunkSearchResult
from src.services.retrieval.semantic_search import SemanticSearchSevice

def format_chunks(
    chunks: List[ChunkSearchResult],
    *,
    include_index: bool = False,
    text_label: str = "text",
    empty_label: str = "(no text passages)",
) -> str:
    """Format chunk search results for prompts or tool output."""
    if not chunks:
        return empty_label

    lines: List[str] = []
    for idx, chunk in enumerate(chunks, 1):
        prefix = f"[{idx}] " if include_index else ""
        text = getattr(chunk, "text", None) or getattr(chunk, "content", "")
        lines.append(
            f"{prefix}chunk_id={chunk.chunk_id}, page={chunk.page_number}, score={chunk.score:.3f}\n"
            f"{text_label}: {text}"
        )
    return "\n\n".join(lines)


def build_chunk_search_tool(
    *,
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
            text_label="content",
            empty_label="(no matching chunks)",
        )

    return search_relevant_chunks
