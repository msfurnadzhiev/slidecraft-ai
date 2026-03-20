"""Image search tool for agent workflows."""

from typing import List
from uuid import UUID

from langchain_core.tools import BaseTool, tool

from src.schemas.document.image import ImageSearchResult
from src.services.retrieval.semantic_search import SemanticSearchSevice

def format_images(
    images: List[ImageSearchResult],
    *,
    include_index: bool = False,
    empty_label: str = "(no images)",
) -> str:
    """Format image search results for prompts or tool output."""
    if not images:
        return empty_label

    lines: List[str] = []
    for idx, image in enumerate(images, 1):
        prefix = f"[{idx}] " if include_index else ""
        description = image.description or "(no description)"
        lines.append(
            f"{prefix}image_id={image.image_id}, page={image.page_number}, score={image.score:.3f}, "
            f"file={image.file_name}\n"
            f"image_url: {image.storage_path}\n"
            f"description: {description}"
        )
    return "\n\n".join(lines)

def build_image_search_tool(
    *,
    document_id: UUID,
    search_service: SemanticSearchSevice,
    result_limit: int,
) -> BaseTool:
    """Create a document-scoped image search tool."""

    @tool("search_relevant_images")
    def search_relevant_images(query: str) -> str:
        """Find semantically relevant document images for a query."""
        if not query.strip():
            return "(empty query)"

        images = search_service.image_semantic_search(
            document_id=document_id,
            query=query,
            result_limit=result_limit,
        )
        return format_images(
            images,
            include_index=True,
            empty_label="(no matching images)",
        )

    return search_relevant_images
