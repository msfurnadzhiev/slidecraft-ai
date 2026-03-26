"""Image search tool for agent workflows."""

from typing import List, Optional
from uuid import UUID

from langchain_core.tools import BaseTool, tool

from src.schemas.document import ImageSearchResult
from src.services.retrieval.semantic_search import SemanticSearchSevice

def format_images(
    images: List[ImageSearchResult],
    include_index: bool = False,
) -> str:
    """
    Convert image search results into a readable string for LLM consumption.

    Args:
        images: Retrieved image results
        include_index: Whether to prefix each image with an index

    Returns:
        Formatted string representation of images
    """
    if not images:
        return "(no images)"

    return "\n\n".join(
        _format_single_image(image, idx if include_index else None)
        for idx, image in enumerate(images, 1)
    )

def _format_single_image(
    image: ImageSearchResult,
    index: Optional[int] = None,
) -> str:
    """Format a single image result into a structured text block."""
    prefix = f"[{index}] " if index else ""

    description = image.description or "(no description)"

    return (
        f"{prefix}image_id={image.image_id}, "
        f"page={image.page_number}, "
        f"score={image.score:.3f}, "
        f"file={image.file_name}\n"
        f"image_url: {image.storage_path}\n"
        f"description: {description}"
    )

def build_image_search_tool(
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
        )

    return search_relevant_images
