"""Agent that generates slide content via LangChain tool-calling."""

import logging
from uuid import UUID

from src.agents.core import BaseAgent
from src.schemas.presentation.slide import SlideContent, SlideStructure
from src.services.retrieval.semantic_search import SemanticSearchSevice
from src.agents.tasks.content_generation.runner import ContentGenerationTask
from src.utils.profiling import trace_runtime

# Raised limits for large documents (400-page books need wider retrieval coverage)
CHUNKS_SEARCH_LIMIT = 20
IMAGES_SEARCH_LIMIT = 10
MAX_TOOL_CALLS = 10

log = logging.getLogger(__name__)


class ContentGeneratorAgent(BaseAgent, name="content-generator"):
    """Generates structured content from slide metadata using semantic search tools."""

    def __init__(
        self,
        search_service: SemanticSearchSevice,
    ) -> None:
        super().__init__()
        self.search_service = search_service
        self._task = ContentGenerationTask(
            chat_model=self.chat_model,
            rate_limiter=self.rate_limiter,
            call_with_retry=self.retry_policy.execute,
        )

    @trace_runtime
    def generate_structure(
        self,
        document_id: UUID,
        slide_structure: SlideStructure,
        max_chunks: int = CHUNKS_SEARCH_LIMIT,
        max_images: int = IMAGES_SEARCH_LIMIT,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> SlideContent:
        """Generate structured content for a single slide."""
        log.debug(
            "Generating content — slide #%d '%s' [%s] (document_id=%s)",
            slide_structure.slide_number,
            slide_structure.title,
            slide_structure.slide_type,
            document_id,
        )
        result = self._task.generate(
            document_id=document_id,
            slide_structure=slide_structure,
            search_service=self.search_service,
            max_chunks=max_chunks,
            max_images=max_images,
            max_tool_calls=max_tool_calls,
        )
        log.debug(
            "Slide #%d done — %d text chunks, %d images",
            slide_structure.slide_number,
            len(result.content or []),
            len(result.images or []),
        )
        return result