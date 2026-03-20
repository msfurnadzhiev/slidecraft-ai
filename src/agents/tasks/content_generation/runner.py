"""Content-generation task runner.

Architecture (two explicit phases, no agentic loop):
  1. Retrieval  — programmatic multi-query search; results deduplicated by ID.
  2. Synthesis  — single direct LLM call with every retrieved chunk embedded in
                  the prompt; the model is instructed to include all of them.

This replaces the previous AgentExecutor approach, which let the model silently
discard the majority of retrieved chunks during synthesis.
"""

import logging
import re
import time
from typing import List
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from src.schemas.document.chunk import ChunkSearchResult
from src.schemas.document.image import ImageSearchResult
from src.schemas.presentation.slide import SlideContent, SlideStructure
from src.services.retrieval.semantic_search import SemanticSearchSevice

from src.agents.core.instrumentation import LLMInstrumentationCallback
from src.agents.tasks.content_generation.prompts import SYNTHESIS_SYSTEM_PROMPT, build_synthesis_prompt

log = logging.getLogger(__name__)


def _build_search_queries(slide: SlideStructure) -> List[str]:
    """Derive diverse search queries from slide metadata without an extra LLM call."""
    queries: List[str] = [slide.title]
    for part in re.split(r"[.;]", slide.description):
        part = part.strip()
        if len(part) > 15:
            queries.append(part)
    # Cap at 6 distinct queries to stay within rate limits
    return queries[:6]


class ContentGenerationTask:
    """Retrieves all relevant content programmatically, then synthesizes with a direct LLM call."""

    def __init__(self, chat_model: object) -> None:
        self.chat_model = chat_model

    def generate(
        self,
        *,
        document_id: UUID,
        slide_structure: SlideStructure,
        search_service: SemanticSearchSevice,
        max_chunks: int,
        max_images: int,
    ) -> SlideContent:
        """Return a SlideContent built from all unique retrieved chunks."""
        instrumentation = LLMInstrumentationCallback()

        slide_label = f"slide #{slide_structure.slide_number} '{slide_structure.title}'"
        t0 = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Phase 1: Programmatic multi-query retrieval                         #
        # ------------------------------------------------------------------ #
        queries = _build_search_queries(slide_structure)

        chunk_map: dict[str, ChunkSearchResult] = {}
        for query in queries:
            hits = search_service.chunk_semantic_search(
                document_id=document_id,
                query=query,
                result_limit=max_chunks,
                # Disable the similarity threshold so we always get the top-N
                # most relevant chunks regardless of absolute cosine distance.
                similarity_threshold=0.0,
            )
            log.debug(
                "Query %r → %d chunk hits — %s",
                query,
                len(hits),
                slide_label,
            )
            for chunk in hits:
                cid = str(chunk.chunk_id)
                if cid not in chunk_map or chunk.score > chunk_map[cid].score:
                    chunk_map[cid] = chunk

        image_map: dict[str, ImageSearchResult] = {}
        for query in [slide_structure.title, slide_structure.description[:120]]:
            for img in search_service.image_semantic_search(
                document_id=document_id,
                query=query,
                result_limit=max_images,
                similarity_threshold=0.0,
            ):
                iid = str(img.image_id)
                if iid not in image_map or img.score > image_map[iid].score:
                    image_map[iid] = img

        chunks = sorted(chunk_map.values(), key=lambda c: c.score, reverse=True)
        images = sorted(image_map.values(), key=lambda i: i.score, reverse=True)

        log.info(
            "Retrieved %d unique chunks and %d images via %d queries — %s",
            len(chunks),
            len(images),
            len(queries),
            slide_label,
        )

        # ------------------------------------------------------------------ #
        # Phase 2: Direct synthesis — all chunks embedded in a single prompt  #
        # ------------------------------------------------------------------ #
        # Callbacks must reach the runnable that performs the LLM call.
        # `chat_model.with_config(callbacks=...).with_structured_output(...)` drops them:
        # RunnableBinding.__getattr__ only wraps methods that accept *config*;
        # with_structured_output does not, so it returns a fresh chain with no callbacks
        # and instrumentation showed "0 API calls" while Gemini was still invoked.
        structured_llm = self.chat_model.with_structured_output(SlideContent)
        messages = [
            SystemMessage(content=SYNTHESIS_SYSTEM_PROMPT),
            HumanMessage(content=build_synthesis_prompt(slide_structure, chunks, images)),
        ]

        result: SlideContent = structured_llm.invoke(
            messages,
            config={"callbacks": [instrumentation]},
        )

        elapsed = time.perf_counter() - t0
        log.debug("Content generation finished — %s in %.2fs", slide_label, elapsed)
        instrumentation.log_summary(slide_label)

        # Guarantee slide metadata fields are always correct
        return result.model_copy(
            update={
                "slide_number": slide_structure.slide_number,
                "slide_type": slide_structure.slide_type,
                "title": result.title or slide_structure.title,
                "description": result.description or slide_structure.description,
            }
        )
