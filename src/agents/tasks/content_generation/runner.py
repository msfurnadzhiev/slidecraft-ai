"""Content-generation task runner."""

import json
import logging
from uuid import UUID
from typing import Callable, List, Optional

from sqlalchemy.orm import Session
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.core.instrumentation import LLMInstrumentationCallback
from src.agents.core.rate_limiter import RateLimiter
from src.utils.profiling import trace_runtime
from src.agents.tasks.content_generation.prompts import (
    AGENT_SYSTEM_PROMPT,
    build_agent_input,
)
from src.agents.tasks.content_generation.tools import (
    build_chunk_search_tool,
    build_image_search_tool,
)
from src.agents.tasks.content_generation.utils import (
    validate_chunks,
    validate_images,
)
from src.schemas.presentation.slide import (
    ImageContent,
    SlideContent,
    SlideStructure,
    TextContent,
)
from src.services.retrieval.semantic_search import SemanticSearchSevice

log = logging.getLogger(__name__)


_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


class ContentGenerationTask:
    """Generates structured slide content using a tool-calling LLM agent."""

    def __init__(
        self,
        chat_model: object,
        rate_limiter: RateLimiter,
        call_with_retry: Callable,
    ) -> None:
        self.chat_model = chat_model
        self.rate_limiter = rate_limiter
        self._call_with_retry = call_with_retry


    @trace_runtime
    def generate(
        self,
        document_id: UUID,
        slide_structure: SlideStructure,
        search_service: SemanticSearchSevice,
        max_chunks: int,
        max_images: int,
        max_tool_calls: int,
    ) -> SlideContent:
        """
        Generate content for a single slide.

        The agent retrieves relevant chunks and images, produces structured output,
        and selected references are validated against the database.

        Args:
            document_id: The ID of the document to search.
            slide_structure: The slide structure to generate content for.
            search_service: The search service to use.
            max_chunks: The maximum number of chunks to search.
            max_images: The maximum number of images to search.
            max_tool_calls: The maximum number of tool calls to make.

        Returns:
            A validated and fully populated SlideContent instance.
        """
        instrumentation = LLMInstrumentationCallback(rate_limiter=self.rate_limiter)

        # Build retrieval tools and execute agent
        tools = self._build_tools(
            document_id, search_service, max_chunks, max_images
        )
        agent_output = self._execute_agent(
            tools, slide_structure, instrumentation, max_tool_calls
        )

        parsed = self._parse_output(agent_output, instrumentation)

        # Validate chunks and images against the database
        db = search_service.db
        validated_content = validate_chunks(parsed.content, db)
        validated_images = validate_images(parsed.images, db)

        result = self._assemble_slide(
            parsed,
            slide_structure,
            validated_content,
            validated_images,
        )

        return result

  
    def _execute_agent(
        self,
        tools: List,
        slide_structure: SlideStructure,
        instrumentation: LLMInstrumentationCallback,
        max_tool_calls: int,
    ) -> str:
        """Run the tool-calling agent and return raw output text.

        Args:
            tools: The tools to use.
            slide_structure: The slide structure to generate content for.
            instrumentation: The instrumentation callback (carries rate_limiter).
            max_tool_calls: The maximum number of tool calls to make.

        Returns:
            The raw output text from the agent.
        """
        prompt = _AGENT_PROMPT.partial(
            input=build_agent_input(slide_structure)
        )

        # Attach callbacks at the model level so on_chat_model_start fires for
        # every LLM turn inside the executor loop, not just at the executor level.
        instrumented_model = self.chat_model.with_config(callbacks=[instrumentation])
        agent = create_tool_calling_agent(instrumented_model, tools, prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_tool_calls,
            handle_parsing_errors=True,
            max_execution_time=120,
        )

        # Pass callbacks at invoke time — the most reliable propagation path
        # in LangChain's Runnable system; with_config on the model alone does
        # not always survive bind_tools() inside create_tool_calling_agent.
        result = self._call_with_retry(
            executor.invoke,
            {},
            config={"callbacks": [instrumentation]},
            on_retry=self.rate_limiter.force_reset,
        )

        return self._extract_text(result["output"])

    @staticmethod
    def _extract_text(output: object) -> str:
        """Normalise agent output to a plain string."""
        if isinstance(output, str):
            return output
        if isinstance(output, dict):
            return output.get("text", str(output))
        if isinstance(output, list):
            return "".join(
                part.get("text", str(part)) if isinstance(part, dict) else str(part)
                for part in output
            )
        return str(output)

    def _parse_output(
        self,
        agent_output: str,
        instrumentation: LLMInstrumentationCallback,
    ) -> SlideContent:
        """Parse and validate agent output into SlideContent schema.
        
        Args:
            agent_output: The raw output text from the agent.
            instrumentation: The instrumentation to use.

        Returns:
            A SlideContent object.
        """
        try:
            return SlideContent.model_validate(json.loads(agent_output))
        except Exception:
            log.debug("Failed to parse agent output: %s", agent_output)

        # Fallback: use LLM to parse the output
        structured_llm = self.chat_model.with_structured_output(SlideContent)

        return self._call_with_retry(
            structured_llm.invoke,
            agent_output,
            config={"callbacks": [instrumentation]},
            on_retry=self.rate_limiter.force_reset,
        )


    def _build_tools(
        self,
        document_id: UUID,
        search_service: SemanticSearchSevice,
        max_chunks: int,
        max_images: int,
    ) -> List:
        """Create document-scoped retrieval tools.
        
        Args:
            document_id: The ID of the document to search.
            search_service: The search service to use.
            max_chunks: The maximum number of chunks to search.
            max_images: The maximum number of images to search.

        Returns:
            A list of tools.
        """
        return [
            build_chunk_search_tool(
                document_id=document_id,
                search_service=search_service,
                result_limit=max_chunks,
            ),
            build_image_search_tool(
                document_id=document_id,
                search_service=search_service,
                result_limit=max_images,
            ),
        ]


    def _assemble_slide(
        self,
        parsed: SlideContent,
        structure: SlideStructure,
        content: Optional[List[TextContent]],
        images: Optional[List[ImageContent]],
    ) -> SlideContent:
        """Merge generated content with slide metadata and validated references.
        
        Args:
            parsed: The parsed SlideContent object.
            structure: The slide structure to generate content for.
            content: The content to merge.
            images: The images to merge.

        Returns:
            A SlideContent object.
        """
        return parsed.model_copy(
            update={
                "slide_number": structure.slide_number,
                "slide_type": structure.slide_type,
                "title": parsed.title or structure.title,
                "description": parsed.description or structure.description,
                "content": content,
                "images": images,
            }
        )
