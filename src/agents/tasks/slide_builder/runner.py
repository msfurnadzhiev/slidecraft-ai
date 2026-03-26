"""Task runner for slide-layout selection and content distribution."""

import logging
from typing import Callable, List, Optional, Set

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.agents.core.instrumentation import LLMInstrumentationCallback
from src.agents.core.rate_limiter import RateLimiter
from src.agents.tasks.slide_builder.prompts import (
    AGENT_SYSTEM_PROMPT,
    build_agent_input,
)
from src.agents.tasks.slide_builder.state import SlideAssignmentState
from src.agents.tasks.slide_builder.tools import (
    build_layout_selector_tool,
    build_placeholder_filler_tool,
)
from src.agents.tasks.slide_builder.utils import split_placeholders
from src.schemas.presentation.slide import SlideAssignment, SlideContent, PlaceholderFill
from src.schemas.template import SlideLayoutResponse
from src.utils.profiling import trace_runtime

log = logging.getLogger(__name__)

_AGENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", AGENT_SYSTEM_PROMPT),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)


class SlideAssignmentTask:
    """Assigns a layout and fills placeholders using a tool-calling LLM agent.

    The agent has two tools:
    - ``select_layout``     — picks a layout and receives the placeholder table.
    - ``fill_placeholder``  — writes content for one placeholder at a time.

    This lets the model self-correct: if a tool call returns an error the agent
    can retry with valid arguments before exhausting ``max_tool_calls``.
    """

    def __init__(
        self,
        chat_model,
        rate_limiter: RateLimiter,
        call_with_retry: Callable,
    ) -> None:
        self.chat_model = chat_model
        self.rate_limiter = rate_limiter
        self._call_with_retry = call_with_retry

    @trace_runtime
    def run(
        self,
        slide: SlideContent,
        layouts: List[SlideLayoutResponse],
        used_layout_indices: Optional[Set[int]] = None,
        max_tool_calls: int = 10,
        revision_feedback: Optional[str] = None,
    ) -> SlideAssignment:
        """Execute layout selection and content assignment for a single slide.

        Args:
            slide: Fully prepared slide content (title, description, chunks, images).
            layouts: Available layout definitions extracted from the template.
            used_layout_indices: Layout indices already chosen for earlier slides.
            max_tool_calls: Maximum agent iterations (select + fill + retries).

        Returns:
            SlideAssignment with layout choice, placeholder fills, and reasoning.
        """
        instrumentation = LLMInstrumentationCallback(rate_limiter=self.rate_limiter)
        state = SlideAssignmentState(layouts=layouts)

        tools = self._build_tools(state)
        self._execute_agent(
            tools, slide, layouts, used_layout_indices, instrumentation,
            max_tool_calls, revision_feedback,
        )

        result = self._assemble_result(state, layouts)

        instrumentation.log_summary(
            f"slide assignment slide #{slide.slide_number} '{slide.title}'"
        )
        return result


    def _build_tools(self, state: SlideAssignmentState) -> List:
        """Create the two tools wired to *state* for this run.

        Args:
            state: Shared mutable state for this agent run.

        Returns:
            List of LangChain tools.
        """
        return [
            build_layout_selector_tool(state),
            build_placeholder_filler_tool(state),
        ]

    def _execute_agent(
        self,
        tools: List,
        slide: SlideContent,
        layouts: List[SlideLayoutResponse],
        used_layout_indices: Optional[Set[int]],
        instrumentation: LLMInstrumentationCallback,
        max_tool_calls: int,
        revision_feedback: Optional[str] = None,
    ) -> None:
        """Run the tool-calling agent to populate shared state.

        Args:
            tools: The select_layout and fill_placeholder tools.
            slide: Slide content passed to the prompt builder.
            layouts: Available layouts passed to the prompt builder.
            used_layout_indices: Already-used layouts passed to the prompt builder.
            instrumentation: Callback that throttles and records token usage.
            max_tool_calls: Maximum AgentExecutor iterations.
            revision_feedback: Actionable feedback from the quality validator
                to include at the top of the agent prompt.
        """
        prompt = _AGENT_PROMPT.partial(
            input=build_agent_input(slide, layouts, used_layout_indices, revision_feedback)
        )

        instrumented_model = self.chat_model.with_config(callbacks=[instrumentation])
        agent = create_tool_calling_agent(instrumented_model, tools, prompt)

        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            max_iterations=max_tool_calls,
            handle_parsing_errors=True,
            max_execution_time=120,
        )

        self._call_with_retry(
            executor.invoke,
            {},
            config={"callbacks": [instrumentation]},
            on_retry=self.rate_limiter.force_reset,
        )

    def _assemble_result(
        self,
        state: SlideAssignmentState,
        layouts: List[SlideLayoutResponse],
    ) -> SlideAssignment:
        """Build a SlideAssignment from the state accumulated by tool calls.

        Args:
            state: Populated by the tools during the agent run.
            layouts: All available layouts (used for fallback).

        Returns:
            A fully populated SlideAssignment instance.
        """
        if state.selected_layout_index is None:
            log.warning(
                "Agent did not call select_layout — falling back to first layout."
            )
            layout_index = layouts[0].layout_index if layouts else 0
            return SlideAssignment(
                layout_index=layout_index,
                placeholder_fills=[],
                reasoning="[fallback] Agent did not select a layout.",
            )

        layout = state.selected_layout
        _, fillable = split_placeholders(layout)

        placeholder_fills = []
        for el in fillable:
            fill_data = state.fills.get(el.placeholder_idx)
            if fill_data is None:
                log.warning(
                    "Agent did not fill placeholder_idx=%d in layout %d — leaving empty.",
                    el.placeholder_idx,
                    state.selected_layout_index,
                )
            placeholder_fills.append(
                PlaceholderFill(
                    placeholder_idx=el.placeholder_idx,
                    text=fill_data.text if fill_data else "",
                    reasoning=fill_data.reasoning if fill_data else "",
                )
            )

        return SlideAssignment(
            layout_index=state.selected_layout_index,
            placeholder_fills=placeholder_fills,
            reasoning=state.layout_reasoning or "Layout selected by agent.",
        )
