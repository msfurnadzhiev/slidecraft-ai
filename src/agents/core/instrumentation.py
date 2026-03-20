"""LangChain callback handler for LLM call and tool usage instrumentation."""

import logging
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.agents.core.utils import estimate_tokens

log = logging.getLogger(__name__)

class LLMInstrumentationCallback(BaseCallbackHandler):
    """Tracks API calls, per-call token usage, and tool invocations for one agent run."""

    def __init__(self) -> None:
        super().__init__()
        self.api_calls: int = 0
        self.tool_calls: Dict[str, int] = {}
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    # ------------------------------------------------------------------
    # LLM hooks
    # ------------------------------------------------------------------

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        **kwargs: Any,
    ) -> None:
        self.api_calls += 1
        contents = (
            getattr(msg, "content", "") or ""
            for batch in messages
            for msg in batch
        )
        estimated = estimate_tokens("".join(contents))
        log.debug(
            "API call #%d — ~%d estimated input tokens",
            self.api_calls,
            estimated,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue
                usage = getattr(msg, "usage_metadata", None)
                if not usage:
                    continue
                input_t: int = usage.get("input_tokens", 0)
                output_t: int = usage.get("output_tokens", 0)
                self.total_input_tokens += input_t
                self.total_output_tokens += output_t
                log.debug(
                    "API call #%d — actual tokens: %d in / %d out",
                    self.api_calls,
                    input_t,
                    output_t,
                )

    # ------------------------------------------------------------------
    # Tool hooks
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        name: str = serialized.get("name", "unknown_tool")
        self.tool_calls[name] = self.tool_calls.get(name, 0) + 1
        log.debug(
            "Tool '%s' call #%d — query: %.150s",
            name,
            self.tool_calls[name],
            input_str,
        )

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        log.debug("Tool response — %d chars", len(str(output)))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def log_summary(self, label: str = "") -> None:
        """Emit a single INFO line with all collected statistics."""
        prefix = f"[{label}] " if label else ""
        tool_summary = ", ".join(
            f"{name}×{count}" for name, count in sorted(self.tool_calls.items())
        )
        log.info(
            "%s%d API calls | tools: [%s] | tokens: %d in / %d out (total %d)",
            prefix,
            self.api_calls,
            tool_summary or "none",
            self.total_input_tokens,
            self.total_output_tokens,
            self.total_input_tokens + self.total_output_tokens,
        )
