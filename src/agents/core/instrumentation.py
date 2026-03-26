"""LangChain callback handler for LLM call and tool usage instrumentation."""

import logging
from typing import Any, Dict, List, TYPE_CHECKING

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from src.agents.core.utils import estimate_tokens

if TYPE_CHECKING:
    from src.agents.core.rate_limiter import RateLimiter

log = logging.getLogger(__name__)

class LLMInstrumentationCallback(BaseCallbackHandler):
    """Tracks API calls, per-call token usage, and tool invocations for one agent run.

    When rate_limiter is provided the callback also enforces RPM/TPM limits:
    it blocks in on_chat_model_start until capacity is available and records
    actual token usage in on_llm_end.
    """

    def __init__(self, rate_limiter: "RateLimiter | None" = None) -> None:
        super().__init__()
        self.rate_limiter = rate_limiter
        self._last_estimated_tokens: int = 0
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
        # msg.content may be a str (simple messages) or a list of content
        # parts (AI tool-call messages in multi-turn executor runs).  Joining
        # a list directly raises TypeError and silently kills the callback.
        parts: List[str] = []
        for batch in messages:
            for msg in batch:
                content = getattr(msg, "content", "") or ""
                if isinstance(content, str):
                    parts.append(content)
                elif isinstance(content, (list, tuple)):
                    for part in content:
                        parts.append(part if isinstance(part, str) else str(part))
                else:
                    parts.append(str(content))

        self._on_llm_call_start("".join(parts))

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        # Some LangChain versions (including langchain_classic) route chat
        # model calls through on_llm_start instead of on_chat_model_start.
        self._on_llm_call_start("".join(prompts))

    def _on_llm_call_start(self, text: str) -> None:
        """Shared logic for both LLM start hooks."""
        self.api_calls += 1
        estimated = estimate_tokens(text)
        self._last_estimated_tokens = estimated

        if self.rate_limiter is not None:
            self.rate_limiter.acquire(estimated)

        log.debug(
            "API call #%d — ~%d estimated input tokens",
            self.api_calls,
            estimated,
        )

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        actual_tokens: int = 0

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
                actual_tokens += input_t + output_t
                self.total_input_tokens += input_t
                self.total_output_tokens += output_t
                log.debug(
                    "API call #%d — actual tokens: %d in / %d out",
                    self.api_calls,
                    input_t,
                    output_t,
                )

        if self.rate_limiter is not None:
            self.rate_limiter.record(actual_tokens or self._last_estimated_tokens)

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
