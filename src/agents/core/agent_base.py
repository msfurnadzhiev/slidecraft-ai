"""Base class shared by all LLM agents."""

import logging
import os
from typing import TYPE_CHECKING

from langchain_google_genai import ChatGoogleGenerativeAI

from src.agents.core.rate_limiter import RateLimiter
from src.agents.core.retry_policy import RetryPolicy
from src.agents.core.agent_models import AgentMeta

if TYPE_CHECKING:
    from src.agents.core.agent_models import Model

log = logging.getLogger(__name__)


class BaseAgent(metaclass=AgentMeta):
    """Shared infrastructure for all LLM agents.

    Subclasses declare their agent name via the ``name`` keyword in the class
    header.  ``AgentMeta`` then resolves the correct ``Model`` from environment
    variables and enforces singleton instantiation.

    The three resources created in ``__init__`` are injected into the agent's
    task runner so every individual LLM call is throttled and retried
    consistently across all agents.
    """

    AGENT_KEY: str | None = None
    MODEL: "Model" = None

    def __init__(self) -> None:
        if not self.AGENT_KEY:
            raise RuntimeError(
                "Agent must define a unique AGENT_KEY via 'name' in the class header."
            )

        self.model = self.__class__.MODEL
        self.api_key = self._get_api_key()
        self.chat_model = self._create_chat_model()
        self.rate_limiter = RateLimiter(rate_limits=self.model.rate_limits)
        self.retry_policy = RetryPolicy()

    def _get_api_key(self) -> str:
        api_key = os.getenv("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY environment variable is required.")
        return api_key

    def _create_chat_model(self, *, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
        return ChatGoogleGenerativeAI(
            model=self.model.name,
            google_api_key=self.api_key,
            temperature=temperature,
            max_retries=0,  # Disable LangChain built-in retry — RetryPolicy handles this
        )
