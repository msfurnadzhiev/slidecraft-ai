"""Common base for LLM-based agents."""

import logging
import os
import re
from typing import Any, List, Type, TypeVar, TYPE_CHECKING

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel

from src.agents.core.utils import estimate_tokens
from src.agents.core.rate_limiter import RateLimiter
from src.agents.core.retry_policy import RetryPolicy
from src.agents.core.agent_models import AgentMeta, AGENT_MODELS

if TYPE_CHECKING:
    from src.agents.core.agent_models import Model

# Type variable for the response model
M = TypeVar("M", bound=BaseModel)

log = logging.getLogger(__name__)

class BaseLLMProcessor(metaclass=AgentMeta):
    """Shared infrastructure for LLM agents."""

    AGENT_KEY: str | None = None
    MODEL: "Model" = None

    def __init__(self, model_name: str | None = None) -> None:
        # Check if the agent has a unique AGENT_KEY
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
            max_retries=0,  # Disable LangChain retry
        )

    def llm_call_structured(
        self,
        contents: Any,
        response_model: Type[M],
        *,
        estimated_tokens: int | None = None,
    ) -> M:
        """Rate-limited LLM call with retry policy."""

        if estimated_tokens is None:
            estimated_tokens = estimate_tokens(contents)

        with self.rate_limiter.limit(estimated_tokens):
            structured_llm = self.chat_model.with_structured_output(response_model)

            return self.retry_policy.execute(
                structured_llm.invoke,
                contents,
            )

    @staticmethod
    def parse_numbered_response(raw: str, expected_count: int) -> List[str]:
        """Parse '[N] ...' numbered LLM responses."""
        pattern = re.compile(r"\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)", re.DOTALL)

        matches = {
            int(m.group(1)): m.group(2).strip()
            for m in pattern.finditer(raw)
        }

        if not matches:
            log.warning("No numbered items parsed from LLM response")

        return [matches.get(i, "") for i in range(1, expected_count + 1)]