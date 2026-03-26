"""Configuration for LLM models."""

import os

from dataclasses import dataclass

from typing import Dict

from src.utils.singleton import SingletonMeta

@dataclass(frozen=True)
class ModelRateLimits:
    """Configuration for a single model."""
    rpm: int                    # Requests per minute
    rpd: int                    # Requests per day
    tpm: int                    # Tokens per minute

@dataclass(frozen=True)
class Model:
    """Model configuration."""
    name: str
    rate_limits: ModelRateLimits

# Model names — importable by agents and constants modules
GEMMA_3_27B_IT = Model(
    name="gemma-3-27b-it", 
    rate_limits=ModelRateLimits(rpm=30, tpm=15_000, rpd=15_000)
)
GEMINI_2_5_FLASH = Model(
    name="gemini-2.5-flash", 
    rate_limits=ModelRateLimits(rpm=5, tpm=250_000, rpd=20)
)
GEMINI_3_1_FLASH_LITE = Model(
    name="gemini-3.1-flash-lite-preview", 
    rate_limits=ModelRateLimits(rpm=15, tpm=250_000, rpd=500)
)

AVAILABLE_MODELS = [
    GEMMA_3_27B_IT,
    GEMINI_2_5_FLASH,
    GEMINI_3_1_FLASH_LITE,
]

AGENT_MODELS: Dict[str, str] = {
    "content-generator": os.getenv("CONTENT_GENERATOR_AGENT_MODEL_NAME"),
    "presentation-structure": os.getenv("PRESENTATION_AGENT_MODEL_NAME"),
    "image-describer": os.getenv("IMAGE_DESCRIBER_AGENT_MODEL_NAME"),
    "content-summarizer": os.getenv("CONTENT_SUMMARIZER_AGENT_MODEL_NAME"),
    "slide-builder": os.getenv("SLIDE_BUILDER_AGENT_MODEL_NAME"),
}

class AgentMeta(SingletonMeta):
    """Metaclass that registers agents with unique names and resolves models."""

    registry: dict[str, type] = {}
    available_models: list["Model"] = []

    def __new__(mcs, clsname, bases, attrs, **kwargs):
        agent_name = kwargs.pop("name", None)
        if agent_name:
            if agent_name in mcs.registry:
                raise ValueError(f"Duplicate agent name detected: {agent_name}")
            attrs["AGENT_KEY"] = agent_name
            mcs.registry[agent_name] = None  # placeholder for class

        cls = super().__new__(mcs, clsname, bases, attrs)

        if agent_name:
            mcs.registry[agent_name] = cls
            cls.MODEL = mcs.get_model_for_agent(agent_name)

        return cls

    @classmethod
    def get_model_for_agent(mcs, agent_name: str) -> "Model":
        """Resolve a Model object for a given agent name."""
        model_name = AGENT_MODELS.get(agent_name)
        if not model_name:
            raise RuntimeError(f"No model mapped for agent '{agent_name}'")

        for model in mcs.available_models:
            if model.name == model_name:
                return model
        raise ValueError(f"Model {model_name} not found in available_models.")


# Register available models with the metaclass
AgentMeta.available_models = AVAILABLE_MODELS
