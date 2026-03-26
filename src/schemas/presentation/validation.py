"""Schemas for slide quality validation results."""

from typing import List

from pydantic import Field

from src.schemas.base import BaseSchema


class SlideValidationResult(BaseSchema):
    """Result produced by the QualityValidatorAgent for a single slide assignment.

    ``passed`` is the primary gate: the workflow loop exits as soon as this is
    True or the maximum number of revision attempts is exhausted.

    ``score`` is a normalised quality score in [0, 1].  Values below the
    workflow's ``MIN_QUALITY_SCORE`` threshold cause ``passed`` to be False.

    ``feedback`` contains actionable, human-readable guidance that is injected
    back into the SlideBuilderAgent prompt on the next revision attempt.

    ``issues`` is a structured list of discrete problems that can be used for
    logging and observability without parsing the free-text feedback.
    """

    passed: bool = Field(
        description="True if the slide assignment meets the minimum quality bar."
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Normalised quality score in [0, 1].",
    )
    feedback: str = Field(
        default="",
        description=(
            "Actionable guidance for the next revision attempt. "
            "Empty when passed=True."
        ),
    )
    issues: List[str] = Field(
        default_factory=list,
        description="Discrete list of problems identified during validation.",
    )
