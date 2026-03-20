"""Common base schema configuration used across Pydantic models."""

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    """Base configuration for Pydantic schemas.

    All application schemas should inherit from BaseSchema to enable consistent
    configuration such as camelCase alias generation and validation behaviour.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        from_attributes=True,
    )

class UserRequest(BaseSchema):
    """User request for a presentation structure."""
    user_request: str