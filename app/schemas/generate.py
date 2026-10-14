"""
Pydantic schemas for the Generation endpoint.
Enforces request validation and defines standardized gateway responses.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from app.config import settings


class GenerateRequest(BaseModel):
    """
    Inference request schema with input validation guards.
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_prompt_length,
        description="The prompt text to submit to Gemini.",
        json_schema_extra={"example": "Explain cloud computing in simple terms."}
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional model identifier override. Defaults to configured gateway model."
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=300.0,
        description="Optional per-request timeout limit in seconds (1.0 to 300.0). Defaults to server configuration."
    )

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        """
        Validate that the prompt is not empty or whitespace-only,
        and does not exceed maximum configured character length.
        """
        if not v or not v.strip():
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        if len(v) > settings.max_prompt_length:
            raise ValueError(f"Prompt cannot exceed {settings.max_prompt_length} characters.")
        return v


class TokenUsage(BaseModel):
    """
    Token usage metadata returned by the provider.
    Includes prompt input tokens, candidate output tokens, and total token count.
    """
    input_tokens: Optional[int] = Field(default=None, description="Input/prompt token count.")
    output_tokens: Optional[int] = Field(default=None, description="Output/candidate text token count.")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed.")


class GenerateResponse(BaseModel):
    """
    Standardized AI generation response schema.
    """
    provider: str = Field(default="Gemini", description="AI provider name.")
    model: str = Field(..., description="The Gemini model identifier used for generation.")
    response: str = Field(..., description="Generated text response from the model.")
    latency_ms: int = Field(..., description="Round-trip latency in milliseconds.")
    usage: Optional[TokenUsage] = Field(default=None, description="Token usage details if available, or null.")
