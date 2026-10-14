"""
Pydantic schemas for Dual-Model AI Comparison (ChatGPT vs Gemini).
Enforces request validation and standardizes comparison responses.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from app.config import settings


class ProviderResult(BaseModel):
    """
    Standardized response payload for a single model inference.
    Used for both ChatGPT (Groq) and Gemini results in dual comparisons.
    """
    provider: str = Field(..., description="Provider name (e.g., 'ChatGPT (Groq)' or 'Gemini').")
    model: str = Field(..., description="Exact model identifier utilized.")
    response_text: Optional[str] = Field(default=None, description="Generated response content.")
    response: Optional[str] = Field(default=None, description="Compatibility alias for response_text.")
    latency_ms: Optional[int] = Field(default=None, description="Round-trip latency in milliseconds. Null if failed.")
    input_tokens: Optional[int] = Field(default=None, description="Prompt/input tokens consumed.")
    output_tokens: Optional[int] = Field(default=None, description="Completion/output tokens generated.")
    total_tokens: Optional[int] = Field(default=None, description="Total tokens consumed.")
    success: bool = Field(default=True, description="True if generation succeeded, False otherwise.")
    error_message: Optional[str] = Field(default=None, description="Error reason if request failed.")


class CompareRequest(BaseModel):
    """
    Dual-comparison inference request schema with input validation guards.
    """
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=settings.max_prompt_length,
        description="The prompt to evaluate across both ChatGPT (Groq) and Gemini.",
        json_schema_extra={"example": "Explain cloud computing in simple words."}
    )
    gemini_model: Optional[str] = Field(
        default=None,
        description="Optional Gemini model override (defaults to server setting)."
    )
    groq_model: Optional[str] = Field(
        default=None,
        description="Optional Groq model override (defaults to server setting)."
    )
    timeout_seconds: Optional[float] = Field(
        default=None,
        ge=1.0,
        le=300.0,
        description="Per-provider request timeout limit in seconds (1.0 to 300.0)."
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


class CompareResponse(BaseModel):
    """
    Standardized side-by-side comparison response schema.
    """
    prompt: str = Field(..., description="The prompt evaluated.")
    chatgpt: ProviderResult = Field(..., description="ChatGPT (Groq demo) response & metrics.")
    gemini: ProviderResult = Field(..., description="Gemini response & metrics.")
    faster_provider: Optional[str] = Field(default=None, description="Name of the faster provider for this run.")
    latency_diff_ms: Optional[int] = Field(default=None, description="Absolute latency difference in milliseconds.")
    latency_diff_percent: Optional[float] = Field(default=None, description="Latency difference percentage relative to slower provider.")
    total_tokens_consumed: Optional[int] = Field(default=None, description="Combined total token consumption if available.")


class ManualEvaluationEntry(BaseModel):
    """
    Schema for manual qualitative evaluation ratings (1-5 scale) across criteria.
    """
    id: Optional[str] = None
    prompt: str
    chatgpt_relevance: int = Field(..., ge=1, le=5)
    chatgpt_clarity: int = Field(..., ge=1, le=5)
    chatgpt_completeness: int = Field(..., ge=1, le=5)
    gemini_relevance: int = Field(..., ge=1, le=5)
    gemini_clarity: int = Field(..., ge=1, le=5)
    gemini_completeness: int = Field(..., ge=1, le=5)
    observation: Optional[str] = Field(default=None, max_length=1000)
    timestamp: Optional[str] = None
