"""
GenAI Generation API Router.
Provides the POST /api/generate endpoint for AI text completions.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.schemas.generate import GenerateRequest, GenerateResponse
from app.services.gemini_service import (
    GeminiService,
    GeminiServiceError,
    get_gemini_service,
)

router = APIRouter(prefix="/api", tags=["Generation"])


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate AI text completion",
    description=(
        "Accepts a validated user prompt and generates an AI response using the configured "
        "Gemini model. Measures round-trip latency and captures token usage metadata."
    ),
    responses={
        200: {"description": "Successful generation response with metrics."},
        400: {"description": "Invalid request parameters."},
        401: {"description": "Gemini API key authentication error."},
        422: {"description": "Prompt validation error (e.g. empty, whitespace, too long)."},
        429: {"description": "Provider rate limit or quota exceeded."},
        500: {"description": "Server configuration error (e.g. missing API key)."},
        502: {"description": "Upstream Gemini provider error."},
        504: {"description": "Gateway timeout communicating with Gemini."},
    },
)
async def generate_text(
    request: GenerateRequest,
    service: GeminiService = Depends(get_gemini_service),
) -> GenerateResponse:
    """
    Generate text using Gemini API with timing and token usage metrics.
    """
    try:
        result = await service.generate_content(
            prompt=request.prompt,
            model=request.model,
            timeout_seconds=request.timeout_seconds,
        )
        return GenerateResponse(**result)
    except GeminiServiceError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message)


from typing import Any
from app.services.evaluation import BenchmarkSummary, calculate_benchmark_metrics


@router.post(
    "/benchmark/summary",
    response_model=BenchmarkSummary,
    summary="Compute benchmark summary metrics",
    description="Calculates reproducible aggregate metrics across benchmark test items without fabricating values.",
)
async def get_benchmark_summary(results: list[dict[str, Any]]) -> BenchmarkSummary:
    """
    Computes summary metrics (success count, failure count, success rate, mean, and median latency).
    """
    return calculate_benchmark_metrics(results)
