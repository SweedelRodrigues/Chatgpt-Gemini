"""
Dual-Model AI Comparison Router.
Provides the POST /api/compare endpoint to evaluate prompts concurrently
across ChatGPT (Groq-hosted model) and Google Gemini.
"""

import asyncio
import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.config import settings
from app.schemas.compare import (
    CompareRequest,
    CompareResponse,
    ManualEvaluationEntry,
    ProviderResult,
)
from app.services.evaluation import BenchmarkSummary, calculate_benchmark_metrics
from app.services.gemini_service import (
    GeminiService,
    GeminiServiceError,
    get_gemini_service,
)
from app.services.groq_service import (
    GroqService,
    GroqServiceError,
    get_groq_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Comparison"])


@router.post(
    "/compare",
    response_model=CompareResponse,
    status_code=status.HTTP_200_OK,
    summary="Compare ChatGPT and Gemini responses side by side",
    description=(
        "Submits a single user prompt concurrently to both ChatGPT (Groq-hosted demo model) "
        "and Google Gemini. Measures individual round-trip latencies, extracts token usage, "
        "and isolates provider failures so one provider failing does not block the other."
    ),
    responses={
        200: {"description": "Comparison generated for both models with metrics and differential summary."},
        422: {"description": "Prompt validation error (e.g. empty, whitespace, too long)."},
    },
)
async def compare_models(
    request: CompareRequest,
    gemini_svc: GeminiService = Depends(get_gemini_service),
    groq_svc: GroqService = Depends(get_groq_service),
) -> CompareResponse:
    """
    Execute concurrent inference requests to Groq (ChatGPT demo) and Google Gemini.
    """
    target_gemini_model = request.gemini_model or settings.default_model
    target_groq_model = request.groq_model or settings.groq_model

    logger.info(
        "Starting dual model comparison (prompt length: %d chars). Models: Gemini='%s', Groq='%s'",
        len(request.prompt),
        target_gemini_model,
        target_groq_model,
    )

    # Launch both provider requests concurrently
    gemini_coroutine = gemini_svc.generate_content(
        prompt=request.prompt,
        model=target_gemini_model,
        timeout_seconds=request.timeout_seconds,
    )
    groq_coroutine = groq_svc.generate_content(
        prompt=request.prompt,
        model=target_groq_model,
        timeout_seconds=request.timeout_seconds,
    )

    gemini_res, groq_res = await asyncio.gather(
        gemini_coroutine,
        groq_coroutine,
        return_exceptions=True,
    )

    # Process Gemini result
    if isinstance(gemini_res, Exception):
        logger.warning("Gemini comparison request failed: %s", type(gemini_res).__name__)
        err_msg = getattr(gemini_res, "message", str(gemini_res))
        gemini_result = ProviderResult(
            provider="Gemini",
            model=target_gemini_model,
            response_text=None,
            response=None,
            latency_ms=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            success=False,
            error_message=err_msg,
        )
    else:
        gemini_result = ProviderResult(
            provider="Gemini",
            model=gemini_res["model"],
            response_text=gemini_res.get("response_text") or gemini_res.get("response"),
            response=gemini_res.get("response"),
            latency_ms=gemini_res.get("latency_ms"),
            input_tokens=gemini_res.get("input_tokens"),
            output_tokens=gemini_res.get("output_tokens"),
            total_tokens=gemini_res.get("total_tokens"),
            success=True,
            error_message=None,
        )

    # Process Groq (ChatGPT) result
    if isinstance(groq_res, Exception):
        logger.warning("Groq comparison request failed: %s", type(groq_res).__name__)
        err_msg = getattr(groq_res, "message", str(groq_res))
        chatgpt_result = ProviderResult(
            provider="ChatGPT (Groq)",
            model=target_groq_model,
            response_text=None,
            response=None,
            latency_ms=None,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            success=False,
            error_message=err_msg,
        )
    else:
        chatgpt_result = ProviderResult(
            provider="ChatGPT (Groq)",
            model=groq_res["model"],
            response_text=groq_res.get("response_text") or groq_res.get("response"),
            response=groq_res.get("response"),
            latency_ms=groq_res.get("latency_ms"),
            input_tokens=groq_res.get("input_tokens"),
            output_tokens=groq_res.get("output_tokens"),
            total_tokens=groq_res.get("total_tokens"),
            success=True,
            error_message=None,
        )

    # Compute differential latency & token metrics
    faster_provider = None
    latency_diff_ms = None
    latency_diff_percent = None

    if (
        chatgpt_result.success
        and gemini_result.success
        and chatgpt_result.latency_ms is not None
        and gemini_result.latency_ms is not None
    ):
        diff = gemini_result.latency_ms - chatgpt_result.latency_ms
        latency_diff_ms = abs(diff)

        if diff > 0:
            faster_provider = "ChatGPT"
        elif diff < 0:
            faster_provider = "Gemini"
        else:
            faster_provider = "Tie"

        slower_latency = max(chatgpt_result.latency_ms, gemini_result.latency_ms)
        if slower_latency > 0:
            latency_diff_percent = round((latency_diff_ms / slower_latency) * 100.0, 1)

    # Sum total tokens if both are known
    total_tokens_consumed = None
    if chatgpt_result.total_tokens is not None or gemini_result.total_tokens is not None:
        total_tokens_consumed = (chatgpt_result.total_tokens or 0) + (gemini_result.total_tokens or 0)

    return CompareResponse(
        prompt=request.prompt,
        chatgpt=chatgpt_result,
        gemini=gemini_result,
        faster_provider=faster_provider,
        latency_diff_ms=latency_diff_ms,
        latency_diff_percent=latency_diff_percent,
        total_tokens_consumed=total_tokens_consumed,
    )


class DualBenchmarkSummary(BaseModel):
    """Aggregated comparison summary for batch benchmark runs."""
    total_prompts: int
    chatgpt: BenchmarkSummary
    gemini: BenchmarkSummary
    faster_provider_counts: dict[str, int]


@router.post(
    "/compare/benchmark/summary",
    response_model=DualBenchmarkSummary,
    summary="Compute dual benchmark summary metrics",
    description="Calculates reproducible aggregate statistics for both ChatGPT and Gemini from batch benchmark runs.",
)
async def get_dual_benchmark_summary(results: list[dict[str, Any]]) -> DualBenchmarkSummary:
    """
    Computes comparative statistics across a series of comparison benchmark test runs.
    """
    total = len(results)
    chatgpt_runs = []
    gemini_runs = []
    faster_counts = {"ChatGPT": 0, "Gemini": 0, "Tie": 0}

    for item in results:
        # Extract ChatGPT item
        c_item = item.get("chatgpt") or {}
        chatgpt_runs.append({
            "status": "success" if c_item.get("success") else "failed",
            "latency_ms": c_item.get("latency_ms"),
            "input_tokens": c_item.get("input_tokens"),
            "output_tokens": c_item.get("output_tokens"),
            "total_tokens": c_item.get("total_tokens"),
            "error": c_item.get("error_message") or "Failed",
            "id": item.get("id"),
            "prompt": item.get("prompt"),
        })

        # Extract Gemini item
        g_item = item.get("gemini") or {}
        gemini_runs.append({
            "status": "success" if g_item.get("success") else "failed",
            "latency_ms": g_item.get("latency_ms"),
            "input_tokens": g_item.get("input_tokens"),
            "output_tokens": g_item.get("output_tokens"),
            "total_tokens": g_item.get("total_tokens"),
            "error": g_item.get("error_message") or "Failed",
            "id": item.get("id"),
            "prompt": item.get("prompt"),
        })

        faster = item.get("faster_provider")
        if faster in faster_counts:
            faster_counts[faster] += 1

    chatgpt_summary = calculate_benchmark_metrics(chatgpt_runs)
    gemini_summary = calculate_benchmark_metrics(gemini_runs)

    return DualBenchmarkSummary(
        total_prompts=total,
        chatgpt=chatgpt_summary,
        gemini=gemini_summary,
        faster_provider_counts=faster_counts,
    )
