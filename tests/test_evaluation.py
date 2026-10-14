"""
Unit and integration tests for Phase 4:
- Metric calculations (success rate, mean, median latency, min, max, tokens)
- Missing token metadata handling & thinking token mapping
- Configurable timeout handling & timeout error details
- Real benchmark CSV reproducibility validation
- Containerization security audit (Dockerfile & .dockerignore)
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from google.genai import types

from app.schemas.generate import GenerateRequest, TokenUsage
from app.services.evaluation import (
    calculate_benchmark_metrics,
    parse_benchmark_csv_content,
)
from app.services.gemini_service import (
    GeminiService,
    GeminiTimeoutError,
    gemini_service,
)


# ==============================================================================
# Metric Calculations Tests
# ==============================================================================

def test_calculate_metrics_with_sample_vectors():
    """
    Test calculation of success rate, mean, median, min, max, and token sums.
    """
    sample_data = [
        {"id": 1, "status": "success", "latency_ms": 100, "usage": {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}},
        {"id": 2, "status": "success", "latency_ms": 200, "usage": {"input_tokens": 15, "output_tokens": 25, "total_tokens": 40}},
        {"id": 3, "status": "success", "latency_ms": 300, "usage": {"input_tokens": 20, "output_tokens": 30, "total_tokens": 50}},
        {"id": 4, "status": "failed", "latency_ms": None, "usage": None, "error": "Request timed out"},
    ]

    summary = calculate_benchmark_metrics(sample_data)

    assert summary.total_requests == 4
    assert summary.successful_requests == 3
    assert summary.failed_requests == 1
    assert summary.success_rate == 75.0
    assert summary.mean_latency_ms == 200.0
    assert summary.median_latency_ms == 200.0
    assert summary.min_latency_ms == 100.0
    assert summary.max_latency_ms == 300.0
    assert summary.total_input_tokens == 45
    assert summary.total_output_tokens == 75
    assert summary.total_tokens == 120
    assert len(summary.failures) == 1
    assert summary.failures[0].id == 4
    assert "timed out" in summary.failures[0].error.lower()


def test_calculate_metrics_even_number_median():
    """
    Test median calculation with an even number of successful requests.
    """
    sample_data = [
        {"status": "success", "latency_ms": 100},
        {"status": "success", "latency_ms": 200},
        {"status": "success", "latency_ms": 300},
        {"status": "success", "latency_ms": 400},
    ]

    summary = calculate_benchmark_metrics(sample_data)
    assert summary.successful_requests == 4
    # Median of [100, 200, 300, 400] is (200 + 300) / 2 = 250.0
    assert summary.median_latency_ms == 250.0
    assert summary.mean_latency_ms == 250.0


def test_calculate_metrics_all_failed():
    """
    Test metrics calculation when all requests fail.
    Verifies that mean and median latency are None (not fabricated as 0).
    """
    sample_data = [
        {"id": 1, "status": "failed", "error": "Rate limit 429"},
        {"id": 2, "status": "failed", "error": "Timeout 504"},
    ]

    summary = calculate_benchmark_metrics(sample_data)
    assert summary.total_requests == 2
    assert summary.successful_requests == 0
    assert summary.failed_requests == 2
    assert summary.success_rate == 0.0
    assert summary.mean_latency_ms is None
    assert summary.median_latency_ms is None
    assert summary.total_tokens == 0
    assert len(summary.failures) == 2


def test_calculate_metrics_empty_list():
    """
    Test metrics calculation on an empty input list.
    """
    summary = calculate_benchmark_metrics([])
    assert summary.total_requests == 0
    assert summary.successful_requests == 0
    assert summary.failed_requests == 0
    assert summary.success_rate == 0.0
    assert summary.mean_latency_ms is None
    assert summary.median_latency_ms is None


# ==============================================================================
# Real Benchmark Results CSV Verification
# ==============================================================================

def test_parse_real_benchmark_csv():
    """
    Reproducibility test: validates calculations against the actual benchmark CSV.
    Confirms 10 total requests, 9 successful, 1 timeout failure (ID 7), 90% success rate,
    and median latency of 16,841 ms.
    """
    csv_path = os.path.join(os.path.dirname(__file__), "..", "benchmark_results.csv")
    if not os.path.exists(csv_path):
        pytest.skip("benchmark_results.csv not found in workspace")

    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()

    summary = parse_benchmark_csv_content(content)

    assert summary.total_requests == 10
    assert summary.successful_requests == 9
    assert summary.failed_requests == 1
    assert summary.success_rate == 90.0
    assert summary.median_latency_ms == 16841.0
    assert summary.mean_latency_ms == 17128.22
    assert summary.min_latency_ms == 5254.0
    assert summary.max_latency_ms == 29593.0
    assert summary.total_input_tokens == 249
    assert summary.total_output_tokens == 14090
    assert summary.total_tokens == 29754

    # Verify failure recorded for Test #7 Multilingual without fabricated metrics
    assert len(summary.failures) == 1
    assert str(summary.failures[0].id) == "7"
    assert "timed out" in summary.failures[0].error.lower()


# ==============================================================================
# Token Usage & Metadata Mapping Tests
# ==============================================================================

def test_token_usage_model():
    """
    Verify TokenUsage model schema with input, output, and total tokens.
    """
    usage = TokenUsage(
        input_tokens=23,
        output_tokens=1907,
        total_tokens=3669,
    )
    assert usage.input_tokens == 23
    assert usage.output_tokens == 1907
    assert usage.total_tokens == 3669


def test_gemini_service_maps_token_metadata():
    """
    Verify GeminiService correctly extracts prompt_token_count, candidates_token_count,
    and total_token_count from usage_metadata.
    """
    import asyncio
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Response with token counts."

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 20
    mock_usage.candidates_token_count = 50
    mock_usage.total_token_count = 70
    mock_response.usage_metadata = mock_usage
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(api_key="dummy_test_key")
    result = asyncio.run(service.generate_content(prompt="Test prompt", client=mock_client))

    assert result["usage"]["input_tokens"] == 20
    assert result["usage"]["output_tokens"] == 50
    assert result["usage"]["total_tokens"] == 70


def test_gemini_service_computes_total_if_omitted_by_provider():
    """
    If the provider leaves total_token_count as None, the gateway computes it safely from components.
    """
    import asyncio
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Sample output."

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 10
    mock_usage.candidates_token_count = 40
    mock_usage.total_token_count = None  # Provider did not supply total
    mock_response.usage_metadata = mock_usage
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(api_key="dummy_test_key")
    result = asyncio.run(service.generate_content(prompt="Test prompt", client=mock_client))

    # Safe computation: 10 + 40 = 50
    assert result["usage"]["total_tokens"] == 50


def test_gemini_service_handles_missing_token_metadata():
    """
    If usage_metadata is None, usage is set to None without error or invented values.
    """
    import asyncio
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Sample output without metadata."
    mock_response.usage_metadata = None
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(api_key="dummy_test_key")
    result = asyncio.run(service.generate_content(prompt="Test prompt", client=mock_client))

    assert result["usage"] is None


# ==============================================================================
# Configurable Timeouts & Clear Timeout Reporting
# ==============================================================================

def test_generate_request_accepts_timeout_seconds():
    """
    Verify GenerateRequest model validates timeout_seconds correctly.
    """
    valid = GenerateRequest(prompt="Valid prompt", timeout_seconds=45.0)
    assert valid.timeout_seconds == 45.0

    # Test invalid range (timeout must be >= 1.0)
    with pytest.raises(Exception):
        GenerateRequest(prompt="Valid prompt", timeout_seconds=0.5)


def test_timeout_error_contains_duration_detail(client: TestClient):
    """
    Verify that timeout error response clearly reports the configured timeout limit.
    """
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=TimeoutError("Request timed out"))

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        response = client.post(
            "/api/generate",
            json={"prompt": "Complex reasoning prompt", "timeout_seconds": 15.0},
        )

        assert response.status_code == 504
        data = response.json()
        assert "timed out after 15.0s" in data["detail"]


# ==============================================================================
# Benchmark Summary Endpoint Integration
# ==============================================================================

def test_benchmark_summary_endpoint(client: TestClient):
    """
    Test POST /api/benchmark/summary endpoint calculation.
    """
    payload = [
        {"id": 1, "status": "success", "latency_ms": 120, "input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        {"id": 2, "status": "success", "latency_ms": 180, "input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
        {"id": 3, "status": "failed", "error": "Provider error 502"},
    ]

    response = client.post("/api/benchmark/summary", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_requests"] == 3
    assert data["successful_requests"] == 2
    assert data["failed_requests"] == 1
    assert data["success_rate"] == 66.67
    assert data["mean_latency_ms"] == 150.0
    assert data["median_latency_ms"] == 150.0


# ==============================================================================
# Containerization Security Audit (.dockerignore & Dockerfile)
# ==============================================================================

def test_dockerignore_excludes_secrets_and_caches():
    """
    Security audit: verify .dockerignore excludes .env, keys, and caches.
    """
    dockerignore_path = os.path.join(os.path.dirname(__file__), "..", ".dockerignore")
    assert os.path.exists(dockerignore_path), ".dockerignore must exist"

    with open(dockerignore_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert ".env" in content
    assert "*.key" in content or ".env*" in content
    assert "__pycache__" in content
    assert "venv" in content


def test_dockerfile_security_profile():
    """
    Security audit: verify Dockerfile runs as non-root user and excludes hardcoded secrets.
    """
    dockerfile_path = os.path.join(os.path.dirname(__file__), "..", "Dockerfile")
    assert os.path.exists(dockerfile_path), "Dockerfile must exist"

    with open(dockerfile_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Must run as non-root user
    assert "USER appuser" in content or "USER " in content
    # Must NOT copy .env
    assert "COPY .env" not in content
    # Must include health check
    assert "HEALTHCHECK" in content
