"""
Unit and integration tests for Dual-Model Comparison (ChatGPT vs Gemini).
Tests:
- GroqService direct execution, auth errors, rate limits, timeouts, token mapping
- Concurrent execution via POST /api/compare
- Partial provider failures (Gemini fails, Groq succeeds and vice versa)
- Dual provider failure isolation
- Prompt validation guards (empty, whitespace, length)
- API key secrecy audit
- Batch benchmark summary calculation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
import pytest
from fastapi.testclient import TestClient
from google.genai import types

from app.config import settings
from app.schemas.compare import CompareRequest, CompareResponse, ProviderResult
from app.services.gemini_service import (
    GeminiAuthError,
    GeminiConfigError,
    GeminiRateLimitError,
    GeminiService,
    GeminiTimeoutError,
    gemini_service,
)
from app.services.groq_service import (
    GroqAuthError,
    GroqConfigError,
    GroqProviderError,
    GroqRateLimitError,
    GroqService,
    GroqTimeoutError,
    groq_service,
)


# ==============================================================================
# Groq Service Unit Tests
# ==============================================================================

@pytest.mark.anyio
async def test_groq_service_success():
    """
    Test that GroqService properly generates content and maps tokens using mocked HTTP client.
    """
    mock_http_client = AsyncMock()
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "model": "llama-3.3-70b-versatile",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Cloud computing provides scalable computing resources."},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 45,
            "total_tokens": 57
        }
    }
    mock_http_client.post.return_value = mock_response

    service = GroqService(api_key="gsk_test_key_12345")
    result = await service.generate_content(
        prompt="Explain cloud computing",
        model="llama-3.3-70b-versatile",
        http_client=mock_http_client
    )

    assert result["provider"] == "ChatGPT (Groq)"
    assert result["model"] == "llama-3.3-70b-versatile"
    assert "Cloud computing" in result["response_text"]
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 45
    assert result["total_tokens"] == 57
    assert result["success"] is True
    assert result["error_message"] is None


@pytest.mark.anyio
async def test_groq_service_missing_key():
    """
    Test that GroqService raises GroqConfigError if the key is missing or blank.
    """
    service = GroqService(api_key=None)
    with patch.object(settings, "groq_api_key", None):
        with pytest.raises(GroqConfigError) as exc_info:
            await service.generate_content(prompt="Test prompt")
        assert "Groq API key is not configured" in str(exc_info.value.message)


@pytest.mark.anyio
async def test_groq_service_auth_error():
    """
    Test that HTTP 401 raises GroqAuthError.
    """
    mock_http_client = AsyncMock()
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 401
    mock_response.text = "Invalid API Key"
    mock_response.json.return_value = {"error": {"message": "Invalid API Key"}}
    mock_http_client.post.return_value = mock_response

    service = GroqService(api_key="gsk_invalid_key")
    with pytest.raises(GroqAuthError):
        await service.generate_content(prompt="Test", http_client=mock_http_client)


@pytest.mark.anyio
async def test_groq_service_rate_limit_error():
    """
    Test that HTTP 429 raises GroqRateLimitError.
    """
    mock_http_client = AsyncMock()
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 429
    mock_response.text = "Rate limit reached"
    mock_response.json.return_value = {"error": {"message": "Rate limit exceeded"}}
    mock_http_client.post.return_value = mock_response

    service = GroqService(api_key="gsk_key")
    with pytest.raises(GroqRateLimitError):
        await service.generate_content(prompt="Test", http_client=mock_http_client)


@pytest.mark.anyio
async def test_groq_service_timeout():
    """
    Test that network timeout raises GroqTimeoutError with configured duration.
    """
    mock_http_client = AsyncMock()
    mock_http_client.post.side_effect = httpx.TimeoutException("Read timed out")

    service = GroqService(api_key="gsk_key")
    with pytest.raises(GroqTimeoutError) as exc_info:
        await service.generate_content(prompt="Test", timeout_seconds=15.0, http_client=mock_http_client)
    assert "timed out after 15.0s" in str(exc_info.value.message)


@pytest.mark.anyio
async def test_groq_service_handles_missing_token_metadata():
    """
    Test that if Groq returns no usage dictionary, tokens are None, not fabricated zeros.
    """
    mock_http_client = AsyncMock()
    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "chatcmpl-test",
        "model": "llama-3.3-70b-versatile",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Valid response"}}],
        "usage": None
    }
    mock_http_client.post.return_value = mock_response

    service = GroqService(api_key="gsk_key")
    result = await service.generate_content(prompt="Test", http_client=mock_http_client)
    assert result["input_tokens"] is None
    assert result["output_tokens"] is None
    assert result["total_tokens"] is None


# ==============================================================================
# Compare Endpoint Integration Tests (POST /api/compare)
# ==============================================================================

def test_compare_both_providers_success(client: TestClient):
    """
    Test concurrent execution when both ChatGPT and Gemini respond successfully.
    Verifies response alignment, metrics, differential calculation, and winner determination.
    """
    mock_gemini_res = {
        "provider": "Gemini",
        "model": "gemini-2.5-flash",
        "response": "Gemini: Cloud computing delivers scalable infrastructure.",
        "response_text": "Gemini: Cloud computing delivers scalable infrastructure.",
        "latency_ms": 350,
        "input_tokens": 10,
        "output_tokens": 30,
        "total_tokens": 40,
        "success": True,
        "error_message": None,
    }

    mock_groq_res = {
        "provider": "ChatGPT (Groq)",
        "model": "llama-3.3-70b-versatile",
        "response": "ChatGPT: Cloud computing enables on-demand computing.",
        "response_text": "ChatGPT: Cloud computing enables on-demand computing.",
        "latency_ms": 200,
        "input_tokens": 12,
        "output_tokens": 28,
        "total_tokens": 40,
        "success": True,
        "error_message": None,
    }

    with patch.object(gemini_service, "generate_content", new_callable=AsyncMock) as mock_gemini, \
         patch.object(groq_service, "generate_content", new_callable=AsyncMock) as mock_groq:
        mock_gemini.return_value = mock_gemini_res
        mock_groq.return_value = mock_groq_res

        payload = {"prompt": "Explain cloud computing in simple terms."}
        response = client.post("/api/compare", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["prompt"] == payload["prompt"]

        # Check ChatGPT result
        assert data["chatgpt"]["provider"] == "ChatGPT (Groq)"
        assert data["chatgpt"]["model"] == "llama-3.3-70b-versatile"
        assert "ChatGPT: Cloud computing" in data["chatgpt"]["response_text"]
        assert data["chatgpt"]["latency_ms"] == 200
        assert data["chatgpt"]["success"] is True

        # Check Gemini result
        assert data["gemini"]["provider"] == "Gemini"
        assert data["gemini"]["model"] == "gemini-2.5-flash"
        assert "Gemini: Cloud computing" in data["gemini"]["response_text"]
        assert data["gemini"]["latency_ms"] == 350
        assert data["gemini"]["success"] is True

        # Check differential calculations (ChatGPT 200ms vs Gemini 350ms -> ChatGPT faster by 150ms)
        assert data["faster_provider"] == "ChatGPT"
        assert data["latency_diff_ms"] == 150
        assert data["latency_diff_percent"] == 42.9 # round(150 / 350 * 100, 1)
        assert data["total_tokens_consumed"] == 80


def test_compare_partial_failure_gemini_fails_chatgpt_succeeds(client: TestClient):
    """
    Test that when Gemini fails (e.g. rate limit), ChatGPT's successful response is still returned.
    The endpoint returns HTTP 200 with isolated failure details.
    """
    mock_groq_res = {
        "provider": "ChatGPT (Groq)",
        "model": "llama-3.3-70b-versatile",
        "response_text": "ChatGPT answer.",
        "response": "ChatGPT answer.",
        "latency_ms": 180,
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "success": True,
        "error_message": None,
    }

    with patch.object(gemini_service, "generate_content", new_callable=AsyncMock) as mock_gemini, \
         patch.object(groq_service, "generate_content", new_callable=AsyncMock) as mock_groq:
        mock_gemini.side_effect = GeminiRateLimitError("Gemini rate limit exceeded.")
        mock_groq.return_value = mock_groq_res

        payload = {"prompt": "Explain containers."}
        response = client.post("/api/compare", json=payload)

        assert response.status_code == 200
        data = response.json()

        # ChatGPT succeeded
        assert data["chatgpt"]["success"] is True
        assert data["chatgpt"]["response_text"] == "ChatGPT answer."
        assert data["chatgpt"]["latency_ms"] == 180

        # Gemini isolated failure
        assert data["gemini"]["success"] is False
        assert "rate limit" in data["gemini"]["error_message"].lower()
        assert data["gemini"]["latency_ms"] is None
        assert data["gemini"]["response_text"] is None

        # Faster provider cannot be computed when one failed
        assert data["faster_provider"] is None
        assert data["latency_diff_ms"] is None


def test_compare_partial_failure_chatgpt_fails_gemini_succeeds(client: TestClient):
    """
    Test that when Groq/ChatGPT fails (e.g. auth error), Gemini's response is still returned.
    """
    mock_gemini_res = {
        "provider": "Gemini",
        "model": "gemini-2.5-flash",
        "response_text": "Gemini answer.",
        "response": "Gemini answer.",
        "latency_ms": 250,
        "input_tokens": 15,
        "output_tokens": 35,
        "total_tokens": 50,
        "success": True,
        "error_message": None,
    }

    with patch.object(gemini_service, "generate_content", new_callable=AsyncMock) as mock_gemini, \
         patch.object(groq_service, "generate_content", new_callable=AsyncMock) as mock_groq:
        mock_gemini.return_value = mock_gemini_res
        mock_groq.side_effect = GroqAuthError("Groq authentication failed.")

        payload = {"prompt": "Explain microservices."}
        response = client.post("/api/compare", json=payload)

        assert response.status_code == 200
        data = response.json()

        # ChatGPT failed
        assert data["chatgpt"]["success"] is False
        assert "authentication failed" in data["chatgpt"]["error_message"].lower()

        # Gemini succeeded
        assert data["gemini"]["success"] is True
        assert data["gemini"]["response_text"] == "Gemini answer."
        assert data["gemini"]["latency_ms"] == 250


def test_compare_both_providers_fail(client: TestClient):
    """
    Test that when both providers fail, the response records both errors gracefully.
    """
    with patch.object(gemini_service, "generate_content", new_callable=AsyncMock) as mock_gemini, \
         patch.object(groq_service, "generate_content", new_callable=AsyncMock) as mock_groq:
        mock_gemini.side_effect = GeminiTimeoutError(timeout_seconds=30.0)
        mock_groq.side_effect = GroqTimeoutError(timeout_seconds=30.0)

        payload = {"prompt": "Heavy computation."}
        response = client.post("/api/compare", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["chatgpt"]["success"] is False
        assert data["gemini"]["success"] is False
        assert "timed out" in data["chatgpt"]["error_message"].lower()
        assert "timed out" in data["gemini"]["error_message"].lower()
        assert data["faster_provider"] is None


def test_compare_validation_empty_prompt(client: TestClient):
    """
    Test that an empty prompt is rejected with HTTP 422.
    """
    response = client.post("/api/compare", json={"prompt": ""})
    assert response.status_code == 422


def test_compare_validation_whitespace_only(client: TestClient):
    """
    Test that a whitespace-only prompt is rejected with HTTP 422.
    """
    response = client.post("/api/compare", json={"prompt": "   \n\t  "})
    assert response.status_code == 422


def test_compare_validation_oversized_prompt(client: TestClient):
    """
    Test that a prompt exceeding 4000 characters is rejected with HTTP 422.
    """
    oversized = "A" * 4001
    response = client.post("/api/compare", json={"prompt": oversized})
    assert response.status_code == 422


def test_compare_api_key_secrecy(client: TestClient):
    """
    Security audit: ensure neither GEMINI_API_KEY nor GROQ_API_KEY appear anywhere in responses.
    """
    mock_gemini_res = {
        "provider": "Gemini",
        "model": "gemini-2.5-flash",
        "response_text": "Gemini text",
        "latency_ms": 200,
        "success": True,
    }
    mock_groq_res = {
        "provider": "ChatGPT (Groq)",
        "model": "llama-3.3-70b-versatile",
        "response_text": "Groq text",
        "latency_ms": 150,
        "success": True,
    }

    with patch.object(gemini_service, "generate_content", new_callable=AsyncMock) as mock_gemini, \
         patch.object(groq_service, "generate_content", new_callable=AsyncMock) as mock_groq:
        mock_gemini.return_value = mock_gemini_res
        mock_groq.return_value = mock_groq_res

        response = client.post("/api/compare", json={"prompt": "Hello"})
        content = response.text

        assert "AIza" not in content
        assert "gsk_" not in content
        assert "API_KEY" not in content


def test_dual_benchmark_summary_endpoint(client: TestClient):
    """
    Test POST /api/compare/benchmark/summary calculates stats for both ChatGPT and Gemini.
    """
    payload = [
        {
            "id": 1,
            "prompt": "Test 1",
            "chatgpt": {"success": True, "latency_ms": 100, "input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "gemini": {"success": True, "latency_ms": 200, "input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
            "faster_provider": "ChatGPT",
        },
        {
            "id": 2,
            "prompt": "Test 2",
            "chatgpt": {"success": True, "latency_ms": 300, "input_tokens": 15, "output_tokens": 25, "total_tokens": 40},
            "gemini": {"success": False, "error_message": "Timed out"},
            "faster_provider": None,
        },
    ]

    response = client.post("/api/compare/benchmark/summary", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_prompts"] == 2
    assert data["chatgpt"]["successful_requests"] == 2
    assert data["chatgpt"]["success_rate"] == 100.0
    assert data["chatgpt"]["mean_latency_ms"] == 200.0

    assert data["gemini"]["successful_requests"] == 1
    assert data["gemini"]["failed_requests"] == 1
    assert data["gemini"]["success_rate"] == 50.0
    assert data["gemini"]["mean_latency_ms"] == 200.0
    assert data["faster_provider_counts"]["ChatGPT"] == 1
