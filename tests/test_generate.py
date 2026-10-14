"""
Unit and integration tests for POST /api/generate and Gemini Service.
All external provider calls are mocked to prevent consuming quota or requiring live credentials.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from google.genai import errors, types

from app.config import settings
from app.services.gemini_service import (
    GeminiAuthError,
    GeminiConfigError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiService,
    GeminiTimeoutError,
    gemini_service,
)


@pytest.fixture
def mock_gemini_client():
    """
    Creates a mocked Gemini SDK client with an async generate_content method.
    """
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Cloud computing is the on-demand availability of computer system resources."

    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 14
    mock_usage.candidates_token_count = 52
    mock_usage.total_token_count = 66
    mock_response.usage_metadata = mock_usage

    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    return mock_client, mock_response


# ==============================================================================
# Endpoint Integration Tests (POST /api/generate)
# ==============================================================================

def test_generate_valid_prompt_success(client: TestClient, mock_gemini_client):
    """
    Test that a valid prompt returns a mocked Gemini response with latency and usage metrics.
    """
    mock_client, _ = mock_gemini_client

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        payload = {"prompt": "Explain cloud computing in simple terms."}
        response = client.post("/api/generate", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert data["provider"] == "Gemini"
        assert data["model"] == settings.default_model
        assert data["response"] == "Cloud computing is the on-demand availability of computer system resources."
        assert isinstance(data["latency_ms"], int)
        assert data["latency_ms"] >= 0
        assert data["usage"] == {
            "input_tokens": 14,
            "output_tokens": 52,
            "total_tokens": 66,
        }


def test_generate_with_model_override(client: TestClient, mock_gemini_client):
    """
    Test that a client can optionally specify a model identifier override.
    """
    mock_client, _ = mock_gemini_client

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        payload = {
            "prompt": "What is an API gateway?",
            "model": "gemini-2.5-pro",
        }
        response = client.post("/api/generate", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "gemini-2.5-pro"


def test_generate_usage_metadata_unavailable(client: TestClient):
    """
    Test that when usage metadata is unavailable from the provider,
    usage is represented as null rather than inventing values.
    """
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Serverless computing allows running code without managing servers."
    mock_response.usage_metadata = None
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        response = client.post(
            "/api/generate",
            json={"prompt": "Explain serverless architecture."},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["response"] == "Serverless computing allows running code without managing servers."
        assert data["usage"] is None


def test_generate_empty_prompt_rejected(client: TestClient):
    """
    Test that an empty prompt string is rejected with HTTP 422 validation error.
    """
    response = client.post("/api/generate", json={"prompt": ""})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_generate_whitespace_only_prompt_rejected(client: TestClient):
    """
    Test that whitespace-only prompts are rejected with HTTP 422 validation error.
    """
    response = client.post("/api/generate", json={"prompt": "   \n\t  "})
    assert response.status_code == 422
    data = response.json()
    assert any("whitespace" in err["msg"].lower() for err in data["detail"])


def test_generate_prompt_exceeding_max_length_rejected(client: TestClient):
    """
    Test that prompts exceeding 4000 characters are rejected with HTTP 422 validation error.
    """
    oversized_prompt = "A" * (settings.max_prompt_length + 1)
    response = client.post("/api/generate", json={"prompt": oversized_prompt})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_generate_missing_api_key(client: TestClient):
    """
    Test that missing API key returns a safe 500 configuration error
    without leaking internal stack traces.
    """
    with patch.object(settings, "gemini_api_key", None):
        # Reset cached client on service
        temp_service = GeminiService(api_key=None)
        with patch("app.api.generate.get_gemini_service", return_value=temp_service):
            response = client.post(
                "/api/generate",
                json={"prompt": "Hello Gemini!"},
            )

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "configured" in data["detail"].lower()
            # Ensure no credentials or tracebacks leaked
            assert "traceback" not in data["detail"].lower()


def test_generate_invalid_api_key_error(client: TestClient):
    """
    Test that invalid API key returns a safe 401 authentication error.
    """
    mock_client = MagicMock()
    mock_err = errors.APIError(401, {"error": {"message": "API key not valid", "status": "UNAUTHENTICATED"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_err)

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        response = client.post(
            "/api/generate",
            json={"prompt": "Explain virtualization."},
        )

        assert response.status_code == 401
        data = response.json()
        assert "authentication" in data["detail"].lower() or "api key" in data["detail"].lower()
        assert "traceback" not in str(data).lower()


def test_generate_rate_limit_error(client: TestClient):
    """
    Test that provider rate limit returns HTTP 429 and a helpful message.
    """
    mock_client = MagicMock()
    mock_err = errors.APIError(
        429,
        {"error": {"message": "Resource has been exhausted", "status": "RESOURCE_EXHAUSTED"}},
    )
    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_err)

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        response = client.post(
            "/api/generate",
            json={"prompt": "Explain edge computing."},
        )

        assert response.status_code == 429
        data = response.json()
        assert "rate limit" in data["detail"].lower() or "quota" in data["detail"].lower()


def test_generate_provider_timeout_error(client: TestClient):
    """
    Test that provider timeout returns HTTP 504 controlled timeout response.
    """
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=asyncio.TimeoutError("Request timed out"))

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        response = client.post(
            "/api/generate",
            json={"prompt": "Explain microservices."},
        )

        assert response.status_code == 504
        data = response.json()
        assert "timed out" in data["detail"].lower()


def test_generate_generic_provider_error(client: TestClient):
    """
    Test that unexpected upstream provider failure returns HTTP 502 without exposing stack traces.
    """
    mock_client = MagicMock()
    mock_err = errors.APIError(500, {"error": {"message": "Internal backend error", "status": "INTERNAL"}})
    mock_client.aio.models.generate_content = AsyncMock(side_effect=mock_err)

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        response = client.post(
            "/api/generate",
            json={"prompt": "Explain container orchestration."},
        )

        assert response.status_code == 502
        data = response.json()
        assert "provider error" in data["detail"].lower()
        assert "traceback" not in str(data).lower()


def test_generate_empty_provider_response(client: TestClient):
    """
    Test that empty response text from Gemini returns HTTP 502 gracefully.
    """
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = ""  # Empty text
    mock_response.usage_metadata = None
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch.object(gemini_service, "get_client", return_value=mock_client):
        response = client.post(
            "/api/generate",
            json={"prompt": "Trigger empty response."},
        )

        assert response.status_code == 502
        data = response.json()
        assert "empty response" in data["detail"].lower()


# ==============================================================================
# Service Provider Direct Unit Tests
# ==============================================================================

def test_service_generate_content_direct():
    """
    Directly test GeminiService method execution, token extraction, and latency timing.
    """
    mock_client = MagicMock()
    mock_response = MagicMock(spec=types.GenerateContentResponse)
    mock_response.text = "Direct service test response."
    mock_usage = MagicMock()
    mock_usage.prompt_token_count = 5
    mock_usage.candidates_token_count = 10
    mock_usage.total_token_count = 15
    mock_response.usage_metadata = mock_usage
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    service = GeminiService(api_key="mock_key_for_unit_test")
    result = asyncio.run(
        service.generate_content(
            prompt="Test prompt",
            client=mock_client,
        )
    )

    assert result["provider"] == "Gemini"
    assert result["response"] == "Direct service test response."
    assert result["latency_ms"] >= 0
    assert result["usage"]["total_tokens"] == 15


def test_service_raises_config_error_on_missing_key():
    """
    Verify GeminiService raises GeminiConfigError when no key is set.
    """
    service = GeminiService(api_key=None)
    with patch.object(settings, "gemini_api_key", None):
        with pytest.raises(GeminiConfigError):
            service.get_client()
