"""
Groq Service Provider Module.
Interacts with Groq's OpenAI-compatible Chat Completions API asynchronously.
Powers the ChatGPT-labelled demo panel for the Cloud Computing comparison project.

Technical Disclosure:
Demonstration responses in the ChatGPT panel are powered by a Groq-hosted open model
(such as Llama 3.3 70B Versatile), not the proprietary OpenAI ChatGPT API.
"""

import asyncio
import logging
import time
from typing import Any, Optional

import httpx
from app.config import settings

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ------------------------------------------------------------------------------
# Custom Service Exceptions
# ------------------------------------------------------------------------------

class GroqServiceError(Exception):
    """Base exception for Groq service operations."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GroqConfigError(GroqServiceError):
    """Raised when Groq API key is missing or misconfigured."""

    def __init__(
        self,
        message: str = "Groq API key is not configured on the server. Please set GROQ_API_KEY in your .env file.",
    ):
        super().__init__(message, status_code=500)


class GroqAuthError(GroqServiceError):
    """Raised when Groq API key is unauthorized or rejected."""

    def __init__(
        self,
        message: str = "Groq API authentication failed. Please verify your configured GROQ_API_KEY.",
    ):
        super().__init__(message, status_code=401)


class GroqRateLimitError(GroqServiceError):
    """Raised when Groq API rate limit or TPM/RPM quota is exceeded."""

    def __init__(
        self,
        message: str = "Groq API rate limit or quota exceeded. Please try again later.",
    ):
        super().__init__(message, status_code=429)


class GroqTimeoutError(GroqServiceError):
    """Raised when Groq API request times out."""

    def __init__(
        self,
        message: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        if message:
            detail = message
        elif timeout_seconds:
            detail = f"Request to Groq API timed out after {timeout_seconds:.1f}s. Please increase timeout or simplify prompt."
        else:
            detail = "Request to Groq API timed out. Please try again later."
        super().__init__(detail, status_code=504)
        self.timeout_seconds = timeout_seconds


class GroqProviderError(GroqServiceError):
    """Raised when Groq API returns an upstream error, model unavailable, or empty response."""

    def __init__(
        self,
        message: str = "Groq provider error occurred while generating content.",
        status_code: int = 502,
    ):
        super().__init__(message, status_code=status_code)


# ------------------------------------------------------------------------------
# Groq Service Provider
# ------------------------------------------------------------------------------

class GroqService:
    """
    Service class managing asynchronous communication with Groq's OpenAI-compatible API.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        self._api_key = api_key
        self._default_model = default_model
        self._timeout_seconds = timeout_seconds

    def _get_api_key(self) -> str:
        """
        Validates and retrieves the Groq API key from instance or global settings.
        Raises GroqConfigError if key is missing or unreplaced placeholder.
        """
        key = self._api_key or settings.groq_api_key
        if not key or not key.strip() or key.strip() == "your_groq_api_key_here":
            raise GroqConfigError(
                "Groq API key is not configured on the server. Please set GROQ_API_KEY in your .env file."
            )
        return key.strip()

    async def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> dict[str, Any]:
        """
        Asynchronously generates text completion from Groq API.
        Measures round-trip latency and captures token usage metadata.

        Args:
            prompt: User prompt text.
            model: Optional model identifier override (defaults to configured groq_model).
            timeout_seconds: Optional per-request timeout limit.
            http_client: Optional injected httpx.AsyncClient (used for testing).

        Returns:
            Dictionary matching ProviderResult schema.
        """
        if not prompt or not prompt.strip():
            raise GroqProviderError("Prompt cannot be empty or contain only whitespace.", status_code=400)

        target_model = model or self._default_model or settings.groq_model
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else (self._timeout_seconds or settings.request_timeout_seconds)
        )

        api_key = self._get_api_key()

        # Log sanitized operation metadata without revealing prompt content or credentials
        logger.info(
            "Calling Groq model '%s' (prompt length: %d chars, timeout: %ss)",
            target_model,
            len(prompt),
            effective_timeout,
        )

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "GenAI-Cloud-Gateway/1.0",
        }

        payload = {
            "model": target_model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
        }

        start_time = time.perf_counter()

        try:
            if http_client is not None:
                response = await http_client.post(
                    GROQ_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=effective_timeout,
                )
            else:
                async with httpx.AsyncClient(timeout=effective_timeout) as client:
                    response = await client.post(
                        GROQ_API_URL,
                        headers=headers,
                        json=payload,
                    )

        except (httpx.TimeoutException, asyncio.TimeoutError, TimeoutError) as exc:
            logger.warning("Groq API call timed out after %s seconds: %s", effective_timeout, type(exc).__name__)
            raise GroqTimeoutError(timeout_seconds=effective_timeout) from exc

        except httpx.RequestError as exc:
            logger.error("Network or connection error communicating with Groq: %s", type(exc).__name__)
            raise GroqProviderError(
                f"Network connection to Groq API failed: {type(exc).__name__}",
                status_code=502,
            ) from exc

        except Exception as exc:
            logger.error("Unexpected error in Groq service: %s", type(exc).__name__)
            raise GroqProviderError(
                "An unexpected error occurred while communicating with Groq.",
                status_code=502,
            ) from exc

        latency_ms = int(round((time.perf_counter() - start_time) * 1000))
        logger.info("Groq call completed in %d ms (status %d)", latency_ms, response.status_code)

        # Handle HTTP error responses
        if response.status_code != 200:
            error_data = {}
            try:
                error_data = response.json()
            except Exception:
                pass

            error_msg = ""
            if isinstance(error_data, dict) and "error" in error_data:
                err_obj = error_data["error"]
                if isinstance(err_obj, dict):
                    error_msg = err_obj.get("message", "")
                elif isinstance(err_obj, str):
                    error_msg = err_obj

            err_lower = (error_msg or response.text).lower()

            if response.status_code == 401 or "invalid api key" in err_lower or "unauthorized" in err_lower:
                raise GroqAuthError()
            elif response.status_code == 429 or "rate limit" in err_lower or "quota" in err_lower:
                raise GroqRateLimitError()
            elif response.status_code in (400, 404) and "model" in err_lower:
                raise GroqProviderError(
                    f"Groq model '{target_model}' is not available or unrecognized: {error_msg or 'Model error'}",
                    status_code=400,
                )
            elif response.status_code == 504:
                raise GroqTimeoutError(timeout_seconds=effective_timeout)
            else:
                detail = error_msg or f"Groq API returned HTTP {response.status_code}"
                raise GroqProviderError(f"Groq API error: {detail}", status_code=502)

        # Parse 200 OK response
        try:
            resp_json = response.json()
        except Exception as exc:
            raise GroqProviderError("Failed to parse JSON response from Groq.") from exc

        choices = resp_json.get("choices", [])
        if not choices:
            raise GroqProviderError("Groq returned an empty choice set.")

        first_choice = choices[0]
        message_obj = first_choice.get("message", {})
        generated_text = message_obj.get("content", "")

        if not generated_text or not generated_text.strip():
            raise GroqProviderError("Groq returned an empty response.")

        # Extract token counts without fabricating values if absent
        input_tokens = None
        output_tokens = None
        total_tokens = None

        usage = resp_json.get("usage")
        if isinstance(usage, dict):
            raw_prompt = usage.get("prompt_tokens")
            raw_completion = usage.get("completion_tokens")
            raw_total = usage.get("total_tokens")

            if isinstance(raw_prompt, int) and not isinstance(raw_prompt, bool):
                input_tokens = raw_prompt
            if isinstance(raw_completion, int) and not isinstance(raw_completion, bool):
                output_tokens = raw_completion
            if isinstance(raw_total, int) and not isinstance(raw_total, bool):
                total_tokens = raw_total
            elif input_tokens is not None or output_tokens is not None:
                total_tokens = (input_tokens or 0) + (output_tokens or 0)

        return {
            "provider": "ChatGPT (Groq)",
            "model": target_model,
            "response_text": generated_text,
            "response": generated_text,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "success": True,
            "error_message": None,
        }


# Global singleton instance
groq_service = GroqService()


def get_groq_service() -> GroqService:
    """Dependency provider for FastAPI dependency injection."""
    return groq_service
