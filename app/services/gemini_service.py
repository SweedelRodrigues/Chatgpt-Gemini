"""
Gemini Service Provider Module.
Interacts with the official Google Gen AI Python SDK asynchronously to generate content.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from google import genai
from google.genai import errors
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Custom Service Exceptions
# ------------------------------------------------------------------------------

class GeminiServiceError(Exception):
    """Base exception for Gemini service operations."""

    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class GeminiConfigError(GeminiServiceError):
    """Raised when Gemini API key or credentials are missing/misconfigured."""

    def __init__(
        self,
        message: str = "Gemini API key is not configured on the server. Please set GEMINI_API_KEY.",
    ):
        super().__init__(message, status_code=500)


class GeminiAuthError(GeminiServiceError):
    """Raised when Gemini API key is invalid or unauthorized."""

    def __init__(
        self,
        message: str = "Gemini API authentication failed. Please verify your configured API key.",
    ):
        super().__init__(message, status_code=401)


class GeminiRateLimitError(GeminiServiceError):
    """Raised when Gemini API rate limit or quota is exceeded."""

    def __init__(
        self,
        message: str = "Gemini API rate limit or quota exceeded. Please try again later.",
    ):
        super().__init__(message, status_code=429)


class GeminiTimeoutError(GeminiServiceError):
    """Raised when Gemini API request times out."""

    def __init__(
        self,
        message: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        if message:
            detail = message
        elif timeout_seconds:
            detail = f"Request to Gemini API timed out after {timeout_seconds:.1f}s. Please increase timeout or simplify prompt."
        else:
            detail = "Request to Gemini API timed out. Please try again later."
        super().__init__(detail, status_code=504)
        self.timeout_seconds = timeout_seconds


class GeminiProviderError(GeminiServiceError):
    """Raised when Gemini API returns an upstream error or empty response."""

    def __init__(
        self,
        message: str = "Gemini provider error occurred while generating content.",
    ):
        super().__init__(message, status_code=502)


# ------------------------------------------------------------------------------
# Gemini Service Provider
# ------------------------------------------------------------------------------

class GeminiService:
    """
    Service class managing client lifecycle and asynchronous content generation.
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
        self._client: Optional[genai.Client] = None
        self._cached_key: Optional[str] = None

    def get_client(self) -> genai.Client:
        """
        Lazily initialize or retrieve the Gemini Client.
        Validates API key configuration before client instantiation.
        """
        key = self._api_key or settings.gemini_api_key
        # Check if missing, empty, or unreplaced placeholder
        if not key or not key.strip() or key.strip() == "your_gemini_api_key_here":
            raise GeminiConfigError(
                "Gemini API key is not configured on the server. Please set GEMINI_API_KEY in your .env file."
            )

        # Cache client instance while key remains identical
        if self._client is None or self._cached_key != key:
            self._client = genai.Client(api_key=key)
            self._cached_key = key

        return self._client

    async def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        client: Optional[genai.Client] = None,
    ) -> dict[str, Any]:
        """
        Calls the Gemini API asynchronously with the provided prompt.

        Args:
            prompt: Validated prompt string.
            model: Optional model identifier override.
            timeout_seconds: Optional timeout in seconds.
            client: Optional injected Gemini client (useful for unit tests).

        Returns:
            Dict containing provider, model, response, latency_ms, and usage.
        """
        target_model = model or self._default_model or settings.default_model
        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else (self._timeout_seconds or settings.request_timeout_seconds)
        )

        active_client = client or self.get_client()

        # Log sanitized operation metadata without revealing prompt or credentials
        logger.info(
            "Calling Gemini model '%s' (prompt length: %d chars)",
            target_model,
            len(prompt),
        )

        # Measure actual elapsed time with time.perf_counter()
        start_time = time.perf_counter()

        try:
            call_coroutine = active_client.aio.models.generate_content(
                model=target_model,
                contents=prompt,
            )

            if effective_timeout and effective_timeout > 0:
                response = await asyncio.wait_for(call_coroutine, timeout=effective_timeout)
            else:
                response = await call_coroutine

        except GeminiServiceError:
            # Re-raise known service exceptions directly
            raise

        except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException) as exc:
            logger.warning(
                "Gemini API call timed out after %s seconds: %s",
                effective_timeout,
                type(exc).__name__,
            )
            raise GeminiTimeoutError(timeout_seconds=effective_timeout) from exc

        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            err_str = str(exc).lower()
            logger.warning(
                "Gemini APIError encountered: code=%s, type=%s",
                code,
                type(exc).__name__,
            )

            if code == 429 or "resource_exhausted" in err_str or "quota" in err_str:
                raise GeminiRateLimitError() from exc
            elif (
                code in (401, 403)
                or "api_key_invalid" in err_str
                or "invalid api key" in err_str
                or "unauthenticated" in err_str
            ):
                raise GeminiAuthError() from exc
            elif code == 504 or "deadline_exceeded" in err_str or "timed out" in err_str:
                raise GeminiTimeoutError(timeout_seconds=effective_timeout) from exc
            else:
                raise GeminiProviderError() from exc

        except Exception as exc:
            logger.error("Unexpected error in Gemini service: %s", type(exc).__name__)
            raise GeminiProviderError(
                "An unexpected error occurred while communicating with Gemini."
            ) from exc

        # Measure elapsed time in milliseconds
        latency_ms = int(round((time.perf_counter() - start_time) * 1000))
        logger.info("Gemini call completed in %d ms", latency_ms)

        # Extract generated response text
        generated_text = getattr(response, "text", None)
        if generated_text is None or not generated_text.strip():
            logger.warning("Gemini returned empty or blocked text response")
            raise GeminiProviderError(
                "Gemini returned an empty response. The content may have been filtered or blocked."
            )

        # Extract input, output, and total token counts from usage metadata
        usage_data = None
        usage_metadata = getattr(response, "usage_metadata", None)
        if usage_metadata is not None:
            raw_input = getattr(usage_metadata, "prompt_token_count", None)
            input_tokens = raw_input if isinstance(raw_input, int) and not isinstance(raw_input, bool) else None

            raw_output = getattr(usage_metadata, "candidates_token_count", None)
            output_tokens = raw_output if isinstance(raw_output, int) and not isinstance(raw_output, bool) else None

            raw_total = getattr(usage_metadata, "total_token_count", None)
            total_tokens = raw_total if isinstance(raw_total, int) and not isinstance(raw_total, bool) else None

            # If total_tokens is missing from provider/mock but input or output exist, compute safely
            if total_tokens is None and (input_tokens is not None or output_tokens is not None):
                total_tokens = (input_tokens or 0) + (output_tokens or 0)

            if (
                input_tokens is not None
                or output_tokens is not None
                or total_tokens is not None
            ):
                usage_data = {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }

        return {
            "provider": "Gemini",
            "model": target_model,
            "response": generated_text,
            "response_text": generated_text,
            "latency_ms": latency_ms,
            "usage": usage_data,
            "input_tokens": usage_data.get("input_tokens") if usage_data else None,
            "output_tokens": usage_data.get("output_tokens") if usage_data else None,
            "total_tokens": usage_data.get("total_tokens") if usage_data else None,
            "success": True,
            "error_message": None,
        }


# Global singleton instance
gemini_service = GeminiService()


def get_gemini_service() -> GeminiService:
    """Dependency provider for FastAPI dependency injection."""
    return gemini_service


async def generate_content(
    prompt: str,
    model: Optional[str] = None,
    timeout_seconds: Optional[float] = None,
    client: Optional[genai.Client] = None,
) -> dict[str, Any]:
    """
    Convenience function that invokes the global GeminiService instance.
    """
    return await gemini_service.generate_content(
        prompt=prompt,
        model=model,
        timeout_seconds=timeout_seconds,
        client=client,
    )
