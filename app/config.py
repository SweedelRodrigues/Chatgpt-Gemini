"""
Configuration module for GenAI Cloud Gateway.
Implements the 12-Factor App methodology by externalizing configuration into environment variables.
"""

from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings model with type safety and default fallbacks.
    Loads values from active environment variables or a local .env file.
    """
    app_name: str = "GenAI Cloud Gateway"
    app_version: str = "0.1.0"
    app_env: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000

    # Gemini API Credentials
    gemini_api_key: Optional[str] = None
    default_model: str = "gemini-2.5-flash-lite"

    # Groq API Credentials (powers the ChatGPT-labelled demo panel)
    groq_api_key: Optional[str] = None
    groq_model: str = "openai/gpt-oss-120b"

    # Gateway rate limiting and validation guards
    max_prompt_length: int = 4000
    request_timeout_seconds: float = 30.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
