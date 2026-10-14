"""
Health check endpoints for GenAI Cloud Gateway.
Provides liveness and readiness monitoring crucial for cloud deployments and load balancers.
"""

from datetime import datetime, timezone
from fastapi import APIRouter
from pydantic import BaseModel
from app.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    """Schema for health status response."""
    status: str
    app_name: str
    version: str
    environment: str
    api_key_configured: bool
    gemini_configured: bool
    groq_configured: bool
    default_model: str
    groq_model: str
    timestamp: str


@router.get("/health", response_model=HealthResponse, summary="Service Health Check")
async def health_check() -> HealthResponse:
    """
    Check the health status of the API Gateway.
    Used by cloud orchestrators, Docker health checks, and monitoring services.
    Validates gateway readiness without invoking external APIs to preserve quota.
    """
    return HealthResponse(
        status="healthy",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.app_env,
        api_key_configured=bool(settings.gemini_api_key),
        gemini_configured=bool(settings.gemini_api_key),
        groq_configured=bool(settings.groq_api_key),
        default_model=settings.default_model,
        groq_model=settings.groq_model,
        timestamp=datetime.now(timezone.utc).isoformat()
    )
