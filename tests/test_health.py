"""
Unit and integration tests for the Health and Root endpoints.
"""

from fastapi.testclient import TestClient


def test_health_endpoint_returns_200(client: TestClient):
    """
    Test that the /health endpoint responds with HTTP 200 and standard health payload.
    Crucial for cloud load balancers and orchestrator liveness checks.
    """
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "GenAI Cloud Gateway"
    assert data["version"] == "0.1.0"
    assert "environment" in data
    assert "api_key_configured" in data
    assert "timestamp" in data


def test_root_endpoint_returns_metadata(client: TestClient):
    """
    Test that the root endpoint responds with HTTP 200 and service metadata.
    """
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "online"
    assert "message" in data
    assert data["docs_url"] == "/docs"
    assert data["health_url"] == "/health"
