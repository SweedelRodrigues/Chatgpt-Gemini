"""
Pytest configuration and common fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """
    Creates a FastAPI TestClient instance reusable across test sessions.
    Runs in-memory requests through the ASGI application stack without needing a running server socket.
    """
    with TestClient(app) as test_client:
        yield test_client
