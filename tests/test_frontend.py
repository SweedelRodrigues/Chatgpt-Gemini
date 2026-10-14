"""
Unit and integration tests for Frontend static assets and dashboard routes.
Verifies proper serving of HTML, CSS, JavaScript, content negotiation, and secret hygiene.
"""

import os
from fastapi.testclient import TestClient


def test_serve_dashboard_app_route(client: TestClient):
    """
    Test that GET /app serves the HTML5 dashboard with HTTP 200 and text/html.
    """
    response = client.get("/app")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "GenAI Cloud Gateway" in response.text
    assert "AI Playground" in response.text
    assert "Evaluation & Benchmarking" in response.text


def test_serve_dashboard_dashboard_route(client: TestClient):
    """
    Test that GET /dashboard also serves the dashboard.
    """
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


def test_static_assets_css_served(client: TestClient):
    """
    Test that GET /static/styles.css is accessible and served with text/css.
    """
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    assert "text/css" in response.headers.get("content-type", "")
    assert "--bg-app" in response.text


def test_static_assets_js_served(client: TestClient):
    """
    Test that GET /static/app.js is accessible and served as javascript.
    """
    response = client.get("/static/app.js")
    assert response.status_code == 200
    content_type = response.headers.get("content-type", "")
    assert "javascript" in content_type
    assert "BENCHMARK_PROMPTS" in response.text
    assert "/api/generate" in response.text


def test_root_content_negotiation_browser_html(client: TestClient):
    """
    Test that a browser requesting HTML from GET / receives the dashboard HTML.
    """
    response = client.get("/", headers={"accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")
    assert "GenAI Cloud Gateway" in response.text


def test_root_json_for_api_clients(client: TestClient):
    """
    Test that API clients requesting */* or JSON receive the JSON metadata including dashboard_url.
    """
    response = client.get("/", headers={"accept": "*/*"})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert data["dashboard_url"] == "/app"
    assert "docs_url" in data
    assert "health_url" in data


def test_static_files_contain_no_credentials():
    """
    Security audit test: verifies that no secrets or API keys are embedded
    inside static frontend files (HTML, CSS, JS).
    """
    static_dir = os.path.join(os.path.dirname(__file__), "..", "app", "static")
    files_to_check = ["index.html", "styles.css", "app.js"]

    for fname in files_to_check:
        fpath = os.path.join(static_dir, fname)
        assert os.path.exists(fpath), f"File {fname} must exist"
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()

        # Check that no Google API key patterns or env secrets appear
        assert "AIza" not in content, f"Possible secret detected in {fname}"
        assert "GEMINI_API_KEY=" not in content, f"Secret assignment detected in {fname}"


def test_html_contains_required_interactive_elements(client: TestClient):
    """
    Verifies that the HTML document contains all required interactive UI elements.
    """
    response = client.get("/app")
    html = response.text

    required_ids = [
        'id="prompt-input"',
        'id="char-counter"',
        'id="btn-generate"',
        'id="btn-clear-prompt"',
        'id="model-select"',
        'id="timeout-select"',
        'id="response-container"',
        'id="metrics-strip"',
        'id="metric-latency"',
        'id="metric-input-tokens"',
        'id="metric-output-tokens"',
        'id="metric-total-tokens"',
        'id="benchmark-table"',
        'id="btn-run-benchmarks"',
        'id="eval-timeout-select"',
        'id="btn-export-csv"',
        'id="kpi-median-latency"',
        'id="kpi-mean-latency"',
        'id="kpi-success-rate"',
    ]

    for req_id in required_ids:
        assert req_id in html, f"Missing required UI element: {req_id}"
