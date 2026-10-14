"""
Unit and integration tests for Markdown rendering and security in the AI Comparison Dashboard.
Verifies Markdown parsing, DOMPurify sanitization, XSS defense, safe link handling,
vendor asset serving, visual hierarchy styling, and plain-text copy preservation.
"""

import os
import re
import subprocess
from fastapi.testclient import TestClient


def test_markdown_vendor_assets_exist():
    """
    Verifies that all required frontend markdown and security vendor assets
    are physically present in app/static/vendor/.
    """
    vendor_dir = os.path.join(os.path.dirname(__file__), "..", "app", "static", "vendor")
    assert os.path.isdir(vendor_dir), f"Vendor directory '{vendor_dir}' must exist"

    required_assets = [
        "marked.min.js",
        "purify.min.js",
        "highlight.min.js",
        "highlight-github-dark.min.css",
    ]

    for asset_name in required_assets:
        asset_path = os.path.join(vendor_dir, asset_name)
        assert os.path.isfile(asset_path), f"Asset '{asset_name}' must exist in {vendor_dir}"
        assert os.path.getsize(asset_path) > 500, f"Asset '{asset_name}' must be non-empty"


def test_markdown_vendor_assets_served_via_fastapi(client: TestClient):
    """
    Verifies that FastAPI properly serves all markdown and security assets with HTTP 200.
    """
    routes_to_test = [
        ("/static/vendor/marked.min.js", "javascript"),
        ("/static/vendor/purify.min.js", "javascript"),
        ("/static/vendor/highlight.min.js", "javascript"),
        ("/static/vendor/highlight-github-dark.min.css", "text/css"),
    ]

    for route, expected_content_type in routes_to_test:
        response = client.get(route)
        assert response.status_code == 200, f"Route {route} failed with HTTP {response.status_code}"
        assert expected_content_type in response.headers.get("content-type", "").lower()
        assert len(response.text) > 100


def test_index_html_includes_markdown_libraries(client: TestClient):
    """
    Verifies that the dashboard HTML loads the markdown parser, sanitizer, and highlight styles.
    """
    response = client.get("/app")
    assert response.status_code == 200
    html = response.text

    assert "/static/vendor/highlight-github-dark.min.css" in html
    assert "/static/vendor/marked.min.js" in html
    assert "/static/vendor/purify.min.js" in html
    assert "/static/vendor/highlight.min.js" in html
    assert 'class="formatted-markdown markdown-body" id="chatgpt-text"' in html
    assert 'class="formatted-markdown markdown-body" id="gemini-text"' in html


def test_styles_css_contains_markdown_styling_rules(client: TestClient):
    """
    Verifies that styles.css defines styles for markdown-body, headings,
    blockquotes, lists, tables, code containers, and copy buttons.
    """
    response = client.get("/static/styles.css")
    assert response.status_code == 200
    css = response.text

    # Headings and typographic hierarchy
    assert ".markdown-body" in css
    assert ".formatted-markdown" in css
    assert ".markdown-body h1" in css
    assert ".markdown-body h2" in css
    assert ".markdown-body h3" in css

    # Lists, blockquotes, and tables
    assert ".markdown-body blockquote" in css
    assert ".markdown-body ul" in css
    assert ".markdown-body ol" in css
    assert ".table-wrapper" in css
    assert ".markdown-body table" in css

    # Code block container and copy code button
    assert ".code-block-container" in css
    assert ".code-block-header" in css
    assert ".btn-copy-code" in css
    assert ".code-lang-tag" in css


def test_app_js_implements_dompurify_sanitization():
    """
    Static code analysis of app.js to confirm strict security sanitization:
    - Calls DOMPurify.sanitize
    - Configures rel="noopener noreferrer" link security hook
    - Forbids script, iframe, and other dangerous HTML tags
    - Forbids event handlers like onerror, onclick, onload
    """
    app_js_path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "app.js")
    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js = f.read()

    assert "renderMarkdownSafely" in app_js
    assert "DOMPurify.sanitize" in app_js
    assert "marked.parse" in app_js
    assert "noopener noreferrer" in app_js
    assert 'target", "_blank"' in app_js

    # Forbidden security vectors in sanitizer config
    assert '"script"' in app_js
    assert '"iframe"' in app_js
    assert '"onerror"' in app_js
    assert '"onclick"' in app_js
    assert '"onload"' in app_js


def test_app_js_copies_plain_text_markdown_not_html():
    """
    Verifies that the Copy Response buttons copy the original plain-text Markdown response,
    not the generated HTML elements (Requirement 6).
    """
    app_js_path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "app.js")
    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js = f.read()

    assert "currentChatgptRawMarkdown" in app_js
    assert "currentGeminiRawMarkdown" in app_js
    assert "copyTextWithFeedback(btnCopyChatgpt, currentChatgptRawMarkdown" in app_js
    assert "copyTextWithFeedback(btnCopyGemini, currentGeminiRawMarkdown" in app_js


def test_marked_parser_in_node_runtime():
    """
    Executes the vendored marked.min.js in Node.js to test that headings,
    bold/italic, lists, blockquotes, tables, inline code, and code blocks
    are properly converted to semantic HTML tags without custom regex bugs.
    """
    node_script = r"""
    const fs = require('fs');
    const vm = require('vm');
    const markedCode = fs.readFileSync('app/static/vendor/marked.min.js', 'utf8');

    const sandbox = { window: {}, global: {}, console: console };
    sandbox.window = sandbox;
    vm.createContext(sandbox);
    vm.runInContext(markedCode, sandbox);

    const markdownInput = "# Heading 1\n## Heading 2\nThis is **bold** and *italic* text.\n\n> Cloud computing scalable.\n\n- Item 1\n- Item 2\n\n| Col1 | Col2 |\n| :--- | :--- |\n| Val1 | Val2 |\n\nHere is `inline code` and a block:\n\n```python\ndef test():\n    return True\n```\n";

    const htmlOutput = sandbox.marked.parse(markdownInput, { gfm: true, breaks: true });
    console.log(JSON.stringify({ html: htmlOutput }));
    """

    res = subprocess.run(
        ["node", "-e", node_script],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True,
        text=True,
        check=True,
    )

    import json
    data = json.loads(res.stdout)
    html = data["html"]

    # Verify semantic HTML elements
    assert "<h1>Heading 1</h1>" in html
    assert "<h2>Heading 2</h2>" in html
    assert "<strong>bold</strong>" in html
    assert "<em>italic</em>" in html
    assert "<blockquote>" in html
    assert "<ul>" in html and "<li>Item 1</li>" in html
    assert "<table>" in html and "Col1</th>" in html
    assert "<code>inline code</code>" in html
    assert "<pre><code" in html


def test_benchmark_data_does_not_contain_html():
    """
    Verifies that benchmark entries and CSV exports do not contain rendered HTML tags (Requirement 9).
    """
    app_js_path = os.path.join(os.path.dirname(__file__), "..", "app", "static", "app.js")
    with open(app_js_path, "r", encoding="utf-8") as f:
        app_js = f.read()

    # Benchmark records store data.chatgpt.response_text, not innerHTML
    assert "testItem.chatgptResponse = data.chatgpt.response_text;" in app_js
    assert "testItem.geminiResponse = data.gemini.response_text;" in app_js
    assert "csvContent" in app_js
    assert "csvEscape" in app_js
