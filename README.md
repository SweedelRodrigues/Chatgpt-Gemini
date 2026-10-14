# ChatGPT vs Gemini — Cloud-Based Generative AI Comparison Gateway

**Course:** Cloud Computing (IPCC22IS72)  
**Project:** Cloud-Based Generative AI: A Case Study of ChatGPT and Gemini (AAT Project)  
**Status:** Feature Complete — Dual-Model Architecture & Premium Comparison Dashboard  

---

## 📌 Technical Disclosure & Architecture Overview

> [!IMPORTANT]
> **Technical Provider Disclosure:**  
> In this academic case study and demonstration, the **ChatGPT-labelled demo panel** is powered by a high-speed, Groq-hosted open foundation model (defaulting to Meta Llama 3.3 70B Versatile or Llama 3.1 8B Instant) via Groq's OpenAI-compatible Chat Completions API. It does **not** call the official closed-source OpenAI API. The actual underlying model name is clearly reported in the UI response badges and response metadata (`model: "llama-3.3-70b-versatile"`). The **Gemini panel** is powered by Google's foundation model (`gemini-2.5-flash`) via the official Google Gen AI Python SDK.

The **GenAI Cloud Gateway** demonstrates how to securely orchestrate, compare, and observe multiple cloud AI providers through an **API Gateway pattern**:
1. **Zero Client-Side Credentials:** API keys for both Google Gemini and Groq are stored exclusively in the cloud backend environment variables and are never transmitted to or readable by the web browser.
2. **Concurrent Dual-Inference:** Evaluates a single prompt against both models simultaneously using non-blocking asynchronous coroutines (`asyncio.gather`), measuring server-side round-trip latencies independently with microsecond precision.
3. **Provider Failure Isolation:** A failure, rate limit, or timeout on one provider does **not** fail the entire comparison; the other provider's response is safely returned and displayed alongside an isolated failure notification.
4. **Normalized Token Accounting:** Captures and reports prompt tokens, completion tokens, and total tokens from each provider's usage metadata without fabricating values.

---

## 🏗️ Architectural Pattern

```
+-----------------------------------------------------------------------------------+
|                           Client Browser (Web UI)                                 |
|  - "ChatGPT vs Gemini" Side-by-Side Responsive Dashboard                          |
|  - Prompt Composer with Live Char Counter & Preset Chips                          |
|  - Independent Dual Response Cards with Code Highlighting & One-Click Copy        |
|  - Performance & Latency Differential Summary (% and ms difference)               |
|  - Manual Qualitative Evaluation Matrix (Relevance, Clarity, Completeness 1-5)    |
|  - 10-Scenario Automated Benchmarking Suite with SVG/CSS Latency Chart & CSV Export|
+-----------------------------------------+-----------------------------------------+
                                          | HTTP / REST (JSON)
                                          v
+-----------------------------------------------------------------------------------+
|                        GenAI Cloud Gateway (FastAPI / Uvicorn)                    |
|                                                                                   |
|  - Static Asset Serving & Browser Content Negotiation (/app, /dashboard, /)       |
|  - Request Validation & Prompt Length Guards (1 to 4000 chars)                    |
|  - Configurable Request Timeouts (1 to 300 seconds)                               |
|  - Cloud Liveness & Readiness Health Probe (/health)                              |
|  - Dual Comparison Endpoint (/api/compare) via asyncio.gather                     |
|  - Single-Model Inference Endpoint (/api/generate)                                |
|  - Batch Summary Analytics Calculators (/api/compare/benchmark/summary)           |
+--------------------+---------------------------------------------+----------------+
                     |                                             |
                     | Google Gen AI SDK                           | Groq OpenAI-Compatible
                     | Secure HTTPS                                | REST API (Async httpx)
                     v                                             v
+------------------------------------+             +--------------------------------+
|      Google Cloud AI Platform      |             |         Groq Cloud AI          |
|    (Gemini 2.5 Flash Foundation)   |             | (Llama 3.3 70B / 3.1 8B Model) |
+------------------------------------+             +--------------------------------+
```

---

## 🗂️ Project Structure

```
genai-cloud-gateway/
├── Dockerfile                # Production multi-stage container image (non-root appuser, dynamic $PORT)
├── .dockerignore             # Strict build exclusions (.env, secrets, caches, tests)
├── docker-compose.yml        # Local container orchestration
├── render.yaml               # Render Blueprint specification for Free Tier web service
├── .env.example              # Template for environment configuration
├── .gitignore                # Protects secrets, virtual environments, and caches
├── requirements.txt          # Production and testing dependencies
├── README.md                 # Complete project documentation and case study guide
├── benchmark_results.csv     # Real evaluation run data (reproducible test suite)
├── app/
│   ├── __init__.py           # Application package marker
│   ├── main.py               # FastAPI entrypoint, middleware, static mounting, & routing
│   ├── config.py             # Centralized settings using pydantic-settings
│   ├── api/
│   │   ├── __init__.py       # API routes package marker
│   │   ├── health.py         # Cloud liveness & readiness health check endpoint (/health)
│   │   ├── compare.py        # Dual-model comparison & batch summary endpoints (/api/compare)
│   │   └── generate.py       # Single-model inference & evaluation endpoints (/api/generate)
│   ├── schemas/
│   │   ├── __init__.py       # Schemas package marker
│   │   ├── compare.py        # Pydantic schemas for CompareRequest, CompareResponse, ProviderResult
│   │   └── generate.py       # Pydantic schemas for single generation and token usage
│   ├── services/
│   │   ├── __init__.py       # Services package marker
│   │   ├── gemini_service.py # Google Gen AI SDK async provider with error mapping
│   │   ├── groq_service.py   # Groq OpenAI-compatible async provider for ChatGPT demo
│   │   └── evaluation.py     # Truthful evaluation calculations (mean/median latency, tokens)
│   └── static/               # Frontend developer dashboard assets
│       ├── index.html        # HTML5 dual-model comparison dashboard & benchmark UI
│       ├── styles.css        # Premium dark navy/charcoal stylesheet (#0B1020, #121A2B)
│       └── app.js            # Vanilla JS dual comparison, benchmark runner & chart controller
└── tests/
    ├── __init__.py           # Test suite package marker
    ├── conftest.py           # Pytest fixtures and TestClient initialization
    ├── test_compare.py       # 15 tests: Groq provider, dual compare, partial failures, secrecy
    ├── test_health.py        # Automated test cases for health and root endpoints
    ├── test_generate.py      # Mocked unit and integration tests for Gemini generation
    ├── test_frontend.py      # Unit and security tests for static assets & dashboard
    └── test_evaluation.py    # Metric calculations, timeouts, metadata & Docker security audit
```

---

## 🔑 Environment Configuration

Create a `.env` file in the project root:
```powershell
Copy-Item .env.example .env
```

Configure your credentials inside `.env`:
```ini
# Gemini API Credentials (from Google AI Studio: https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# Groq API Credentials (from Groq Console: https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Application Configuration
APP_NAME="GenAI Cloud Gateway"
APP_ENV=development
HOST=0.0.0.0
PORT=8000

# Gateway Default Parameters
DEFAULT_MODEL=gemini-2.5-flash
MAX_PROMPT_LENGTH=4000
REQUEST_TIMEOUT_SECONDS=30.0
```

---

## 🚀 Running Locally

### 1. Start the Development Server
Run Uvicorn from the project directory:
```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access the Application
Open your browser:
- **ChatGPT vs Gemini Comparison Dashboard:** [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app) (or [http://127.0.0.1:8000/](http://127.0.0.1:8000/))
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Probe:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

---

## 💻 Dashboard Features & Walkthrough

### 1. Dual AI Playground
1. Enter any query or click a sample prompt chip:
   - *Explain cloud computing in simple words.*
   - *Write a Python program to check whether a number is prime.*
   - *Explain the difference between AI and machine learning.*
   - *Give five practical applications of generative AI.*
2. Press `Ctrl + Enter` or click **⚡ Compare Responses**.
3. Observe live loading animations on both cards simultaneously.
4. Review the outputs:
   - **Left Panel (ChatGPT):** Shows output, latency in `ms`, input/output/total token counts, and Groq model disclosure.
   - **Right Panel (Gemini):** Shows output, latency in `ms`, input/output/total token counts, and Google SDK integration badge.
5. **Comparison Summary Banner:** Displays which provider responded faster, the latency gap in milliseconds and percentage, and combined token consumption.
6. **Manual Evaluation Section:** Rate both responses on Relevance, Clarity, and Completeness (1–5 scale) and add qualitative observations.

### 2. Automated Benchmarking Suite
1. Switch to the **Evaluation & Benchmarking** tab.
2. Select all or any combination of the 10 curated cloud test scenarios.
3. Click **▶ Run Selected Tests**.
4. The gateway executes tests sequentially with throttling pauses to avoid triggering free-tier rate limits.
5. Live KPI cards calculate:
   - Total Prompts, Completed, Failed, Success Rate
   - **ChatGPT Median Latency vs Gemini Median Latency**
   - Mean Latency & Total Tokens Consumed
6. **Interactive Latency Chart:** Visually compares side-by-side bar lengths for each prompt.
7. Click **📥 Export to CSV** to download a comprehensive evaluation report.

---

## ☁️ Render Free Web Service Deployment Guide

The gateway is 100% compatible with **Render's Free Web Service plan** ($0/month, 512 MB RAM, 0.1 CPU).

### Blueprint Deployment (Recommended)
1. Push your repository to GitHub.
2. In the Render Dashboard, click **New + $\rightarrow$ Blueprint**.
3. Select your repository. Render automatically reads [render.yaml](file:///C:/Users/USER/.gemini/antigravity-ide/scratch/genai-cloud-gateway/render.yaml).
4. When prompted by the Render dashboard, enter your secret API keys:
   - `GEMINI_API_KEY`: *[Your Google Gemini API Key]*
   - `GROQ_API_KEY`: *[Your Groq API Key]*
5. Click **Apply**. Render will install dependencies and start the gateway.

### Manual Setup Parameters
| Field | Value | Notes |
| :--- | :--- | :--- |
| **Runtime** | `Python 3` | Native Python runtime (Oregon or Ohio region) |
| **Plan** | `Free` | $0/mo (Free web service) |
| **Build Command** | `pip install -r requirements.txt` | Installs runtime dependencies |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` | Binds to 0.0.0.0 on Render's dynamic port |
| **Health Check Path** | `/health` | Cloud liveness check under Advanced Settings |

---

## 🧪 Automated Testing Suite

The complete pytest suite contains **53 automated tests** covering all providers, schemas, routes, and edge cases.

To execute the full test suite:
```powershell
python -m pytest tests/ -v
```

### Test Suite Results:
```powershell
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\Users\USER\.gemini\antigravity-ide\scratch\genai-cloud-gateway
collected 53 items

tests/test_compare.py (15 tests) ............... PASSED
tests/test_evaluation.py (14 tests) ............. PASSED
tests/test_frontend.py (8 tests) ........       PASSED
tests/test_generate.py (14 tests) .............. PASSED
tests/test_health.py (2 tests) ..               PASSED

======================== 53 passed, 1 warning in 0.50s ========================
```
*All external API calls to Google and Groq are fully mocked during test execution to ensure zero quota consumption, zero cost, and sub-second test execution.*

---

## 🔒 Security Best Practices Implemented

1. **Zero Secret Leakage:** Neither API key (`GEMINI_API_KEY` or `GROQ_API_KEY`) is ever logged, returned in HTTP responses, or embedded in frontend assets.
2. **Git & Docker Exclusions:** `.gitignore` and `.dockerignore` exclude `.env`, `*.key`, local caches, virtual environments, and test logs.
3. **Safe Content Rendering:** All generated AI outputs are bound using `textContent`, mitigating Cross-Site Scripting (XSS).
4. **Input Validation Guards:** Prompts are validated against empty strings, whitespace-only content, and maximum length limits (4000 characters).
5. **Configurable Timeouts:** Both providers enforce request timeouts (defaulting to 30s) to prevent hanging threads and resource exhaustion.
