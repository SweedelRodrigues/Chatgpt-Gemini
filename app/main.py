"""
GenAI Cloud Gateway - Main Application Entrypoint.
College Mini-Project: Cloud Computing (IPCC22IS72)
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from app.config import settings
from app.api.health import router as health_router
from app.api.generate import router as generate_router
from app.api.compare import router as compare_router
from app.services.gemini_service import GeminiServiceError
from app.services.groq_service import GroqServiceError

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(
    title="ChatGPT vs Gemini — GenAI Cloud Gateway",
    version=settings.app_version,
    description="A Cloud-Based Generative AI Comparison Gateway (ChatGPT vs Gemini). College Mini-Project: Cloud Computing (IPCC22IS72).",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for local development and future frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(GeminiServiceError)
async def gemini_service_error_handler(request: Request, exc: GeminiServiceError):
    """
    Global exception handler for Gemini provider errors, preventing stack trace exposure.
    """
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


@app.exception_handler(GroqServiceError)
async def groq_service_error_handler(request: Request, exc: GroqServiceError):
    """
    Global exception handler for Groq provider errors, preventing stack trace exposure.
    """
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# Mount static assets directory
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Register route modules
app.include_router(health_router)
app.include_router(generate_router)
app.include_router(compare_router)


@app.get("/app", tags=["Frontend"], include_in_schema=False)
@app.get("/dashboard", tags=["Frontend"], include_in_schema=False)
async def serve_dashboard():
    """
    Serves the developer dashboard and evaluation suite.
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/", tags=["Root"])
async def root(request: Request):
    """
    Gateway root endpoint providing service overview and documentation links.
    Returns the web dashboard to interactive browsers, or JSON metadata to API clients.
    """
    accept = request.headers.get("accept", "")
    if "text/html" in accept and not accept.startswith("*/*"):
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    return {
        "message": "Welcome to the GenAI Cloud Gateway API (ChatGPT vs Gemini Comparison)",
        "service": settings.app_name,
        "version": settings.app_version,
        "docs_url": "/docs",
        "health_url": "/health",
        "generate_url": "/api/generate",
        "compare_url": "/api/compare",
        "dashboard_url": "/app",
        "status": "online"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=(settings.app_env == "development")
    )
