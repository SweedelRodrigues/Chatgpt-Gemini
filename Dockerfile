# ==============================================================================
# GenAI Cloud Gateway — Production Container Image
# Course: Cloud Computing (IPCC22IS72)
# Multi-stage security profile: Minimal base, non-root user, zero baked secrets
# ==============================================================================

FROM python:3.11-slim as runtime

# Define non-root user and application directory
ARG USER_NAME=appuser
ARG USER_UID=1000
ARG USER_GID=1000
WORKDIR /app

# Set runtime Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

# Install required operating system security certificates
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir --upgrade pip

# Create unprivileged system user for non-root execution
RUN groupadd --gid ${USER_GID} ${USER_NAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} --create-home --shell /bin/bash ${USER_NAME}

# Copy and install dependencies in a dedicated cacheable layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code (excluding secrets via .dockerignore)
COPY app/ app/
COPY README.md .

# Enforce secure non-root file ownership
RUN chown -R ${USER_NAME}:${USER_NAME} /app

# Drop root privileges
USER ${USER_NAME}

# Expose default HTTP port
EXPOSE 8000

# Container liveness health check probe using Python standard library
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request, os; port = os.environ.get('PORT', '8000'); urllib.request.urlopen(f'http://127.0.0.1:{port}/health')" || exit 1

# Launch production ASGI server (respects dynamic PORT if provided by cloud platforms like Render)
CMD ["sh", "-c", "python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
