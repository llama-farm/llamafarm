ARG BASE_IMAGE=python:3.12-slim-bookworm
FROM ${BASE_IMAGE} AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for layer caching
COPY pyproject.toml ./

# Copy local dependency sources (llamafarm-common, llamafarm-llama)
COPY ../../common /deps/common
COPY ../../packages/llamafarm-llama /deps/llamafarm-llama

# Install dependencies
RUN uv pip install --system -e . \
    --override "llamafarm-common @ file:///deps/common" \
    --override "llamafarm-llama @ file:///deps/llamafarm-llama"

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash edge && \
    chown -R edge:edge /app
USER edge

# Create data directory
RUN mkdir -p /home/edge/.llamafarm

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11540/health')" || exit 1

# Expose port
EXPOSE 11540

# Environment defaults
ENV LF_RUNTIME_PORT=11540 \
    LF_RUNTIME_HOST=0.0.0.0 \
    LOG_LEVEL=INFO \
    MODEL_UNLOAD_TIMEOUT=300

# Run the edge runtime
CMD ["python", "server.py"]
