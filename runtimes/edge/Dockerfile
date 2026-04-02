# ============================================================
# Builder stage — install build tools and compile dependencies
# ============================================================
FROM ubuntu:24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive

# Build-time option: set to "false" to skip vision deps (ultralytics/YOLO)
ARG ENABLE_VISION=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a self-contained venv we can COPY to the runtime stage
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy local dependency sources
COPY common /deps/common
COPY packages/llamafarm-llama /deps/llamafarm-llama

# Copy edge runtime pyproject.toml for dependency resolution
COPY runtimes/edge/pyproject.toml ./

# Install CPU-only PyTorch first (saves ~2GB on AMD64 vs CUDA default).
# ARM64 PyPI wheels are already CPU-only, so this is a no-op there.
RUN uv pip install --no-cache \
    torch torchvision \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Install local deps as non-editable (no need to copy /deps/ to runtime)
RUN uv pip install --no-cache /deps/common /deps/llamafarm-llama

# Install edge runtime deps (vision conditional)
RUN if [ "$ENABLE_VISION" = "true" ]; then \
      uv pip install --no-cache --no-sources ".[vision]" && \
      uv pip install --no-cache pi-heif; \
    else \
      uv pip install --no-cache --no-sources "."; \
    fi

# Copy application code
COPY runtimes/edge/ .

# ============================================================
# Runtime stage — minimal image with only what's needed to run
# ============================================================
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime-only system libraries (no build-essential, cmake, git, python3-dev)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
WORKDIR /app
COPY --from=builder /app/ ./

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
    MODEL_UNLOAD_TIMEOUT=300 \
    GGUF_MODELS_DIR=/models \
    PRELOAD_MODELS="" \
    PRELOAD_N_CTX="" \
    YOLO_AUTOINSTALL=false

# Run the edge runtime
CMD ["python", "server.py"]
