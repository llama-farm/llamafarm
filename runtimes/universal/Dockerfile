# Use Python 3.12 slim image with multi-arch support
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Build arg to control PyTorch variant (empty = CUDA, cpu = CPU-only)
# Usage: docker build --build-arg PYTORCH_VARIANT=cpu ...
ARG PYTORCH_VARIANT=

# Build arg to control llama.cpp backend
# Options: cpu, vulkan, cuda12, cuda11 (Linux), metal (macOS only)
# Usage: docker build --build-arg LLAMA_BACKEND=vulkan ...
# If not specified, auto-detects based on platform (cpu for Linux x86_64)
ARG LLAMA_BACKEND=

# Disable/limit bytecode compilation to avoid CI stalls
ENV UV_COMPILE_BYTECODE=0
ENV UV_COMPILE_BYTECODE_WORKERS=1
ENV UV_COMPILE_BYTECODE_TIMEOUT=15

# Set UV cache directory to a location that will be writable by any user
ENV UV_CACHE_DIR=/tmp/uv-cache

# Set UV to use a writable location for virtual environments
ENV UV_PROJECT_ENVIRONMENT=/tmp/uv-venv

# Set PyTorch index URL based on build arg
ENV UV_EXTRA_INDEX_URL=${PYTORCH_VARIANT:+https://download.pytorch.org/whl/${PYTORCH_VARIANT}}

ENV LF_DATA_DIR=/var/lib/llamafarm

# Set llama.cpp cache directory for pre-downloaded binaries
ENV LLAMAFARM_CACHE_DIR=/var/lib/llamafarm/llama-cache

# Pass LLAMA_BACKEND build arg to runtime environment (if specified)
ENV LLAMAFARM_BACKEND=${LLAMA_BACKEND}

RUN apt update && apt install -y curl unzip && rm -rf /var/lib/apt/lists/*

# Create UV cache and virtual environment directories with proper permissions for any user
RUN mkdir -p /tmp/uv-cache && chmod 777 /tmp/uv-cache \
    && mkdir -p /tmp/uv-venv && chmod 777 /tmp/uv-venv \
    && mkdir -p /var/lib/llamafarm && chmod 777 /var/lib/llamafarm \
    && mkdir -p /var/lib/llamafarm/llama-cache && chmod 777 /var/lib/llamafarm/llama-cache

# Set working directory
WORKDIR /app

# Copy the required packages from the monorepo (lightweight server only)
# Note: This Dockerfile should be run from the repo root: docker build -f runtimes/universal/Dockerfile .
COPY config/ ./config/
COPY common/ ./common/
COPY packages/llamafarm-llama/ ./packages/llamafarm-llama/
COPY rag/schema.yaml ./rag/schema.yaml
COPY ruff.toml ./ruff.toml

# Install config dependencies including dev dependencies for datamodel generation
RUN cd config && uv sync --locked --all-extras --verbose
RUN cd common && uv sync --locked --all-extras --verbose

# Generate config schema and datamodel types
RUN cd config && uv run python generate_types.py
RUN rm -rf rag server


# Copy universal runtime dependency files and README
COPY runtimes/universal/pyproject.toml runtimes/universal/uv.lock runtimes/universal/README.md ./runtimes/universal/

# Copy universal runtime source code (needed before installing package)
COPY runtimes/universal/ ./runtimes/universal/

# Install server dependencies and install package in editable mode
RUN cd runtimes/universal && uv sync --locked --verbose \
    && uv cache clean \
    && rm -rf /tmp/uv-cache/*

# Pre-download llama.cpp binaries during build for faster container startup
# This downloads the appropriate binary based on platform and LLAMA_BACKEND setting
# The binary is cached in LLAMAFARM_CACHE_DIR so it's available at runtime
RUN cd runtimes/universal && uv run python scripts/download_llama_binary.py

# Expose port (default is 11540, can be overridden with LF_RUNTIME_PORT env var)
EXPOSE 11540

# Change to /tmp so any runtime files are created in writable location
WORKDIR /tmp

ENV HOST=0.0.0.0

# Run the application from the universal runtime directory
CMD ["uv", "run", "--project", "/app/runtimes/universal", "python", "/app/runtimes/universal/server.py"]
