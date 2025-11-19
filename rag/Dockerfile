# Use Python 3.12 slim image with multi-arch support
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Disable/limit bytecode compilation to avoid CI stalls
ENV UV_COMPILE_BYTECODE=0
ENV UV_COMPILE_BYTECODE_WORKERS=1
ENV UV_COMPILE_BYTECODE_TIMEOUT=15

# Set UV cache directory to a location that will be writable by any user
ENV UV_CACHE_DIR=/tmp/uv-cache

# Set UV to use a writable location for virtual environments
ENV UV_PROJECT_ENVIRONMENT=/tmp/uv-venv

# Optimize UV for faster builds
ENV UV_NO_SYNC_IGNORE_FILES=1
ENV UV_HTTP_TIMEOUT=300

# Create a writable control directory for Celery filesystem transport
RUN mkdir -p /tmp/control && chmod 777 /tmp/control

ENV LF_DATA_DIR=/var/lib/llamafarm

RUN apt update && apt install -y curl && rm -rf /var/lib/apt/lists/*

# Create UV cache and virtual environment directories with proper permissions for any user
RUN mkdir -p /tmp/uv-cache && chmod 777 /tmp/uv-cache \
    && mkdir -p /tmp/uv-venv && chmod 777 /tmp/uv-venv \
    && mkdir -p /var/lib/llamafarm && chmod 777 /var/lib/llamafarm

# Set working directory
WORKDIR /app

# Copy dependency files first for better Docker layer caching
COPY config/ ./config/
COPY common/ ./common/
COPY observability/ ./observability/
COPY rag/pyproject.toml rag/uv.lock rag/README.md ./rag/
COPY ruff.toml ./ruff.toml

# Install dependencies in separate layers (cached unless lock files change)
RUN cd config && uv sync --locked --all-extras
RUN cd rag && uv sync --locked

# Now copy the rest of the source code (changes won't invalidate dependency cache)
COPY config/ ./config/
COPY common/ ./common/
COPY observability/ ./observability/
COPY rag/ ./rag/

# Install each package with its own dependencies
RUN cd config && uv sync --locked --all-extras --verbose
RUN cd common && uv sync --locked --all-extras --verbose
RUN cd rag && uv sync --locked  --verbose

# Generate config schema and datamodel types
RUN cd config && uv run python generate_types.py

# Fix ownership and permissions of UV directories after they've been populated
RUN chmod -R 777 /tmp/uv-cache /tmp/uv-venv

# Change to /tmp so Celery control files are created in writable location
WORKDIR /tmp

# Run the RAG Celery worker from the rag directory
CMD ["uv", "run", "--project", "/app/rag", "python", "/app/rag/main.py"]
