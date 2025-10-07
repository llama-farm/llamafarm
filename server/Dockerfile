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

# Copy the required packages from the monorepo (lightweight server only)
# Note: This Dockerfile should be run from the repo root: docker build -f server/Dockerfile .
COPY config/ ./config/
COPY rag/schema.yaml ./rag/schema.yaml

# Install config dependencies only (server no longer needs RAG/models)
RUN ls -la config
RUN cd config && uv sync --locked  --verbose

# Compile the config schema
RUN cd config && uv run python compile_schema.py
RUN rm -rf rag


# Copy server dependency files and README
COPY server/pyproject.toml server/uv.lock server/README.md ./server/

# Install server dependencies
RUN cd server && uv sync --locked --verbose

# Fix ownership and permissions of UV directories after they've been populated
RUN chmod -R 777 /tmp/uv-cache /tmp/uv-venv

# Copy server source code
COPY server/ ./server/

# Expose port
EXPOSE 8000

# Change to /tmp so Celery control files are created in writable location
WORKDIR /tmp

# Run the application from the server directory
CMD ["uv", "run", "--project", "/app/server", "python", "/app/server/main.py"]
