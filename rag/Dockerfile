# Use Python 3.12 slim image
FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

# Disable/limit bytecode compilation to avoid CI stalls
ENV UV_COMPILE_BYTECODE=0
ENV UV_COMPILE_BYTECODE_WORKERS=1
ENV UV_COMPILE_BYTECODE_TIMEOUT=15

ENV LF_DATA_DIR=/var/lib/llamafarm

RUN apt update && apt install -y curl && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy the required packages from the monorepo
# Note: This Dockerfile should be run from the repo root: docker build -f server/Dockerfile .
COPY config/ ./config/
COPY rag/ ./rag/

# Install each package with its own dependencies
RUN ls -la config
RUN cd config && uv sync --locked  --verbose
RUN cd rag && uv sync --locked  --verbose

# Compile the config schema
RUN cd config && uv run python compile_schema.py

WORKDIR /app/rag

# Run the RAG Celery worker
CMD ["uv", "run", "python", "main.py"]
