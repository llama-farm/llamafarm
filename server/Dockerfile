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

# Copy server source code
COPY server/ ./server/

# Expose port
EXPOSE 8000

WORKDIR /app/server

# Run the application
CMD ["uv", "run", "python", "main.py"]
