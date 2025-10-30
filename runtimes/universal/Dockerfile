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

# Install config dependencies including dev dependencies for datamodel generation
RUN ls -la config
RUN cd config && uv sync --locked --all-extras --verbose

# Generate config schema and datamodel types
RUN cd config && ./generate-types.sh
RUN rm -rf rag server


# Copy universal runtime dependency files and README
COPY runtimes/universal/pyproject.toml runtimes/universal/uv.lock runtimes/universal/README.md ./runtimes/universal/

# Install server dependencies and clean up cache to reduce layer size
RUN cd runtimes/universal && uv sync --locked --verbose \
    && uv cache clean \
    && rm -rf /tmp/uv-cache/*

# Copy universal runtime source code
COPY runtimes/universal/ ./runtimes/universal/

# Expose port
EXPOSE 8000

# Change to /tmp so any runtime files are created in writable location
WORKDIR /tmp

# Run the application from the universal runtime directory
CMD ["uv", "run", "--project", "/app/runtimes/universal", "python", "/app/runtimes/universal/server.py"]
