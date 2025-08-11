# Use Python 3.12 slim image
FROM python:3.12-slim

RUN apt update && apt install -y curl && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/


# Copy the required packages from the monorepo
# Note: This Dockerfile should be run from the repo root: docker build -f server/Dockerfile .
COPY config/ ./config/
COPY rag/ ./rag/
COPY models/ ./models/

# Install each package with its own dependencies
RUN ls -la config
RUN cd config && uv sync --locked
RUN cd rag && uv sync --locked
RUN cd models && uv sync --locked

# Copy server dependency files and README
COPY server/pyproject.toml server/uv.lock server/README.md ./server/

# Install server dependencies
RUN cd server && uv sync --locked

# Copy server source code
COPY server/ ./server/

# Expose port
EXPOSE 8000

WORKDIR /app/server

# Run the application
CMD ["uv", "run", "python", "main.py"]
