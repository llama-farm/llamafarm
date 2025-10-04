# Use Python 3.12 slim image with multi-arch support
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install uv for fast Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Copy dependency files
COPY runtime/pyproject.toml runtime/uv.lock ./

# Install dependencies
RUN uv sync --locked --no-dev

# Copy source code
COPY runtime/ .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uv", "run", "python", "main.py"]
