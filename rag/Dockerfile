FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim

ENV UV_COMPILE_BYTECODE=0
ENV UV_COMPILE_BYTECODE_WORKERS=1
ENV UV_COMPILE_BYTECODE_TIMEOUT=15

ENV LF_DATA_DIR=/var/lib/llamafarm

RUN apt update && apt install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY config/ ./config/
COPY rag/ ./rag/
COPY models/ ./models/

RUN ls -la config
RUN cd config && uv sync --locked --verbose
RUN cd rag && uv sync --locked --verbose
RUN cd models && uv sync --locked --verbose

COPY server/pyproject.toml server/uv.lock server/README.md ./server/

RUN cd server && uv sync --locked --verbose

COPY server/ ./server/

EXPOSE 8001

WORKDIR /app/rag

CMD ["uv", "run", "python", "-m", "rag.worker"]
