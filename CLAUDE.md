# LlamaFarm

Modular AI development framework. Build AI apps locally with YAML configs, CLI, and your own data.

## Monorepo overview

This repository consists of multiple components

**CLI**
Description: Primary user interface for LlamaFarm. Orchestrates local service lifecycle (starting/stopping server, RAG, and runtime processes), handles project initialization (`lf init`), provides interactive chat sessions (`lf chat`), and coordinates model downloads.
Tech Stack: Go 1.24+, Cobra
Directory: `cli/`

**Server**
Description: Central API gateway that coordinates all LlamaFarm services. Routes inference requests to the Universal Runtime, dispatches RAG tasks to Celery workers, and serves the Designer UI in production.
Tech Stack: Python 3.12+, FastAPI, Celery
Directory: `server/`

**RAG**
Description: Celery-based worker for retrieval-augmented generation. Handles document ingestion, embedding generation, vector storage, and semantic search.
Tech Stack: Python 3.11+, LlamaIndex, ChromaDB, Celery
Directory: `rag/`

**Universal Runtime**
Description: Local ML inference server with OpenAI-compatible API. Supports text generation, embeddings, OCR, anomaly detection, and classification across HuggingFace and GGUF models.
Tech Stack: Python 3.11+, transformers, torch, llama-cpp-python
Directory: `runtimes/universal/`

**Designer**
Description: Browser-based project workbench for building AI applications. Provides config editing, chat testing, dataset management, RAG configuration, and model selection.
Tech Stack: TypeScript, React, Vite, TanStack Query
Directory: `designer/`

**Build**
Description: Nx monorepo configuration for cross-project builds and development workflows.
Tech Stack: Nx workspace
Directory: (root)

See `.claude/rules/` for detailed guidelines.

## Building and Running

- Use the `nx` package to build and run projects within this monorepo.
- Check the `project.json` files at the repo root and project directories for available scripts

### Examples
- `nx dev` - Starts server, rag, and designer together
- `nx run generate-types` - Compiles the llamfarm schema and generates python and go types
- `nx start server` - Starts the server
- `nx start rag` - Starts the RAG celery worker
- `nx start universal-runtime` - Starts the universal runtime
- `nx build cli` - Builds the CLI binary and places it at `./dist/lf`
