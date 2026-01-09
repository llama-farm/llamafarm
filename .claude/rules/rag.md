# RAG Service

## Overview
- Celery-based worker for retrieval-augmented generation pipelines
- Handles document ingestion, embedding, and semantic search
- Runs as a separate process receiving tasks from the Server
- Uses ChromaDB for vector storage

## Architecture

### Entry Points
- `rag/celery_app.py` - Celery application configuration and worker entry
- `rag/debug_tasks.py` - Debug utilities for task inspection

### Directory Structure
- **tasks/** - Celery task definitions
  - `ingest_tasks.py` - Document ingestion tasks
  - `query_tasks.py` - RAG query execution
  - `search_tasks.py` - Semantic search operations
  - `delete_tasks.py` - Document/collection deletion
  - `stats_tasks.py` - Collection statistics
  - `health_tasks.py` - Health check tasks
- **core/** - Pipeline logic
  - `enhanced_pipeline.py` - Main RAG pipeline orchestration
  - `ingest_handler.py` - Document ingestion workflow
  - `document_manager.py` - Document lifecycle management
  - `blob_processor.py` - Binary file processing
  - `strategies/` - Pluggable retrieval strategies
- **components/** - Modular RAG components
  - `embedders/` - Embedding model integrations (calls Universal Runtime)
  - `extractors/` - Content extraction (text, metadata)
  - `parsers/` - Document parsers (PDF, DOCX, etc.)
  - `preprocessors/` - Text preprocessing
  - `retrievers/` - Retrieval implementations
  - `stores/` - Vector store adapters (ChromaDB)
  - `metadata/` - Metadata extraction and handling
- **utils/** - Shared utilities

### Data Flow
1. Server dispatches ingest/query task via Celery
2. Task handler loads document(s)
3. Parser extracts text content
4. Preprocessor cleans/chunks text
5. Embedder generates vectors (via Universal Runtime API)
6. Store persists vectors to ChromaDB
7. Retriever performs similarity search on query

## Development

### Running
- `nx start rag` or `uv run celery -A rag.celery_app worker --loglevel=info`
- Requires broker (filesystem-based by default, or Redis/RabbitMQ)

### Testing
- `cd rag && uv run pytest`
- Tests in `rag/tests/`
