# Server

## Overview
- Central API gateway coordinating all LlamaFarm services
- Routes inference requests to the Universal Runtime
- Dispatches RAG tasks to Celery workers
- Serves the Designer UI in production
- Provides MCP (Model Context Protocol) server endpoints

## Architecture

### Entry Points
- `server/main.py` - Application entry point, FastAPI app initialization
- `server/api/main.py` - API factory and route registration

### Directory Structure
- **api/** - REST API layer
  - `routers/` - FastAPI route handlers (projects, chat, datasets, models, etc.)
  - `middleware/` - Request/response middleware (logging, CORS, auth)
- **services/** - Business logic layer
  - `runtime_service/` - Communication with Universal Runtime
  - Various service modules for projects, datasets, chat orchestration
- **agents/** - AI agent implementations
  - `base/` - Base agent classes and interfaces
- **context_providers/** - Context injection for chat/inference
- **tools/** - Tool implementations for agents
  - `mcp_tool/` - MCP tool factory and handlers
- **core/** - Shared infrastructure
  - `config.py` - Configuration loading
  - `settings.py` - Environment-based settings
  - `celery/` - Celery client for RAG task dispatch
- **seeds/** - Default project templates copied on startup

### Key APIs
- `/api/projects` - Project CRUD operations
- `/api/chat` - Chat completions (proxies to Runtime)
- `/api/datasets` - Dataset management
- `/api/models` - Model listing and configuration
- `/api/rag` - RAG operations (delegates to Celery worker)
- `/mcp` - MCP server endpoints

## Development

### Running
- `nx start server` or `uv run uvicorn server.main:app --reload`
- Default port: 8000

### Testing
- `cd server && uv run pytest`
- Tests in `server/tests/`
