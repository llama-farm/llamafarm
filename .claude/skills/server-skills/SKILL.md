---
name: server-skills
description: Server-specific best practices for FastAPI, Celery, and Pydantic. Extends python-skills with framework-specific patterns.
allowed-tools: Read, Grep, Glob, Bash
user-invocable: false
---

# Server Skills for LlamaFarm

Framework-specific patterns and code review checklists for the LlamaFarm Server component.

## Overview

| Property | Value |
|----------|-------|
| Path | `server/` |
| Python | 3.12+ |
| Framework | FastAPI 0.116+ |
| Task Queue | Celery 5.5+ |
| Validation | Pydantic 2.x, pydantic-settings |
| Logging | structlog with FastAPIStructLogger |

## Links to Shared Skills

This skill extends the shared Python skills. See:

- [Python Patterns](../python-skills/patterns.md) - Dataclasses, comprehensions, imports
- [Async Patterns](../python-skills/async.md) - async/await, asyncio, concurrency
- [Typing Patterns](../python-skills/typing.md) - Type hints, generics, Pydantic
- [Testing Patterns](../python-skills/testing.md) - Pytest, fixtures, mocking
- [Error Handling](../python-skills/error-handling.md) - Exceptions, logging, context managers
- [Security Patterns](../python-skills/security.md) - Path traversal, injection, secrets

## Server-Specific Checklists

| Topic | File | Key Points |
|-------|------|------------|
| FastAPI | [fastapi.md](fastapi.md) | Routes, dependencies, middleware, exception handlers |
| Celery | [celery.md](celery.md) | Task patterns, error handling, retries, signatures |
| Pydantic | [pydantic.md](pydantic.md) | Pydantic v2 models, validation, serialization |
| Performance | [performance.md](performance.md) | Async patterns, caching, connection pooling |

## Architecture Overview

```
server/
├── main.py                 # Uvicorn entry point, MCP mount
├── api/
│   ├── main.py             # FastAPI app factory, middleware setup
│   ├── errors.py           # Custom exceptions + exception handlers
│   ├── middleware/         # ASGI middleware (structlog, errors)
│   └── routers/            # API route modules
│       ├── projects/       # Project CRUD endpoints
│       ├── datasets/       # Dataset management
│       ├── rag/            # RAG query endpoints
│       └── ...
├── core/
│   ├── settings.py         # pydantic-settings configuration
│   ├── logging.py          # structlog setup, FastAPIStructLogger
│   └── celery/             # Celery app configuration
│       ├── celery.py       # Celery app instance
│       └── rag_client.py   # RAG task signatures and helpers
├── services/               # Business logic layer
│   ├── project_service.py  # Project CRUD operations
│   ├── dataset_service.py  # Dataset management
│   └── ...
├── agents/                 # AI agent implementations
└── tests/                  # Pytest test suite
```

## Quick Reference

### Settings Pattern (pydantic-settings)

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings, env_file=".env"):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

settings = Settings()  # Module-level singleton
```

### Structured Logging

```python
from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)
logger.info("Operation completed", extra={"count": 10, "duration_ms": 150})
logger.bind(namespace=namespace, project=project_id)  # Add context
```

### Custom Exceptions

```python
# Define exception hierarchy
class NotFoundError(Exception): ...
class ProjectNotFoundError(NotFoundError):
    def __init__(self, namespace: str, project_id: str):
        self.namespace = namespace
        self.project_id = project_id
        super().__init__(f"Project {namespace}/{project_id} not found")

# Register handler in api/errors.py
async def _handle_project_not_found(request: Request, exc: Exception) -> Response:
    payload = ErrorResponse(error="ProjectNotFound", message=str(exc))
    return JSONResponse(status_code=404, content=payload.model_dump())

def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ProjectNotFoundError, _handle_project_not_found)
```

### Service Layer Pattern

```python
class ProjectService:
    @classmethod
    def get_project(cls, namespace: str, project_id: str) -> Project:
        project_dir = cls.get_project_dir(namespace, project_id)
        if not os.path.isdir(project_dir):
            raise ProjectNotFoundError(namespace, project_id)
        # ... load and validate
```

## Review Checklist Summary

1. **FastAPI Routes** (High priority)
   - Proper async/sync function choice
   - Response model defined with `response_model=`
   - OpenAPI metadata (operation_id, tags, summary)
   - HTTPException with proper status codes

2. **Celery Tasks** (High priority)
   - Use signatures for cross-service calls
   - Implement proper timeout and polling
   - Handle task failures gracefully
   - Store group metadata for parallel tasks

3. **Pydantic Models** (Medium priority)
   - Use Pydantic v2 patterns (model_config, Field)
   - Proper validation with field constraints
   - Serialization with model_dump()

4. **Performance** (Medium priority)
   - Avoid blocking calls in async functions
   - Use proper connection pooling for external services
   - Implement caching where appropriate

See individual topic files for detailed checklists with grep patterns.
