---
name: python-skills
description: Shared Python best practices for LlamaFarm. Covers patterns, async, typing, testing, error handling, and security.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# Python Skills for LlamaFarm

Shared Python best practices and code review checklists for all Python components in the LlamaFarm monorepo.

## Applicable Components

| Component | Path | Python | Key Dependencies |
|-----------|------|--------|-----------------|
| Server | `server/` | 3.12+ | FastAPI, Celery, Pydantic, structlog |
| RAG | `rag/` | 3.11+ | LlamaIndex, ChromaDB, Celery |
| Universal Runtime | `runtimes/universal/` | 3.11+ | PyTorch, transformers, FastAPI |
| Config | `config/` | 3.11+ | Pydantic, JSONSchema |
| Common | `common/` | 3.10+ | HuggingFace Hub |

## Quick Reference

| Topic | File | Key Points |
|-------|------|------------|
| Patterns | [patterns.md](patterns.md) | Dataclasses, Pydantic, comprehensions, imports |
| Async | [async.md](async.md) | async/await, asyncio, concurrent execution |
| Typing | [typing.md](typing.md) | Type hints, generics, protocols, Pydantic |
| Testing | [testing.md](testing.md) | Pytest fixtures, mocking, async tests |
| Errors | [error-handling.md](error-handling.md) | Custom exceptions, logging, context managers |
| Security | [security.md](security.md) | Path traversal, injection, secrets, deserialization |

## Code Style

LlamaFarm uses `ruff` with shared configuration in `ruff.toml`:

```toml
line-length = 88
target-version = "py311"
select = ["E", "F", "I", "B", "UP", "SIM"]
```

Key rules:
- **E, F**: Core pyflakes and pycodestyle
- **I**: Import sorting (isort)
- **B**: Bugbear (common pitfalls)
- **UP**: Upgrade syntax to modern Python
- **SIM**: Simplify code patterns

## Architecture Patterns

### Settings with pydantic-settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings, env_file=".env"):
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 14345

settings = Settings()  # Singleton at module level
```

### Structured Logging with structlog

```python
from core.logging import FastAPIStructLogger  # Server
from core.logging import RAGStructLogger      # RAG
from core.logging import UniversalRuntimeLogger  # Runtime

logger = FastAPIStructLogger(__name__)
logger.info("Operation completed", extra={"count": 10, "duration_ms": 150})
```

### Abstract Base Classes for Extensibility

```python
from abc import ABC, abstractmethod

class Component(ABC):
    def __init__(self, name: str, config: dict[str, Any] | None = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}

    @abstractmethod
    def process(self, documents: list[Document]) -> ProcessingResult:
        pass
```

### Dataclasses for Internal Data

```python
from dataclasses import dataclass, field

@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

### Pydantic Models for API Boundaries

```python
from pydantic import BaseModel, Field, ConfigDict

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] | None = "float"

    model_config = ConfigDict(str_strip_whitespace=True)
```

## Directory Structure

Each Python component follows this structure:

```
component/
├── pyproject.toml     # UV-managed dependencies
├── core/              # Core functionality
│   ├── __init__.py
│   ├── settings.py    # Pydantic Settings
│   └── logging.py     # structlog setup
├── services/          # Business logic (server)
├── models/            # ML models (runtime)
├── tasks/             # Celery tasks (rag)
├── utils/             # Utility functions
└── tests/
    ├── conftest.py    # Shared fixtures
    └── test_*.py
```

## Review Checklist Summary

When reviewing Python code in LlamaFarm:

1. **Patterns** (Medium priority)
   - Modern Python syntax (3.10+ type hints)
   - Dataclass vs Pydantic used appropriately
   - No mutable default arguments

2. **Async** (High priority)
   - No blocking calls in async functions
   - Proper asyncio.Lock usage
   - Cancellation handled correctly

3. **Typing** (Medium priority)
   - Complete return type hints
   - Generic types parameterized
   - Pydantic v2 patterns

4. **Testing** (Medium priority)
   - Fixtures properly scoped
   - Async tests use pytest-asyncio
   - Mocks cleaned up

5. **Errors** (High priority)
   - Custom exceptions with context
   - Structured logging with extra dict
   - Proper exception chaining

6. **Security** (Critical priority)
   - Path traversal prevention
   - Input sanitization
   - Safe deserialization

See individual topic files for detailed checklists with grep patterns.
