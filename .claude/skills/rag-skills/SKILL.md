---
name: rag-skills
description: RAG-specific best practices for LlamaIndex, ChromaDB, and Celery workers. Covers ingestion, retrieval, embeddings, and performance.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# RAG Skills for LlamaFarm

Framework-specific patterns and code review checklists for the RAG component.

**Extends**: [python-skills](../python-skills/SKILL.md) - All Python best practices apply here.

## Component Overview

| Aspect | Technology | Version |
|--------|------------|---------|
| Python | Python | 3.11+ |
| Document Processing | LlamaIndex | 0.13+ |
| Vector Storage | ChromaDB | 1.0+ |
| Task Queue | Celery | 5.5+ |
| Embeddings | Universal/Ollama/OpenAI | Multiple |

## Directory Structure

```
rag/
├── api.py                 # Search and database APIs
├── celery_app.py          # Celery configuration
├── main.py                # Entry point
├── core/
│   ├── base.py            # Document, Component, Pipeline ABCs
│   ├── factories.py       # Component factories
│   ├── ingest_handler.py  # File ingestion with safety checks
│   ├── blob_processor.py  # Binary file processing
│   ├── settings.py        # Pydantic settings
│   └── logging.py         # RAGStructLogger
├── components/
│   ├── embedders/         # Embedding providers
│   ├── extractors/        # Metadata extractors
│   ├── parsers/           # Document parsers (LlamaIndex)
│   ├── retrievers/        # Retrieval strategies
│   └── stores/            # Vector stores (ChromaDB, FAISS)
├── tasks/                 # Celery tasks
│   ├── ingest_tasks.py    # File ingestion
│   ├── search_tasks.py    # Database search
│   ├── query_tasks.py     # Complex queries
│   ├── health_tasks.py    # Health checks
│   └── stats_tasks.py     # Statistics
└── utils/
    └── embedding_safety.py  # Circuit breaker, validation
```

## Quick Reference

| Topic | File | Key Points |
|-------|------|------------|
| LlamaIndex | [llamaindex.md](llamaindex.md) | Document parsing, chunking, node conversion |
| ChromaDB | [chromadb.md](chromadb.md) | Collections, embeddings, distance metrics |
| Celery | [celery.md](celery.md) | Task routing, error handling, worker config |
| Performance | [performance.md](performance.md) | Batching, caching, deduplication |

## Core Patterns

### Document Dataclass

```python
from dataclasses import dataclass, field
from typing import Any

@dataclass
class Document:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str | None = None
    embeddings: list[float] | None = None
```

### Component Abstract Base Class

```python
from abc import ABC, abstractmethod

class Component(ABC):
    def __init__(
        self,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.logger = RAGStructLogger(__name__).bind(name=self.name)
        self.project_dir = project_dir

    @abstractmethod
    def process(self, documents: list[Document]) -> ProcessingResult:
        pass
```

### Retrieval Strategy Pattern

```python
class RetrievalStrategy(Component, ABC):
    @abstractmethod
    def retrieve(
        self,
        query_embedding: list[float],
        vector_store,
        top_k: int = 5,
        **kwargs
    ) -> RetrievalResult:
        pass

    @abstractmethod
    def supports_vector_store(self, vector_store_type: str) -> bool:
        pass
```

### Embedder with Circuit Breaker

```python
class Embedder(Component):
    DEFAULT_FAILURE_THRESHOLD = 5
    DEFAULT_RESET_TIMEOUT = 60.0

    def __init__(self, ...):
        super().__init__(...)
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.get("failure_threshold", 5),
            reset_timeout=config.get("reset_timeout", 60.0),
        )
        self._fail_fast = config.get("fail_fast", True)

    def embed_text(self, text: str) -> list[float]:
        self.check_circuit_breaker()
        try:
            embedding = self._call_embedding_api(text)
            self.record_success()
            return embedding
        except Exception as e:
            self.record_failure(e)
            if self._fail_fast:
                raise EmbedderUnavailableError(str(e)) from e
            return [0.0] * self.get_embedding_dimension()
```

## Review Checklist Summary

When reviewing RAG code:

1. **LlamaIndex** (Medium priority)
   - Proper chunking configuration
   - Metadata preservation during parsing
   - Error handling for unsupported formats

2. **ChromaDB** (High priority)
   - Thread-safe client access
   - Proper distance metric selection
   - Metadata type compatibility

3. **Celery** (High priority)
   - Task routing to correct queue
   - Error logging with context
   - Proper serialization

4. **Performance** (Medium priority)
   - Batch processing for embeddings
   - Deduplication enabled
   - Appropriate caching

See individual topic files for detailed checklists with grep patterns.
