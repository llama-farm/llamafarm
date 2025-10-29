# Observability Architecture - Phase 1: Event Logging

## Overview

Add **universal event logging** for all LlamaFarm operations (inference, RAG processing, dataset ingestion, etc.) using **filestore-only storage**. All events live in `~/.llamafarm/projects/{ns}/{proj}/event_logs/` with a simple API for retrieval.

**Phase 1 Scope** (This PR):
- ✅ Universal event logger (shared across inference + RAG + all components)
- ✅ Event logging API (read-only endpoints)
- ✅ Config versioning (hash-based config storage)
- ❌ Metrics API (future)
- ❌ CLI commands (future)
- ❌ Log shipping plugins (future)
- ❌ Enhanced file metadata (future)

## Key Concepts

### 1. Universal Event Logger
Single `EventLogger` class used by **ALL** components (server, RAG worker, etc.):
- Start a new event with `start_event(event_type, request_id)`
- Log sub-events with `log_event(event_name, data)`
- Complete event with `complete_event()` or `fail_event(error)`
- Handles all file I/O, config hashing, buffering
- Thread-safe, async writes

### 2. Config Versioning
Every event includes a `config_hash` that references the project config snapshot:
- Hash config → store in `configs/{hash}.json`
- Include hash in event log
- Later: compare performance across config versions
- Audit trail for config changes

### 3. Simple Event Structure
All events follow the same schema:
- Top-level event container (event_id, type, config_hash, status)
- Flat list of timestamped sub-events inside
- Each sub-event has: timestamp, event_name, duration_ms, data

## New API Endpoints

### Event Logs API (Phase 1)
```
GET /v1/projects/{ns}/{id}/event_logs
  Query params:
    - type: "inference" | "rag_processing" | "dataset_ingestion" (filter by event type)
    - start_time: ISO 8601 timestamp
    - end_time: ISO 8601 timestamp
    - limit: int (default: 10, max: 100)
    - offset: int (for pagination)

GET /v1/projects/{ns}/{id}/event_logs/{event_id}
  Returns: Full event log with all sub-events
```

### Future Endpoints (NOT in Phase 1)
```
# Metrics API (Phase 2+)
GET /v1/projects/{ns}/{id}/metrics/inference
GET /v1/projects/{ns}/{id}/metrics/rag

# Files/Metadata API (Phase 3+)
GET /v1/projects/{ns}/{id}/files
```

## File Structure Changes

### New Universal Event Log Storage

**Base Directory** (configurable via `LF_DATA_DIR` environment variable):
```
# Local development (default)
~/.llamafarm/projects/{namespace}/{project}/

# Docker (via volume mount)
/data/projects/{namespace}/{project}/

# Custom
${LF_DATA_DIR}/projects/{namespace}/{project}/
```

**Event Logs Structure**:
```
${LF_DATA_DIR}/projects/{namespace}/{project}/
  event_logs/                                    # NEW: Universal event logs
    evt_inference_20251029_143022_abc123.json
    evt_rag_processing_20251029_143525_def456.json
    evt_dataset_ingestion_20251029_144010_ghi789.json

  configs/                                       # NEW: Config snapshots
    sha256_a1b2c3d4e5f6.json
    sha256_f6e5d4c3b2a1.json

  lf_data/                                       # Existing data
    meta/
    raw/
    stores/
    logs/                                        # Old processing logs
```

**Event ID Format**: `evt_{type}_{timestamp}_{random}`
- `evt_inference_20251029_143022_abc123.json`
- `evt_rag_processing_20251029_143525_def456.json`

**Why Single Directory**:
- Unified event log system for all operations
- Simple chronological ordering
- Easy cross-component correlation
- Future-ready for advanced querying
- **Docker-friendly** - single volume mount point

## New Files to Create (Phase 1 Only)

### Shared Library (Root Level)
Create a new **shared observability package** at the repository root:

```
observability/                           # NEW: Shared observability library
  __init__.py
  event_logger.py                        # Universal EventLogger class
  config_versioning.py                   # Config hashing/versioning utilities
```

**Why root-level shared package?**
- ✅ Server can import: `from observability.event_logger import EventLogger`
- ✅ RAG can import: `from observability.event_logger import EventLogger`
- ✅ Future runtimes can import: `from observability.event_logger import EventLogger`
- ✅ CLI can read events without depending on server/rag code
- ✅ No circular dependencies between server ↔ rag

**Universal EventLogger** (`observability/event_logger.py`)
- Class: `EventLogger`
- **Super Simple Interface**:
  - `__init__(event_type, request_id, namespace, project, config_hash)` → EventLogger
  - `log_event(event_name, data)` → void  # Just throw any dict at it!
  - `complete_event()` → void
  - `fail_event(error)` → void
- **Thread-safe** - handles parallel sub-events automatically
- **No JSON formatting required** - pass plain Python dicts, logger handles serialization
- **Buffered writes** - all I/O handled internally
- **Zero dependencies** on server or rag code

**Key Design Principle**:
> Caller just throws data at the logger. Logger handles all the complexity (threading, JSON, I/O, buffering).

**Config Versioning** (`observability/config_versioning.py`)
- `hash_config(config)` → config_hash
- `save_config_snapshot(config, config_hash, namespace, project)` → void
- `get_config_by_hash(config_hash, namespace, project)` → config
- Works with Pydantic `LlamaFarmConfig` objects

### API Endpoints (Server Only)
```
server/api/routers/event_logs/
  __init__.py
  router.py          # Event logs API endpoints
  models.py          # Pydantic models for requests/responses
```

Endpoints:
- `GET /event_logs` - List events with filtering
- `GET /event_logs/{event_id}` - Get single event

**Note**: API endpoints use the shared `observability` library to read event logs

## Files to Modify (Phase 1 Only)

### Server (Inference Logging)
```
server/services/project_chat_service.py
```
- Import: `from observability.event_logger import EventLogger`
- Import: `from observability.config_versioning import hash_config, save_config_snapshot`
- Integrate EventLogger at request entry
- Log sub-events for each step
- Complete event on success/failure

### RAG (Processing Logging)
```
rag/core/ingest_handler.py
```
- Import: `from observability.event_logger import EventLogger`
- Import: `from observability.config_versioning import hash_config, save_config_snapshot`
- Integrate EventLogger at processing start
- Log sub-events for parsing, chunking, embedding, storage
- Complete event on success/failure

### Future Runtimes (Universal, Lemonade, etc.)
**Same shared library**:
```python
from observability.event_logger import EventLogger
from observability.config_versioning import hash_config

# Example: Universal runtime logging model load
logger = EventLogger(
    event_type="model_load",
    request_id=f"load_{uuid.uuid4().hex[:12]}",
    namespace=namespace,
    project=project,
    config_hash=hash_config(config)
)

logger.log_event("model_download_start", {...})
logger.log_event("model_loaded", {...})
logger.complete_event()
```

## Configuration Changes (Phase 1)

**NO configuration needed** - using sensible hardcoded defaults:

```python
# server/core/event_logger.py
DEFAULT_CONFIG = {
    "enabled": True,
    "buffer_size": 10,
    "debug_mode": False,
    "chunk_preview_length": 100,
    "max_chunks_preview": 2,
    "content_preview_length": 200,
}
```

**Future**: Add `observability` section to `llamafarm.yaml` for user customization (Phase 2+)

## Universal Event Schema

All events follow the same structure (unified for inference, RAG, etc.):

```json
{
  "event_id": "evt_inference_20251029_143022_abc123",
  "event_type": "inference",
  "request_id": "req_abc123",
  "timestamp": "2025-10-29T14:30:22.123456Z",
  "namespace": "default",
  "project": "my-project",
  "config_hash": "sha256_a1b2c3d4e5f6",

  "events": [
    {
      "timestamp": "2025-10-29T14:30:22.123Z",
      "event_name": "request_received",
      "duration_ms": 0,
      "data": {
        "endpoint": "/v1/projects/default/my-project/chat/completions",
        "method": "POST"
      }
    },
    {
      "timestamp": "2025-10-29T14:30:22.125Z",
      "event_name": "rag_query_start",
      "duration_ms": 2,
      "data": {
        "database": "main_database",
        "query": "What are neural scaling laws?",
        "top_k": 5
      }
    },
    {
      "timestamp": "2025-10-29T14:30:22.170Z",
      "event_name": "rag_retrieval_complete",
      "duration_ms": 45,
      "data": {
        "chunks_retrieved": 5,
        "avg_score": 0.88,
        "chunks_preview": [
          {
            "rank": 1,
            "content_preview": "Neural scaling laws describe...",
            "source": "paper.pdf",
            "score": 0.92
          }
        ]
      }
    },
    {
      "timestamp": "2025-10-29T14:30:22.175Z",
      "event_name": "llm_inference_start",
      "duration_ms": 3,
      "data": {
        "model": "gemma3:1b",
        "runtime": "ollama"
      }
    },
    {
      "timestamp": "2025-10-29T14:30:23.375Z",
      "event_name": "llm_inference_complete",
      "duration_ms": 1200,
      "data": {
        "completion_tokens": 80,
        "total_tokens": 230,
        "finish_reason": "stop"
      }
    },
    {
      "timestamp": "2025-10-29T14:30:23.377Z",
      "event_name": "response_complete",
      "duration_ms": 2,
      "data": {
        "content_preview": "Neural scaling laws describe...",
        "total_latency_ms": 1254
      }
    }
  ],

  "status": "completed",
  "error": null,
  "metadata": {
    "client_ip": "127.0.0.1",
    "user_agent": "LlamaFarmCLI/1.0"
  }
}
```

### Event Types
- `inference` - Chat/completion requests
- `rag_processing` - Dataset ingestion and processing
- `dataset_ingestion` - File uploads
- (Future: `rag_query`, `embedding_generation`, etc.)

## Config Versioning (Efficient Deduplication)

All events include a `config_hash` field that references the project config at the time of execution.

### Config Storage Location
```
${LF_DATA_DIR}/projects/{namespace}/{project}/configs/
  sha256_a1b2c3d4e5f6.json  # Full config snapshot (version 1)
  sha256_f6e5d4c3b2a1.json  # Different config version (version 2)
  sha256_a1b2c3d4e5f6.json  # Same as version 1 - NOT saved again
```

**Location**: Project-scoped at `${LF_DATA_DIR}/projects/{namespace}/{project}/configs/`
- Local: `~/.llamafarm/projects/{namespace}/{project}/configs/`
- Docker: `/data/projects/{namespace}/{project}/configs/` (via volume mount)

### How It Works (Low Overhead, Smart Deduplication)

#### 1. Hash Config (Fast, Deterministic)
```python
# observability/config_versioning.py

import hashlib
import json
from pathlib import Path
from typing import Any

def hash_config(config: Any) -> str:
    """
    Generate deterministic hash of config (FAST).

    Uses SHA256 for collision resistance.
    Excludes dynamic fields (timestamps, etc).

    Args:
        config: LlamaFarmConfig pydantic object

    Returns:
        Short hash like "sha256_a1b2c3d4e5f6"
    """
    # Serialize config deterministically
    # - Sort keys for consistency
    # - Exclude None values
    # - Exclude dynamic timestamp fields
    config_json = config.model_dump_json(
        sort_keys=True,
        exclude_none=True,
        exclude={'created_at', 'updated_at', 'last_modified'}
    )

    # Fast SHA256 hash
    hash_digest = hashlib.sha256(config_json.encode('utf-8')).hexdigest()

    # Return short hash (16 chars = 64 bits, good enough for collision resistance)
    return f"sha256_{hash_digest[:16]}"


def save_config_snapshot(
    config: Any,
    config_hash: str,
    namespace: str,
    project: str
) -> bool:
    """
    Save config snapshot ONLY if it doesn't already exist (deduplication).

    Docker-compatible: Uses LF_DATA_DIR env var for path resolution.

    Returns:
        True if new snapshot saved, False if already exists
    """
    import os

    # Docker-compatible path resolution
    data_dir = os.getenv('LF_DATA_DIR', str(Path.home() / ".llamafarm"))
    project_dir = Path(data_dir) / "projects" / namespace / project
    configs_dir = project_dir / "configs"
    configs_dir.mkdir(parents=True, exist_ok=True)

    config_file = configs_dir / f"{config_hash}.json"

    # Check if already exists (deduplication)
    if config_file.exists():
        return False  # Already saved, no overhead

    # Save new config snapshot
    config_json = config.model_dump_json(indent=2, exclude_none=True)

    # Atomic write
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(
        'w',
        delete=False,
        dir=configs_dir,
        suffix='.tmp'
    ) as tmp:
        tmp.write(config_json)
        tmp_path = tmp.name

    # Atomic move (POSIX systems)
    os.replace(tmp_path, config_file)

    return True  # New snapshot saved


def get_config_by_hash(
    config_hash: str,
    namespace: str,
    project: str
) -> dict | None:
    """
    Retrieve config snapshot by hash.

    Docker-compatible: Uses LF_DATA_DIR env var for path resolution.

    Returns:
        Config dict or None if not found
    """
    import os

    # Docker-compatible path resolution
    data_dir = os.getenv('LF_DATA_DIR', str(Path.home() / ".llamafarm"))
    project_dir = Path(data_dir) / "projects" / namespace / project
    config_file = project_dir / "configs" / f"{config_hash}.json"

    if not config_file.exists():
        return None

    with open(config_file) as f:
        return json.load(f)
```

#### 2. Usage Pattern (Minimal Overhead)
```python
# In project_chat_service.py or ingest_handler.py

from observability.config_versioning import hash_config, save_config_snapshot

# Get project config (already loaded in memory)
project = ProjectService.get_project(namespace, project_id)
config = project.config

# Hash config (FAST: ~0.1ms for typical config)
config_hash = hash_config(config)

# Save snapshot (FAST: only if new, filesystem check + atomic write)
is_new = save_config_snapshot(config, config_hash, namespace, project_id)

if is_new:
    logger.info(f"New config version: {config_hash}")
else:
    logger.debug(f"Using existing config version: {config_hash}")

# Create event logger with config hash
event_logger = EventLogger(
    event_type="inference",
    request_id=request_id,
    namespace=namespace,
    project=project_id,
    config_hash=config_hash  # Just a string reference
)
```

### Performance Characteristics

| Operation | Time | Overhead |
|-----------|------|----------|
| Hash config | ~0.1ms | Negligible (SHA256 on ~10KB JSON) |
| Check if exists | ~0.01ms | Single filesystem stat call |
| Save new snapshot | ~1-2ms | Only on config change (rare) |
| **Total per request** | **~0.11ms** | **Minimal** |

### Deduplication Benefits

**Scenario**: 10,000 inference requests with same config
- **Without deduplication**: Save 10,000 copies of config (100MB+)
- **With deduplication**: Save 1 copy, check existence 9,999 times
  - Disk space: 10KB (single config)
  - Time overhead: 0.11ms per request
  - Total savings: 99.99% disk space, near-zero overhead

**Scenario**: Config changes 3 times over day
- Save exactly 3 snapshots
- All events reference correct config_hash
- Easy to correlate performance changes with config versions

### Example Config Snapshots

**Config Version 1** (`sha256_a1b2c3d4e5f6.json`):
```json
{
  "version": "v1",
  "name": "my-project",
  "namespace": "default",
  "runtime": {
    "default_model": "fast",
    "models": {
      "fast": {
        "provider": "ollama",
        "model": "gemma3:1b",
        "base_url": "http://localhost:11434/v1"
      }
    }
  },
  "rag": {
    "databases": [
      {
        "name": "main_database",
        "type": "chroma",
        "retrieval_strategies": [...]
      }
    ]
  }
}
```

**Config Version 2** (`sha256_f6e5d4c3b2a1.json`):
```json
{
  "version": "v1",
  "name": "my-project",
  "namespace": "default",
  "runtime": {
    "default_model": "powerful",  // CHANGED
    "models": {
      "fast": {...},
      "powerful": {                // NEW
        "provider": "ollama",
        "model": "qwen3:8b",
        "base_url": "http://localhost:11434/v1"
      }
    }
  },
  "rag": {...}
}
```

### Event Log Reference

Events reference config by hash:
```json
{
  "event_id": "evt_inference_20251029_143022_abc123",
  "config_hash": "sha256_a1b2c3d4e5f6",  // Points to config version 1
  "events": [...]
}
```

### Future: Config Comparison API

```
GET /v1/projects/{ns}/{id}/configs/{hash1}/compare/{hash2}

Response:
{
  "hash1": "sha256_a1b2c3d4e5f6",
  "hash2": "sha256_f6e5d4c3b2a1",
  "diff": {
    "changed": [
      "runtime.default_model: 'fast' → 'powerful'",
      "runtime.models.powerful: added"
    ],
    "added": ["runtime.models.powerful"],
    "removed": []
  }
}
```

### Benefits

1. **Low Overhead**:
   - Hash: ~0.1ms (SHA256 on small JSON)
   - Dedup check: ~0.01ms (single filesystem stat)
   - Total: **~0.11ms per request**

2. **Smart Deduplication**:
   - Only save when config actually changes
   - Hash-based: same config = same hash = skip save
   - Typical deployment: 1-5 config versions total

3. **Performance Tracking**:
   - Compare metrics across config versions
   - "Model X is 20% faster than Model Y"
   - "This RAG strategy retrieves better chunks"

4. **Audit Trail**:
   - Know exact config for every inference
   - Reproduce bugs with exact config
   - Track config evolution over time

5. **Disk Efficient**:
   - No redundant config storage
   - Typical project: < 100KB for all config versions
   - Scales to millions of events

## Thread Safety & Parallel Operations

### EventLogger is Thread-Safe

The logger uses internal locking to handle parallel sub-events:

```python
# observability/event_logger.py

import threading
from datetime import datetime, timezone
from typing import Any

class EventLogger:
    """Thread-safe event logger for parallel operations."""

    def __init__(
        self,
        event_type: str,
        request_id: str,
        namespace: str,
        project: str,
        config_hash: str
    ):
        self.event_type = event_type
        self.request_id = request_id
        self.namespace = namespace
        self.project = project
        self.config_hash = config_hash

        # Event storage (internal buffer)
        self._events: list[dict] = []
        self._lock = threading.Lock()  # Thread safety
        self._start_time = datetime.now(timezone.utc)

    def log_event(self, event_name: str, data: dict[str, Any]) -> None:
        """
        Log a sub-event. Thread-safe, no formatting required.

        Just throw any dict at it - logger handles the rest!
        """
        with self._lock:  # Thread-safe
            now = datetime.now(timezone.utc)
            duration_ms = (now - self._start_time).total_seconds() * 1000

            # Simple event structure - logger adds timestamp and duration
            event = {
                "timestamp": now.isoformat(),
                "event_name": event_name,
                "duration_ms": round(duration_ms, 2),
                "data": data  # Caller's data - any dict!
            }

            self._events.append(event)

    def complete_event(self) -> None:
        """Write event to disk. All JSON serialization happens here."""
        with self._lock:
            self._write_to_disk(status="completed", error=None)

    def fail_event(self, error: str) -> None:
        """Write failed event to disk."""
        with self._lock:
            self._write_to_disk(status="failed", error=error)

    def _write_to_disk(self, status: str, error: str | None) -> None:
        """
        Internal method - handles all JSON serialization and I/O.

        Caller never deals with JSON!
        """
        import json
        import uuid
        from pathlib import Path

        # Generate event ID
        timestamp = self._start_time.strftime("%Y%m%d_%H%M%S")
        random_id = uuid.uuid4().hex[:6]
        event_id = f"evt_{self.event_type}_{timestamp}_{random_id}"

        # Build complete event structure
        full_event = {
            "event_id": event_id,
            "event_type": self.event_type,
            "request_id": self.request_id,
            "timestamp": self._start_time.isoformat(),
            "namespace": self.namespace,
            "project": self.project,
            "config_hash": self.config_hash,
            "events": self._events,  # All sub-events
            "status": status,
            "error": error,
            "metadata": {}
        }

        # Write to file (atomic) - Docker-compatible path resolution
        import os

        # Use LF_DATA_DIR env var if set, otherwise default to ~/.llamafarm
        data_dir = os.getenv('LF_DATA_DIR', str(Path.home() / ".llamafarm"))
        event_logs_dir = Path(data_dir) / "projects" / self.namespace / self.project / "event_logs"
        event_logs_dir.mkdir(parents=True, exist_ok=True)

        event_file = event_logs_dir / f"{event_id}.json"

        # Single JSON serialization - happens once at the end
        with open(event_file, 'w') as f:
            json.dump(full_event, f, indent=2)
```

### Usage with Parallel Operations

**Example**: Parallel RAG retrieval + LLM inference

```python
import asyncio
from observability.event_logger import EventLogger

logger = EventLogger(
    event_type="inference",
    request_id=request_id,
    namespace=namespace,
    project=project_id,
    config_hash=config_hash
)

# Parallel operations - both can log simultaneously
async def rag_query():
    logger.log_event("rag_query_start", {"database": "main_db"})
    results = await fetch_rag_results()
    logger.log_event("rag_query_complete", {"chunks": len(results)})
    return results

async def warm_model():
    logger.log_event("model_warmup_start", {"model": "gemma3:1b"})
    await warmup_llm()
    logger.log_event("model_warmup_complete", {})

# Run in parallel - logger handles thread safety
rag_results, _ = await asyncio.gather(
    rag_query(),
    warm_model()
)

# Continue with inference
logger.log_event("inference_start", {"model": "gemma3:1b"})
# ... rest of inference ...

logger.complete_event()
```

**Result**: Events are logged in timestamp order, not call order:
```json
{
  "events": [
    {"timestamp": "2025-10-29T14:30:22.100Z", "event_name": "rag_query_start", ...},
    {"timestamp": "2025-10-29T14:30:22.105Z", "event_name": "model_warmup_start", ...},
    {"timestamp": "2025-10-29T14:30:22.150Z", "event_name": "rag_query_complete", ...},
    {"timestamp": "2025-10-29T14:30:22.180Z", "event_name": "model_warmup_complete", ...},
    {"timestamp": "2025-10-29T14:30:22.200Z", "event_name": "inference_start", ...}
  ]
}
```

### No Formatting Required - Just Plain Dicts

**Caller doesn't deal with JSON or formatting**:

```python
# ❌ BAD: Don't do JSON conversion yourself
logger.log_event("step", json.dumps({"key": "value"}))  # NO!

# ✅ GOOD: Just pass a dict
logger.log_event("step", {"key": "value"})  # YES!

# ✅ GOOD: Any Python data types (logger handles serialization)
logger.log_event("chunk_retrieval", {
    "chunks": 5,
    "avg_score": 0.88,
    "sources": ["paper.pdf", "notes.txt"],
    "metadata": {
        "retrieval_strategy": "hybrid",
        "top_k": 10
    }
})
```

**Logger handles**:
- JSON serialization (once, at the end)
- Timestamp injection
- Duration calculation
- Thread safety
- File I/O
- Buffering

**Caller just provides**:
- Event name (string)
- Event data (plain dict)

### Performance: Single JSON Serialization

```python
# What happens internally:

# During request (NO JSON conversion):
logger.log_event("step1", {"data": 1})  # Just append to list
logger.log_event("step2", {"data": 2})  # Just append to list
logger.log_event("step3", {"data": 3})  # Just append to list

# At the end (SINGLE JSON conversion):
logger.complete_event()  # json.dump() called once for entire event
```

**Performance benefit**:
- ❌ Multiple conversions: 3 × json.dumps() = slower
- ✅ Single conversion: 1 × json.dump() = faster
- No intermediate JSON strings in memory
- Efficient bulk write

## Event Logger Usage Examples

### Inference (Server)
```python
from observability.event_logger import EventLogger
from observability.config_versioning import hash_config, save_config_snapshot

# In project_chat_service.py
async def handle_chat_completion(
    self,
    namespace: str,
    project_id: str,
    request: ChatRequest,
    session_id: str | None = None
):
    # Get project config and hash it
    project = ProjectService.get_project(namespace, project_id)
    config_hash = hash_config(project.config)
    save_config_snapshot(project.config, config_hash, namespace, project_id)

    # Start event
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    logger = EventLogger(
        event_type="inference",
        request_id=request_id,
        namespace=namespace,
        project=project_id,
        config_hash=config_hash
    )

    try:
        # Log each step - just throw dicts at the logger!
        # NO JSON conversion needed, NO formatting required

        logger.log_event("request_received", {
            "endpoint": f"/v1/projects/{namespace}/{project_id}/chat/completions",
            "method": "POST",
            "model": request.model
        })

        # RAG query
        if rag_enabled:
            logger.log_event("rag_query_start", {
                "database": database,
                "query": query,
                "top_k": top_k
            })

            rag_results = await self._perform_rag_query(...)

            # Just pass the data - logger handles serialization
            logger.log_event("rag_retrieval_complete", {
                "chunks_retrieved": len(rag_results),
                "avg_score": sum(r.score for r in rag_results) / len(rag_results),
                "top_chunks": [
                    {
                        "source": r.source,
                        "score": r.score,
                        "content_preview": r.content[:100]
                    }
                    for r in rag_results[:2]  # Top 2 chunks
                ]
            })

        # LLM inference
        logger.log_event("llm_inference_start", {
            "model": model_name,
            "runtime": runtime
        })

        completion = await self._call_llm(...)

        logger.log_event("llm_inference_complete", {
            "tokens": completion.usage.total_tokens if completion.usage else 0,
            "finish_reason": completion.choices[0].finish_reason
        })

        # Response
        response_content = completion.choices[0].message.content
        logger.log_event("response_complete", {
            "content_preview": response_content[:200] if len(response_content) > 200 else response_content,
            "content_length": len(response_content)
        })

        # Single JSON write happens here (efficient!)
        logger.complete_event()
        return completion

    except Exception as e:
        # Log error and write to disk
        logger.fail_event(str(e))
        raise
```

### RAG Processing
```python
from observability.event_logger import EventLogger
from observability.config_versioning import hash_config, save_config_snapshot

# In rag/core/ingest_handler.py
def process_file(
    self,
    file_path: str,
    dataset_name: str,
    database_name: str,
    namespace: str,
    project: str
):
    # Get config and hash
    config = load_config(...)
    config_hash = hash_config(config)
    save_config_snapshot(config, config_hash, namespace, project)

    # Start event
    request_id = f"proc_{uuid.uuid4().hex[:12]}"
    logger = EventLogger(
        event_type="rag_processing",
        request_id=request_id,
        namespace=namespace,
        project=project,
        config_hash=config_hash
    )

    try:
        # Just log plain dicts - NO formatting required!

        logger.log_event("file_parsed", {
            "filename": os.path.basename(file_path),
            "size_bytes": os.path.getsize(file_path),
            "parser": parser_name,
            "mime_type": "application/pdf"
        })

        chunks = self._create_chunks(...)

        logger.log_event("chunks_created", {
            "chunk_count": len(chunks),
            "strategy": strategy_name,
            "avg_chunk_size": sum(len(c.content) for c in chunks) / len(chunks)
        })

        embeddings = self._generate_embeddings(chunks)

        logger.log_event("embeddings_generated", {
            "embedding_count": len(embeddings),
            "embedder": embedder_name,
            "embedding_dimension": len(embeddings[0]) if embeddings else 0
        })

        self._store_chunks(chunks, embeddings, database)

        logger.log_event("chunks_stored", {
            "database": database_name,
            "stored_count": len(chunks),
            "storage_type": "chroma"
        })

        # Single JSON write - efficient!
        logger.complete_event()

    except Exception as e:
        logger.fail_event(str(e))
        raise
```

## Key Architectural Principles

1. **Shared Code First**: EventLogger lives in `observability/` at repo root, importable by ALL components
   - Server: `from observability.event_logger import EventLogger`
   - RAG: `from observability.event_logger import EventLogger`
   - Future runtimes: `from observability.event_logger import EventLogger`
   - Zero cross-dependencies (server ↔ rag)

2. **Dead Simple Interface**:
   - Caller: `logger.log_event("step_name", {"any": "data"})`
   - Logger: Handles threading, JSON, I/O, buffering, timestamps
   - **No formatting required from caller** - just throw dicts at it!

3. **Thread-Safe by Design**: Works seamlessly with parallel operations
   - Multiple threads can call `log_event()` simultaneously
   - Internal locking prevents race conditions
   - Events ordered by actual timestamp (not call order)

4. **Universal Event Logger**: Single shared class used across ALL operations (inference, RAG, model loading, etc.)

5. **Filestore Only**: No SQLite, PostgreSQL, or any database. All data in flat JSON files.

6. **Config Versioning**: Hash-based config snapshots for reproducibility and performance tracking.

7. **Simple Event Structure**: Flat list of timestamped events within a parent event container.

8. **Performance**: Buffered writes (flush on complete/fail), no blocking on inference path.

9. **Minimal Scope (Phase 1)**: Just logging + read API. No metrics, no CLI, no log shipping.

## Implementation Checklist (Phase 1)

### Shared Library (Root Level)
- [ ] Create `observability/` directory at repo root
- [ ] `observability/__init__.py` - Package init
- [ ] `observability/event_logger.py` - Universal EventLogger class
  - [ ] `__init__()` - Initialize event
  - [ ] `log_event()` - Add sub-event
  - [ ] `complete_event()` - Write to disk
  - [ ] `fail_event()` - Write with error status
  - [ ] Buffering logic
  - [ ] File I/O with atomic writes
  - [ ] **Zero dependencies** on server/rag code

- [ ] `observability/config_versioning.py` - Config hashing and storage
  - [ ] `hash_config()` - Generate config hash
  - [ ] `save_config_snapshot()` - Store config by hash
  - [ ] `get_config_by_hash()` - Retrieve config snapshot

### API Endpoints (Server)
- [ ] `server/api/routers/event_logs/router.py` - Event logs API
  - [ ] `GET /event_logs` - List with filtering
  - [ ] `GET /event_logs/{event_id}` - Get single event
  - [ ] Pydantic models for request/response
  - [ ] Import from `observability.event_logger`

### Integration Points
- [ ] `server/services/project_chat_service.py` - Add inference logging
  - [ ] Import: `from observability.event_logger import EventLogger`
  - [ ] Import: `from observability.config_versioning import hash_config, save_config_snapshot`
  - [ ] Start event at request entry
  - [ ] Log sub-events for each step
  - [ ] Complete/fail event

- [ ] `rag/core/ingest_handler.py` - Add RAG processing logging
  - [ ] Import: `from observability.event_logger import EventLogger`
  - [ ] Import: `from observability.config_versioning import hash_config, save_config_snapshot`
  - [ ] Start event at processing start
  - [ ] Log sub-events for parsing, chunking, etc.
  - [ ] Complete/fail event

- [ ] Ensure `observability/` is in Python path for both server and rag
  - [ ] Add to server: `sys.path` or PYTHONPATH
  - [ ] Add to rag: `sys.path` or PYTHONPATH
  - [ ] OR: Make `observability` a proper package with setup.py

### Testing
- [ ] Unit tests for EventLogger
  - [ ] Test event creation
  - [ ] Test sub-event logging
  - [ ] Test complete/fail
  - [ ] Test buffering
  - [ ] Test thread safety

- [ ] Unit tests for config hashing
  - [ ] Test hash determinism
  - [ ] Test config snapshot storage
  - [ ] Test config retrieval

- [ ] Integration tests for event log API
  - [ ] Test GET /event_logs with filters
  - [ ] Test GET /event_logs/{event_id}
  - [ ] Test pagination

- [ ] End-to-end test
  - [ ] Inference request → event log created
  - [ ] RAG processing → event log created
  - [ ] Verify config hash in event
  - [ ] Verify all sub-events captured

### Documentation
- [ ] API documentation for event_logs endpoints
- [ ] Code comments in EventLogger
- [ ] Update main README with observability section
- [ ] Example usage in docstrings

## Repository Structure

```
llamafarm/
  observability/              # NEW: Shared observability library
    __init__.py
    event_logger.py           # Universal EventLogger (no deps on server/rag)
    config_versioning.py      # Config hashing utilities

  server/
    core/
    services/
      project_chat_service.py # Uses: from observability.event_logger import ...
    api/
      routers/
        event_logs/           # NEW: Event logs API
          __init__.py
          router.py
          models.py

  rag/
    core/
      ingest_handler.py       # Uses: from observability.event_logger import ...

  config/                     # Existing shared config
    schema.yaml
    datamodel.py

  cli/                        # Go CLI
    cmd/
      # Future: lf logs commands
```

## Estimated Timeline

**Phase 1**: 3-5 days
- Day 1: Shared library (`observability/`) + EventLogger + config versioning
- Day 2: API endpoints + integration (inference)
- Day 3: Integration (RAG processing) + testing
- Day 4-5: Documentation + polish

## Future Phases (Not in this PR)

- **Phase 2**: Metrics API
  - Read event logs
  - Compute aggregates (latency percentiles, error rates, etc.)
  - Compare configs by performance

- **Phase 3**: CLI commands
  - `lf logs list`
  - `lf logs show <event_id>`
  - `lf metrics inference`

- **Phase 4**: Log shipping plugins
  - S3 export
  - Datadog integration
  - CloudWatch integration

- **Phase 5**: Enhanced file metadata
  - Processing history per file
  - Multi-database associations
  - Tags and user metadata

## Docker Compatibility

### Environment Variable: `LF_DATA_DIR`

All file paths use `LF_DATA_DIR` environment variable for flexibility:

```python
# observability/event_logger.py and observability/config_versioning.py

import os
from pathlib import Path

# Docker-compatible path resolution
data_dir = os.getenv('LF_DATA_DIR', str(Path.home() / ".llamafarm"))
project_dir = Path(data_dir) / "projects" / namespace / project
```

### Docker Compose Configuration

```yaml
# docker-compose.yml

version: '3.8'

services:
  server:
    image: llamafarm-server:latest
    environment:
      - LF_DATA_DIR=/data              # Use /data instead of ~/.llamafarm
    volumes:
      - llamafarm-data:/data           # Persistent volume
    ports:
      - "8000:8000"

  rag:
    image: llamafarm-rag:latest
    environment:
      - LF_DATA_DIR=/data              # Same data directory as server
    volumes:
      - llamafarm-data:/data           # Shared volume (same as server)
    depends_on:
      - server

volumes:
  llamafarm-data:                      # Shared persistent volume
    driver: local
```

### Key Points for Docker

1. **Shared Volume**: Both server and RAG containers mount the same volume
   - Server writes inference events
   - RAG writes processing events
   - Both write to same `${LF_DATA_DIR}/projects/{ns}/{proj}/event_logs/`

2. **Single Mount Point**:
   ```
   /data/projects/{namespace}/{project}/
     event_logs/     # Event logs
     configs/        # Config snapshots
     lf_data/        # Existing data (raw, meta, stores)
   ```

3. **No Home Directory Issues**:
   - ❌ `~/.llamafarm` - Breaks in Docker (different user)
   - ✅ `/data` - Explicit mount point, works everywhere

4. **Consistent Paths**:
   - Local dev: `LF_DATA_DIR=~/.llamafarm` (default)
   - Docker: `LF_DATA_DIR=/data` (via env var)
   - Custom: `LF_DATA_DIR=/mnt/nfs/llamafarm` (networked storage)

### Path Resolution Behavior

```python
# Local development (default)
import os
os.getenv('LF_DATA_DIR', '~/.llamafarm')
# → ~/.llamafarm/projects/default/my-project/event_logs/

# Docker container
os.getenv('LF_DATA_DIR', '~/.llamafarm')  # LF_DATA_DIR=/data
# → /data/projects/default/my-project/event_logs/

# Custom deployment
os.getenv('LF_DATA_DIR', '~/.llamafarm')  # LF_DATA_DIR=/mnt/storage
# → /mnt/storage/projects/default/my-project/event_logs/
```

### Volume Persistence

**Data survives container restarts**:
```bash
# Stop containers
docker-compose down

# Data still exists in volume
docker volume inspect llamafarm-data

# Start containers - data is still there
docker-compose up -d

# Event logs persist across restarts!
```

### Multi-Container Write Safety

Both server and RAG containers write to the same directory:
- ✅ **Safe**: Each writes unique files (different event_id)
- ✅ **Safe**: Config deduplication (atomic writes with `os.replace()`)
- ✅ **Safe**: No shared file handles
- ✅ **Safe**: Filesystem handles concurrency

**File naming prevents conflicts**:
```
evt_inference_20251029_143022_abc123.json      # Server writes
evt_rag_processing_20251029_143525_def456.json # RAG writes
```

Different event types + timestamps + random IDs = no collisions.

### Testing Docker Setup

```bash
# Test local first
export LF_DATA_DIR=~/.llamafarm
python test_event_logging.py
# Check: ~/.llamafarm/projects/default/test/event_logs/

# Test Docker path
export LF_DATA_DIR=/tmp/test-data
python test_event_logging.py
# Check: /tmp/test-data/projects/default/test/event_logs/

# Test in Docker
docker-compose up -d
docker exec llamafarm-server-1 python test_event_logging.py
# Check volume: docker exec llamafarm-server-1 ls /data/projects/default/test/event_logs/
```

## Dependencies

### Required
- No new dependencies! Uses built-in Python libraries:
  - `json` - Event serialization
  - `hashlib` - Config hashing
  - `uuid` - Event ID generation
  - `datetime` - Timestamps
  - `threading` - Thread-safe operations
  - `pathlib` - File path handling
  - `os` - Environment variable access (LF_DATA_DIR)

### Optional (for future phases)
- `httpx` - Log shipping (Phase 4)
- `boto3` - S3/CloudWatch shipping (Phase 4)
- `numpy` - Fast percentile calculations (Phase 2)
