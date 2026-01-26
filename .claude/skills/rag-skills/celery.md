# Celery Patterns for RAG Workers

Best practices for Celery task definition, error handling, and worker configuration.

## Architecture Overview

```
Server (FastAPI) -> Celery Broker -> RAG Worker -> Task Execution
                         |                              |
                    filesystem/Redis              ChromaDB/Embedders
```

## Key Files

| File | Purpose |
|------|---------|
| `rag/celery_app.py` | Celery app configuration |
| `rag/main.py` | Worker entry point |
| `rag/tasks/*.py` | Task definitions |

## Celery App Configuration

Location: `rag/celery_app.py`

```python
from celery import Celery, signals

app = Celery("LlamaFarm-RAG-Worker")

app.conf.update(
    # Broker configuration
    broker_url=settings.CELERY_BROKER_URL or "filesystem://",
    result_backend=settings.CELERY_RESULT_BACKEND,

    # Serialization
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Task routing - CRITICAL: only handle rag.* tasks
    task_routes={
        "rag.*": {"queue": "rag"},
    },

    # Import task modules
    imports=(
        "tasks.search_tasks",
        "tasks.ingest_tasks",
        "tasks.query_tasks",
        "tasks.health_tasks",
        "tasks.stats_tasks",
    ),
)

# Prevent Celery from overriding logging
@signals.setup_logging.connect
def setup_celery_logging(**kwargs):
    pass
```

## Task Definition Pattern

### Base Task Class

```python
from celery import Task

class IngestTask(Task):
    """Base task with error logging."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(
            "RAG ingest task failed",
            extra={
                "task_id": task_id,
                "task_name": self.name,
                "error": str(exc),
                "task_args": args,
                "task_kwargs": kwargs,
            },
        )
```

### Task Definition

```python
@app.task(bind=True, base=IngestTask, name="rag.ingest_file")
def ingest_file_task(
    self,
    project_dir: str,
    strategy_name: str,
    database_name: str,
    source_path: str,
    filename: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """
    Ingest a file using RAG system.

    Args:
        project_dir: Project directory path
        strategy_name: Data processing strategy
        database_name: Target database
        source_path: File path to ingest
        filename: Optional display name

    Returns:
        Tuple of (success, details_dict)
    """
    logger.info(
        "Starting RAG file ingestion",
        extra={
            "task_id": self.request.id,
            "project_dir": project_dir,
            "source_path": source_path,
        },
    )

    try:
        # Task implementation
        handler = IngestHandler(...)
        result = handler.ingest_file(...)
        return True, result
    except Exception as e:
        logger.error(
            "Ingestion failed",
            extra={"task_id": self.request.id, "error": str(e)},
        )
        return False, {"error": str(e)}
```

## Worker Configuration

### Thread Pool (Required for ChromaDB)

```python
def run_worker():
    # MUST use thread pool to avoid SQLite locking with ChromaDB
    worker_args = ["worker", "-Q", "rag", "--pool=threads"]

    concurrency = os.getenv("LF_CELERY_CONCURRENCY")
    if concurrency:
        worker_args.extend(["--concurrency", concurrency])

    app.worker_main(argv=worker_args)
```

### macOS Fork Safety

```python
import multiprocessing
import sys

if sys.platform == "darwin":
    multiprocessing.set_start_method("spawn", force=True)
```

## Code Review Checklist

### 1. Task Naming Convention

**Description**: Tasks must follow the `rag.*` naming pattern for routing.

**Search Pattern**:
```bash
grep -rn '@app.task.*name="rag\.' rag/tasks/
```

**Pass Criteria**:
- All tasks named `rag.<action>`
- Names are descriptive
- Consistent naming pattern

**Fail Criteria**:
- Missing name parameter
- Name doesn't start with `rag.`
- Inconsistent naming

**Severity**: Critical

**Recommendation**: Always use `name="rag.<module>.<action>"` format.

---

### 2. Task Binding

**Description**: Tasks should be bound to access request context.

**Search Pattern**:
```bash
grep -rn '@app.task.*bind=True' rag/tasks/
```

**Pass Criteria**:
- bind=True in decorator
- self.request.id used for logging
- Task has access to context

**Fail Criteria**:
- bind=False or missing
- No task ID in logs
- Missing request context

**Severity**: Medium

**Recommendation**: Use `bind=True` for all tasks that need logging.

---

### 3. Base Task Class Usage

**Description**: Tasks should use base class for consistent error handling.

**Search Pattern**:
```bash
grep -rn 'base=.*Task' rag/tasks/
```

**Pass Criteria**:
- Custom base task defined
- on_failure logs errors
- Base task used in decorator

**Fail Criteria**:
- No base task
- Missing on_failure handler
- Inconsistent base usage

**Severity**: Medium

**Recommendation**: Define base task with on_failure logging.

---

### 4. Error Logging Context

**Description**: Task errors must include sufficient context for debugging.

**Search Pattern**:
```bash
grep -rn 'logger.error.*extra=\|on_failure' rag/tasks/
```

**Pass Criteria**:
- task_id included in logs
- Error message captured
- Input parameters logged

**Fail Criteria**:
- Missing task_id
- Only exception logged
- No input context

**Severity**: High

**Recommendation**: Include task_id, error, and key parameters in error logs.

---

### 5. JSON Serialization Compatibility

**Description**: Task arguments and returns must be JSON-serializable.

**Search Pattern**:
```bash
grep -rn 'def .*_task\(' rag/tasks/ -A 20
```

**Pass Criteria**:
- Arguments are primitive types
- Return values are dicts/lists
- No Path or custom objects

**Fail Criteria**:
- Path objects as arguments
- Custom class returns
- Datetime without conversion

**Severity**: High

**Recommendation**: Convert Path to str, datetime to ISO string.

---

### 6. Thread Pool Requirement

**Description**: Worker must use thread pool for ChromaDB compatibility.

**Search Pattern**:
```bash
grep -rn "pool=threads\|--pool" rag/
```

**Pass Criteria**:
- --pool=threads in worker args
- No prefork pool
- Documentation mentions requirement

**Fail Criteria**:
- prefork or default pool
- Missing pool specification
- No documentation

**Severity**: Critical

**Recommendation**: Always use `--pool=threads` for RAG workers.

---

### 7. Queue Routing

**Description**: RAG tasks must route to the `rag` queue.

**Search Pattern**:
```bash
grep -rn 'task_routes\|"-Q".*rag' rag/
```

**Pass Criteria**:
- task_routes configured for rag.*
- Worker consumes only rag queue
- No celery queue consumption

**Fail Criteria**:
- Missing task_routes
- Worker on default queue
- Mixed queue consumption

**Severity**: High

**Recommendation**: Configure task_routes and use `-Q rag` for worker.

## Anti-Patterns

### Missing Task Name

```python
# BAD: Auto-generated name
@app.task
def ingest_file(project_dir, source_path):
    ...

# GOOD: Explicit name with rag prefix
@app.task(bind=True, name="rag.ingest_file")
def ingest_file_task(self, project_dir, source_path):
    ...
```

### Non-Serializable Arguments

```python
# BAD: Path object
@app.task(name="rag.process")
def process_task(file_path: Path):
    ...

# GOOD: String path
@app.task(name="rag.process")
def process_task(file_path: str):
    path = Path(file_path)
    ...
```

### Silent Error Handling

```python
# BAD: Swallowed exception
@app.task(name="rag.search")
def search_task(query):
    try:
        return search(query)
    except Exception:
        return []  # Silent failure!

# GOOD: Logged and re-raised
@app.task(bind=True, base=SearchTask, name="rag.search")
def search_task(self, query):
    try:
        return search(query)
    except Exception as e:
        logger.error(
            "Search failed",
            extra={"task_id": self.request.id, "error": str(e)},
        )
        raise  # Let Celery handle retry/failure
```

### Wrong Pool Type

```python
# BAD: Prefork causes ChromaDB SQLite errors
app.worker_main(argv=["worker", "-Q", "rag", "--pool=prefork"])

# GOOD: Thread pool for ChromaDB
app.worker_main(argv=["worker", "-Q", "rag", "--pool=threads"])
```

## Task Return Patterns

### Success with Details

```python
return {
    "status": "success",
    "filename": filename,
    "document_count": len(documents),
    "stored_count": stored,
    "skipped_count": skipped,
}
```

### Error with Context

```python
return {
    "status": "error",
    "message": str(e),
    "filename": filename,
    "reason": "embedder_unavailable",
}
```

### Tuple Pattern (Boolean + Details)

```python
# For tasks that need simple success/fail check
return success, details_dict
```
