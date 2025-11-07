# LlamaFarm Common Utilities

Shared Python utilities used across all LlamaFarm services (server, rag, runtimes).

## Purpose

This package provides common functionality that needs to be shared across multiple Python services in the LlamaFarm ecosystem. By centralizing these utilities, we avoid code duplication and ensure consistency across services.

## Installation

The package is installed as an editable local dependency in each service's `pyproject.toml`:

```toml
[project]
dependencies = [
    # ... other dependencies
    "llamafarm-common",
]

[tool.uv.sources]
llamafarm-common = { path = "../common", editable = true }
```

## Modules

### `pidfile`

Manages PID files for service discovery and lifecycle management.

**Functions:**

- `write_pid(service_name: str)` - Writes the current process ID to `~/.llamafarm/pids/{service_name}.pid` and registers signal handlers for cleanup
- `cleanup_pid(service_name: str)` - Manually removes a PID file (used in application lifecycle hooks)

**Usage:**

```python
from llamafarm_common.pidfile import write_pid, cleanup_pid

# At service startup
write_pid("server")  # or "rag", "universal-runtime"

# In FastAPI lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    yield
    # Shutdown
    cleanup_pid("server")

# In Celery worker
from celery import signals

@signals.worker_process_shutdown.connect
def cleanup_pid_on_shutdown(**kwargs):
    cleanup_pid("rag")
```

**Features:**

- Automatically registers signal handlers (SIGTERM, SIGINT, SIGHUP) to clean up PID files on termination
- Thread-safe and signal-safe cleanup
- Creates `~/.llamafarm/pids/` directory if it doesn't exist
- Works across Linux, macOS, and Windows

## Development

To make changes to this package:

1. Edit the files in `common/llamafarm_common/`
2. The changes will immediately be available to dependent services (thanks to editable install)
3. No need to rebuild or reinstall

## Adding New Utilities

When adding new shared utilities:

1. Create a new module in `common/llamafarm_common/`
2. Export it from `common/llamafarm_common/__init__.py` if needed
3. Update this README with documentation
4. Ensure all Python services that use it have run `uv sync` to pick up the new code

## Notes

- This package has no external dependencies (only Python stdlib)
- Requires Python >= 3.10 (aligned with other LlamaFarm services)
- Uses `hatchling` as the build backend for consistency with other packages
