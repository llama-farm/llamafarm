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

### `model_utils`

Common utilities for parsing and selecting GGUF model files with quantization variants.

**Functions:**

- `parse_model_with_quantization(model_name: str) -> tuple[str, str | None]` - Parse model names with optional quantization suffix (e.g., `"model:Q4_K_M"`)
- `parse_quantization_from_filename(filename: str) -> str | None` - Extract quantization type from GGUF filenames
- `select_gguf_file(gguf_files: list[str], preferred_quantization: str | None = None) -> str | None` - Intelligently select the best GGUF file based on quantization preference

**Constants:**

- `GGUF_QUANTIZATION_PREFERENCE_ORDER` - Default quantization preference order (Q4_K_M, Q4_K, Q5_K_M, Q5_K, Q8_0, etc.)

**Usage:**

```python
from llamafarm_common import (
    parse_model_with_quantization,
    parse_quantization_from_filename,
    select_gguf_file,
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
)

# Parse model name with quantization
model_id, quant = parse_model_with_quantization("unsloth/Qwen3-4B-GGUF:Q8_0")
# Returns: ("unsloth/Qwen3-4B-GGUF", "Q8_0")

# Parse quantization from filename
quant = parse_quantization_from_filename("model.Q4_K_M.gguf")
# Returns: "Q4_K_M"

# Select best GGUF file
files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.F16.gguf"]
selected = select_gguf_file(files)  # Returns "model.Q4_K_M.gguf" (default)
selected = select_gguf_file(files, "Q8_0")  # Returns "model.Q8_0.gguf"
```

**Features:**

- Case-insensitive quantization parsing
- Supports all common GGUF quantization types (Q2_K through F32)
- Intelligent fallback when preferred quantization is not available
- Used by both server and runtime for consistent model selection

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

## Testing

Run tests with:

```bash
cd common
uv run pytest tests/
```

## Notes

- This package has no external dependencies in production (only Python stdlib)
- Requires Python >= 3.10 (aligned with other LlamaFarm services)
- Uses `hatchling` as the build backend for consistency with other packages
- Dev dependencies include `pytest` for testing
