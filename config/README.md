# LlamaFarm Configuration Module

Utilities for loading, validating, and working with `llamafarm.yaml`.

## What It Does
- Loads project configuration from `llamafarm.yaml` (YAML only).
- Validates against `config/schema.yaml` (which in turn references `rag/schema.yaml`).
- Provides generated Pydantic/Go types for editor autocomplete and runtime safety.
- Offers helpers used by the CLI and server to resolve namespaces, projects, and runtime settings.

## Key Files
| Path | Purpose |
| ---- | ------- |
| `schema.yaml` | JSON Schema for the project config (`version`, `runtime`, `prompts`, `rag`, `datasets`). |
| `datamodel.py` | Auto-generated Pydantic models consumed by Python components. |
| `config_types.go` | Auto-generated Go structs consumed by the CLI. |
| `generate-types.sh` | Script that regenerates both datamodels from the schema. |

## Regenerating Types
Run this whenever you change `schema.yaml` or update anything referenced by it (e.g., `rag/schema.yaml`).

```bash
cd config
./generate-types.sh
```

The script installs `datamodel-code-generator` if needed and updates `datamodel.py` / `config_types.go` in place.

## Loading Configs in Python
```python
from config import load_config
from config.datamodel import LlamaFarmConfig

cfg: LlamaFarmConfig = load_config()              # Finds llamafarm.yaml automatically
print(cfg.runtime.provider)                       # "openai" or "ollama"
print(cfg.rag.databases[0].default_retrieval_strategy)
```

## Schema Highlights
- `version`: Must be `v1`.
- `runtime`: Requires `provider`, `model`, and (for non-default hosts) `base_url`/`api_key`. Supports optional `instructor_mode` and `model_api_parameters`.
- `prompts`: An array of `{role, content}` entries injected before the conversation.
- `rag`: Inlines the full RAG schema (databases, embedding strategies, retrieval strategies, data processing strategies).
- `datasets`: Optional list used to keep dataset metadata in sync with the server.

Refer to the [Configuration Guide](../docs/website/docs/configuration/index.md) for user-facing documentation and field descriptions.

## Tests
```bash
uv run --group test python -m pytest config/tests
```

These tests ensure schema validation stays compatible with generated types and server expectations.
