# LlamaFarm Developer Guide

## IMPORTANT: Commit & PR Rules
- **DO NOT attribute Claude** in commits or PRs (no co-author, no mentions)
- Use [Conventional Commits](https://www.conventionalcommits.org/): `type(component): description`
- Example: `fix(cli): cmd 'lf project list' doesn't honor cwd flag`

## Architecture Overview

```
llamafarm/
├── cli/           # Go CLI (Cobra) - `lf` command
├── server/        # FastAPI + Celery - main API server
├── rag/           # RAG pipeline - parsers, embedders, stores
├── runtimes/      # Model runtimes (universal, lemonade)
├── config/        # Schema definitions + type generation
├── designer/      # React frontend (Nx workspace)
└── docs/          # Docusaurus documentation
```

**Data Flow:**
```
CLI/API → FastAPI Server → Runtime (Universal preferred, Ollama, Lemonade)
                        → RAG Pipeline → Vector Store (Chroma/Qdrant)
                        → Celery Workers (async processing)
```

**Universal Runtime** (preferred):
- Default runtime for all model inference
- Supports GGUF models, HuggingFace transformers, embeddings, classifiers
- Located in `runtimes/universal/`
- Auto-manages model loading/unloading

## Development Setup

```bash
# Python (server, rag, runtimes)
uv sync                    # Install dependencies
uv run pytest              # Run tests

# Go (CLI)
go build ./...             # Build
go test ./...              # Test

# Frontend (designer)
nx serve designer          # Dev server
nx build designer          # Production build

#Universal Runtime
nx start universal-runtime #starts universal runtime

# Full stack
nx start server            # Start FastAPI server
nx start rag               # Start RAG worker
nx start universal-runtime
```


## Coding Standards

### Python (Ruff enforced)
- **Formatter/Linter:** Ruff (config in `ruff.toml`)
- **Package manager:** `uv` (NOT pip)
- **Pre-commit hooks:** `uvx pre-commit install`
- **Type hints:** Required for all functions
- **Settings:** Use `from core.settings import settings` (never `os.environ`)

```bash
# Run linter manually
uvx ruff check .
uvx ruff format .
```

### Go
- Standard `go fmt` and `go vet`
- CLI commands in `cli/cmd/`

### TypeScript/React
- ESLint + Prettier
- Components in `designer/src/components/`

## Schema & Type Generation

When modifying `config/schema.yaml` or `rag/schema.yaml`:
```bash
cd config && uv run python generate_types.py
```
Outputs: `config/datamodel.py` (Pydantic), `config/config_types.go` (Go)

## Testing

```bash
# Python
uv run pytest                              # All tests
uv run pytest tests/test_foo.py -v         # Specific file
uv run pytest -k "test_name"               # By name

# Go
go test ./...
go test ./cmd/... -v

# Mock settings in tests
@patch("module.settings")
def test_foo(mock_settings):
    mock_settings.some_setting = False
```

## Key Files

| Purpose | Location |
|---------|----------|
| API routes | `server/api/routers/` |
| Services | `server/services/` |
| Agents | `server/agents/` |
| RAG parsers | `rag/components/parsers/` |
| RAG stores | `rag/components/stores/` |
| CLI commands | `cli/cmd/` |
| Config schema | `config/schema.yaml` |
| RAG schema | `rag/schema.yaml` |

## Common Tasks

### Add new API endpoint
1. Create router in `server/api/routers/`
2. Register in `server/api/main.py`
3. Add types to router file

### Add new parser/extractor
1. Implement in `rag/components/parsers/` or `extractors/`
2. Update `rag/schema.yaml`
3. Regenerate types

### Add CLI command
1. Add command in `cli/cmd/`
2. Register via `rootCmd.AddCommand()`
3. Update docs

## Don'ts
- Don't use `os.environ` directly (use settings)
- Don't skip pre-commit hooks
- Don't commit `.env` files or secrets
- Don't create unnecessary abstractions
- Don't add features beyond what's requested
