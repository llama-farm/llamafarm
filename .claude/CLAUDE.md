# LlamaFarm – Comprehensive Guide for LLM Assistants

This document summarizes everything an automated collaborator needs to get LlamaFarm running, extend it, and keep configuration accurate.

## 1. Prerequisites & Fast Start
1. **Install Docker** – required for auto-starting the API + RAG worker.
2. **Install Ollama** – current default runtime; download from https://ollama.com/download.
3. **Install the CLI (lf)**
   ```bash
   # macOS / Linux
   curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
   ```
   - Windows: download `lf.exe` from the latest release and add it to PATH.
4. **Adjust Ollama context window** – open Ollama → Settings → Advanced → set context window size (e.g., 100k tokens) to match production expectations.

### Optional: Lemonade Runtime Setup
Lemonade provides local LLM inference with NPU/GPU acceleration. It's optional but provides additional model options.

**Installation:**
```bash
# Install via pip (automatic via LlamaFarm on first use)
uv pip install lemonade-sdk
```

**Key Details:**
- **Package name:** `lemonade-sdk` (NOT `lemonade-server-dev`)
- **Command:** `lemonade-server-dev` (when installed via PyPI)
- **Base URL format:** `http://localhost:{port}/api/v1` (note the `/api/v1` suffix)
- **Default port:** 8000 (configure different port via `lemonade.port` in config)
- **Backends:** `llamacpp` (recommended), `transformers`, `onnx`
- **Auto-starts:** via `nx start lemonade` when configured

**Configuration in llamafarm.yaml:**
```yaml
runtime:
  models:
    lemon:
      provider: lemonade
      model: Qwen3-0.6B-GGUF  # Any GGUF model from Lemonade's library
      huggingface_token: hf_xxxxx  # Optional, for gated models
      lemonade:
        backend: llamacpp      # llamacpp | transformers | onnx
        port: 11534           # Different from default to avoid conflicts
        context_size: 32768   # Context window size
```

**Important Notes:**
- Lemonade SDK auto-installs on first `nx start lemonade` if not present
- Hardware detection is automatic (Metal on macOS, CUDA/Vulkan on Linux, CPU fallback)
- Each Lemonade model instance requires its own port
- Models are pulled from HuggingFace on first use

### First Run
```bash
lf init my-project        # generates llamafarm.yaml via server template
lf start                  # starts FastAPI + Celery workers in Docker, opens chat TUI
```
- Use `Ctrl+C` to exit the chat UI.
- Once running, the CLI provides:
  - Dataset lifecycle: `lf datasets create|upload|process`
  - Retrieval: `lf rag query --database ...`
  - Health checks: `lf rag health`, `lf rag stats`
  - One-off chat: `lf chat [...flags]`

## 2. Repository Layout (top-level):
- `README.md` – CLI-first quickstart, extensibility highlights, testing commands.
- `config/` – schema + generated types for `llamafarm.yaml`.
- `server/` – FastAPI app, Celery task definitions.
- `rag/` – ingestion/parsing utilities, Celery worker entry.
- `cli/` – Go-based CLI (`lf`) built with Cobra.
- `docs/website/` – Docusaurus doc site (`nx build docs`).
- `.claude/` – this helper file and any future LLM-specific directions.

## 3. Key Documentation Pages (relative paths):
- Quickstart onboarding: `docs/website/docs/quickstart/index.md`
- CLI reference: `docs/website/docs/cli/index.md`
- Configuration guide: `docs/website/docs/configuration/index.md`
- RAG guide: `docs/website/docs/rag/index.md`
- Models/runtime: `docs/website/docs/models/index.md`
- Extensibility overview: `docs/website/docs/extending/index.md`
- Examples: `docs/website/docs/examples/index.md`
- Troubleshooting: `docs/website/docs/troubleshooting/index.md`

## 4. Configuration (`llamafarm.yaml`)
Schema files:
- `config/schema.yaml` – top-level project schema.
- `rag/schema.yaml` – inlined for RAG-specific fields (databases, strategies, parsers, extractors).

Required sections:
- `version` – currently `v1`.
- `name`, `namespace` – identify the project/tenant.
- `runtime` – **supports two formats:**
  - **Multi-model (recommended):** `runtime.models` dict + `runtime.default_model`
  - **Legacy (deprecated):** `runtime.provider`, `runtime.model`, `runtime.base_url`
- `prompts` – list of `{role, content}` messages.
- `rag` – `databases` + `data_processing_strategies` definitions.
- `datasets` (optional) – keep dataset metadata in sync.

### Multi-Model Configuration (NEW)
The new multi-model format allows switching between different models via API or CLI:

```yaml
runtime:
  default_model: fast  # Which model to use by default

  models:
    fast:
      description: "Fast Ollama model for quick responses"
      provider: ollama
      model: gemma3:1b
      base_url: http://localhost:11434/v1  # Note: /v1 suffix required for Ollama
      prompt_format: unstructured

    powerful:
      description: "More capable model"
      provider: ollama
      model: qwen3:8b
      base_url: http://localhost:11434/v1

    lemon:
      description: "Lemonade runtime with local GGUF model"
      provider: lemonade
      model: Qwen3-0.6B-GGUF
      lemonade:
        backend: llamacpp
        port: 11534
        context_size: 32768
```

**Using multi-model:**
- CLI: `lf chat --model powerful "your question"`
- CLI: `lf models list` (shows all available models)
- API: `POST /v1/projects/{ns}/{id}/chat/completions` with `{"model": "powerful", ...}`
- API: `GET /v1/projects/{ns}/{id}/models` (lists all models)

**Backward Compatibility:**
- Legacy single-model configs are auto-converted to multi-model format internally
- No breaking changes for existing configurations
- ModelService.normalize_config() handles conversion automatically

Reference docs: `docs/website/docs/configuration/index.md` and `docs/website/docs/configuration/example-configs.md`.

## 5. Editing Schemas & Types
When you change `config/schema.yaml` or `rag/schema.yaml`:
```bash
cd config
uv run python generate_types.py
```
Outputs:
- `config/datamodel.py` (Pydantic models)
- `config/config_types.go` (Go structs for CLI)

Update the CLI/server code to handle new fields, then adjust docs accordingly.

## 6. Extending Components
### Runtime Providers
1. Add provider enum value in `config/schema.yaml` (`runtime.provider`).
2. Regenerate types (`config/generate_types.py`).
3. Update runtime selection logic:
   - Server: `server/services/runtime_service.py` (or relevant module).
   - CLI: wherever runtime resolution occurs (e.g., `cli/cmd/config/types.go` consumers).
4. Document usage in `docs/website/docs/models/index.md`.

### RAG Stores / Parsers / Extractors
1. Implement the component inside `rag/` (e.g., new store class, parser implementation).
2. Register it with the ingestion pipeline.
3. Update `rag/schema.yaml` definitions:
   - `databaseDefinition.type`
   - Parser/extractor enums & config schemas.
4. Regenerate types and update docs (`docs/website/docs/rag/index.md`).

### CLI Commands
- Add Cobra command under `cli/cmd/`, hook it via `rootCmd.AddCommand(...)`.
- Follow patterns for config resolution (`config.GetServerConfig`) and server auto-start (`ensureServerAvailable`).
- Write tests (`*_test.go`) and update the CLI reference documentation.

## 7. Common CLI Commands
```bash
# Dataset lifecycle
lf datasets add research -s universal_processor -b main_db  # Create dataset
lf datasets ingest research ./examples/fda_rag/files/*.pdf  # Add files (supports glob patterns)
lf datasets ingest research ./examples/fda_rag/files/       # Add entire directory
lf datasets process research                                 # Process into vector database
lf datasets list                                             # List all datasets
lf datasets remove research                                  # Delete a dataset

# Retrieval & health
lf rag query --database main_db "Which letters mention clinical trials?"
lf rag query --database main_db --top-k 5 --score-threshold 0.7 "Your query"
lf rag health
lf rag stats

# Chat
lf start                                      # Opens interactive TUI chat
lf chat "Summarize neural scaling laws"      # One-off chat with RAG (if configured)
lf chat --database main_db "Query with specific DB"
lf chat --no-rag "Explain attention mechanisms"  # LLM only, no RAG
lf chat --curl "What models are configured?"     # Show curl equivalent
```

## 8. Testing & Validation
- Server & RAG Python tests: `uv run --group test python -m pytest`
- RAG-specific tests: `uv run pytest tests/`
- Specific test file: `uv run pytest tests/test_greeting_logic.py -v`
- CLI: `go test ./...`
- Docs build: `nx build docs` *(clear `.nx` cache if you hit sqlite disk I/O errors)*

**Test Best Practices:**
- When testing settings/environment variables, use `@patch` to mock the settings object directly
- Never use `monkeypatch.setenv()` for settings that are loaded at import time
- Example: `@patch("agents.project_chat_orchestrator.settings")` then set `mock_settings.lf_dev_mode_greeting_enabled = False`

## 9. Environment Variables & Settings

**Settings Management:**
- All environment variables should be managed through `server/core/settings.py` using `pydantic_settings.BaseSettings`
- **NEVER** access `os.environ` directly in application code
- Settings are loaded from `.env` file at import time
- The `settings` singleton is available via `from core.settings import settings`

**Available Settings:**
- `lf_dev_mode_docs_enabled: bool = True` - Enable/disable dev mode documentation context injection
- `lf_dev_mode_greeting_enabled: bool = True` - Enable/disable greeting messages in project_seed
- See `server/core/settings.py` for complete list of available settings

**Adding New Settings:**
1. Add the setting to the `Settings` class in `server/core/settings.py`
2. Provide a default value
3. Use type hints for proper validation
4. Import and use via `from core.settings import settings`
5. Update tests to mock the settings object if needed

## 10. Upgrade / Development Notes
- Always run research/plan steps from `.agents/commands/` before making changes.
- Keep docs in sync with behaviour—update Docusaurus pages when workflows/schema change.
- Never commit secrets; use local `.env` and update `.env.example` for new variables.
- When adding user-facing features, ensure README + docs + sample configs all reflect the changes.
- **Code Review Standards:**
  - Use `settings` from `core.settings` instead of `os.environ` directly
  - Avoid dynamic imports or module lookup hacks (e.g., `sys.modules.get("os", __import__("os"))`)
  - Follow centralized configuration management patterns

## 11. Additional Resources
- Project structure overview: `AGENTS.md`
- Contribution process: `CONTRIBUTING.md`
- Credits: `docs/CREDITS.md`
- Examples: `examples/` directory with ready-made configs (`fda_rag`, `gov_rag`).

Use this guide as the master reference when assisting with LlamaFarm tasks.
