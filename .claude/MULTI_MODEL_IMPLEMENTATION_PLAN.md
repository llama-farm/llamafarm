# Multi-Model Implementation Plan

**Status:** Planning Phase
**Date:** 2025-10-02
**Goal:** Enable multiple model configurations in `llamafarm.yaml` with OpenAI-compatible API model selection

---

## 🎯 Overview

Enable users to configure multiple models in their `llamafarm.yaml` and switch between them using:
- OpenAI API's `model` parameter (e.g., `{"model": "fast-model", ...}`)
- CLI flags (e.g., `lf chat --model fast-model`)
- Default model fallback for backward compatibility

### Design Principles

1. **Backward Compatible**: Existing single-model configs work without changes
2. **OpenAI-Compatible**: Use standard `model` parameter for selection
3. **Extensible**: Designed for future fallback/routing features
4. **Provider-Agnostic**: Works with Ollama, Lemonade, OpenAI, etc.
5. **Minimal Changes**: Leverage existing provider registry pattern

---

## 📋 Configuration Design

### Current Schema (Backward Compatible)

```yaml
version: v1
name: my-project
namespace: default

runtime:
  provider: ollama
  model: gemma3:1b
  base_url: http://localhost:11434/v1
```

### New Multi-Model Schema

```yaml
version: v1
name: my-project
namespace: default

runtime:
  default_model: fast-model  # NEW: Default model to use

  models:  # NEW: Named model configurations
    fast-model:
      description: "Fast local model for quick responses"
      provider: ollama
      model: gemma3:1b
      base_url: http://localhost:11434/v1
      prompt_format: unstructured

    smart-model:
      description: "Larger model for complex reasoning"
      provider: ollama
      model: qwen2.5:7b
      base_url: http://localhost:11434/v1
      prompt_format: structured
      instructor_mode: md_json

    tiny-model:
      description: "Tiny Lemonade model for testing"
      provider: lemonade
      model: Qwen3-0.6B-GGUF
      lemonade:
        backend: llamacpp
        context_size: 32768

prompts:
  - role: system
    content: "You are a helpful assistant."
```

### Migration Strategy: Support Both Formats

**Option A: User has old single-model config**
```yaml
runtime:
  provider: ollama
  model: gemma3:1b
```
→ Auto-creates a default model named "default" internally

**Option B: User has new multi-model config**
```yaml
runtime:
  default_model: fast-model
  models:
    fast-model: {...}
    smart-model: {...}
```
→ Uses named models directly

---

## 🏗️ Schema Changes

### File: `config/schema.yaml`

```yaml
runtime:
  type: object
  description: Runtime configuration for LLM inference
  properties:
    # NEW: Multi-model configuration
    default_model:
      type: ["string", "null"]
      description: "Name of the default model to use (references models.{name})"

    models:
      type: ["object", "null"]
      description: "Named model configurations for multi-model support"
      additionalProperties:
        type: object
        required: [provider, model]
        properties:
          description:
            type: ["string", "null"]
            description: "Human-readable description of this model configuration"
          provider:
            type: string
            enum: [openai, ollama, lemonade]
            description: "Runtime provider"
          model:
            type: string
            description: "Model name or ID"
          base_url:
            type: ["string", "null"]
            description: "Base URL for the provider"
          api_key:
            type: ["string", "null"]
            description: "API key for the provider"
          huggingface_token:
            type: ["string", "null"]
            description: "Hugging Face API token"
          instructor_mode:
            type: ["string", "null"]
            description: "Instructor mode for structured output"
          prompt_format:
            type: string
            enum: [structured, unstructured]
            default: unstructured
          model_api_parameters:
            type: ["object", "null"]
            additionalProperties: true
          lemonade:
            type: ["object", "null"]
            properties:
              backend:
                type: string
                enum: [onnx, llamacpp, transformers]
              port:
                type: integer
              context_size:
                type: integer
              model_path:
                type: ["string", "null"]

    # LEGACY: Keep old single-model fields for backward compatibility
    provider:
      type: ["string", "null"]
      enum: [openai, ollama, lemonade]
      description: "LEGACY: Use models.{name}.provider instead"
    model:
      type: ["string", "null"]
      description: "LEGACY: Use models.{name}.model instead"
    base_url:
      type: ["string", "null"]
      description: "LEGACY: Use models.{name}.base_url instead"
    # ... other legacy fields
```

**Key Design Decision**: Keep legacy fields as optional to support old configs, but prioritize `models` if present.

---

## 🔧 Code Changes

### 1. Configuration Loading (`server/services/project_service.py`)

**Add model resolution logic:**

```python
def load_config(namespace: str, project_id: str) -> LlamaFarmConfig:
    """Load and normalize config to support both legacy and multi-model formats."""
    config = _load_yaml_config(namespace, project_id)

    # Normalize: Convert legacy single-model to multi-model format
    if config.runtime.models is None or len(config.runtime.models) == 0:
        # Legacy format: runtime.provider + runtime.model
        if config.runtime.provider and config.runtime.model:
            config.runtime.models = {
                "default": ModelConfig(
                    provider=config.runtime.provider,
                    model=config.runtime.model,
                    base_url=config.runtime.base_url,
                    api_key=config.runtime.api_key,
                    # ... copy all legacy fields
                )
            }
            config.runtime.default_model = "default"

    # Set default_model if not specified
    if not config.runtime.default_model and config.runtime.models:
        config.runtime.default_model = list(config.runtime.models.keys())[0]

    return config
```

### 2. Model Selection Service (NEW)

**File: `server/services/model_service.py`**

```python
from config.datamodel import LlamaFarmConfig, ModelConfig

class ModelService:
    """Service for resolving and managing model configurations."""

    @staticmethod
    def get_model_config(
        project_config: LlamaFarmConfig,
        model_name: str | None = None
    ) -> ModelConfig:
        """Get model configuration by name, falling back to default.

        Args:
            project_config: Project configuration
            model_name: Optional model name to select (from API request)

        Returns:
            ModelConfig for the selected model

        Raises:
            ValueError: If model_name doesn't exist
        """
        # Use requested model or fall back to default
        selected_model = model_name or project_config.runtime.default_model

        if not selected_model:
            raise ValueError("No model specified and no default_model configured")

        if selected_model not in project_config.runtime.models:
            available = ", ".join(project_config.runtime.models.keys())
            raise ValueError(
                f"Model '{selected_model}' not found. Available: {available}"
            )

        return project_config.runtime.models[selected_model]

    @staticmethod
    def list_models(project_config: LlamaFarmConfig) -> list[dict]:
        """List all available models with metadata."""
        models = []
        for name, config in project_config.runtime.models.items():
            models.append({
                "id": name,
                "description": config.description or "",
                "provider": config.provider.value,
                "model": config.model,
                "is_default": name == project_config.runtime.default_model
            })
        return models
```

### 3. Update Chat Orchestrator

**File: `server/agents/project_chat_orchestrator.py`**

```python
class ProjectChatOrchestratorAgent(LFAgent):
    def __init__(
        self,
        project_config: LlamaFarmConfig,
        project_dir: str,
        model_name: str | None = None,  # NEW: Model selection
    ):
        # NEW: Resolve model configuration
        from services.model_service import ModelService
        self.model_config = ModelService.get_model_config(project_config, model_name)

        # Use model_config instead of project_config.runtime
        client = _get_client_for_model(self.model_config)

        # ... rest of init
```

**Update `_get_client()` to use ModelConfig:**

```python
def _get_client_for_model(model_config: ModelConfig) -> AsyncOpenAI:
    """Get client for a specific model configuration."""
    provider = get_provider(model_config.provider)

    # Create a temporary LlamaFarmConfig-like object for the provider
    # (Providers expect full config, but we only pass model-specific config)
    temp_config = type('obj', (object,), {
        'runtime': model_config
    })()

    return provider.get_client(temp_config)
```

### 4. Update API Router

**File: `server/api/routers/projects/projects.py`**

```python
async def chat(
    request: ChatRequest,
    namespace: str,
    project_id: str,
    response: Response,
    session_id: str | None = Header(None, alias="X-Session-ID"),
    x_no_session: str | None = Header(None, alias="X-No-Session"),
):
    """Send a message to the chat agent."""
    project_config = ProjectService.load_config(namespace, project_id)

    # NEW: Extract model from request (OpenAI-compatible)
    model_name = request.model if hasattr(request, 'model') else None

    # Create agent with model selection
    agent = ProjectChatOrchestratorAgentFactory.create_agent(
        project_config,
        project_dir=project_dir,
        model_name=model_name  # NEW
    )

    # ... rest of handler
```

**Update ChatRequest model:**

```python
class ChatRequest(BaseModel):
    message: str
    model: str | None = None  # NEW: OpenAI-compatible model selection
```

### 5. Add Models List Endpoint (NEW)

**File: `server/api/routers/projects/projects.py`**

```python
@router.get("/{namespace}/{project_id}/models")
async def list_models(namespace: str, project_id: str):
    """List available models for this project (OpenAI-compatible)."""
    project_config = ProjectService.load_config(namespace, project_id)
    models = ModelService.list_models(project_config)

    # OpenAI-compatible response format
    return {
        "object": "list",
        "data": [
            {
                "id": m["id"],
                "object": "model",
                "created": int(time.time()),
                "owned_by": m["provider"],
                "description": m["description"],
                "is_default": m["is_default"]
            }
            for m in models
        ]
    }
```

### 6. CLI Updates

**File: `cli/cmd/chat.go`** (or equivalent)

Add `--model` flag:

```go
var chatModelName string

chatCmd.Flags().StringVar(&chatModelName, "model", "", "Model to use (overrides default)")
```

Update API request:

```go
requestBody := map[string]interface{}{
    "message": userMessage,
}

if chatModelName != "" {
    requestBody["model"] = chatModelName
}
```

---

## 🔄 Migration Path

### Phase 1: Backward Compatibility (Week 1)

1. Update schema with `models` and `default_model` (optional)
2. Add config normalization in `ProjectService.load_config()`
3. Update tests to verify legacy configs still work
4. **No breaking changes**

### Phase 2: Model Service (Week 1-2)

1. Create `ModelService` for model resolution
2. Update `ProjectChatOrchestratorAgent` to accept `model_name`
3. Update factory to pass model selection
4. Add unit tests for model selection logic

### Phase 3: API & CLI (Week 2)

1. Add `model` field to `ChatRequest`
2. Add `GET /models` endpoint
3. Add CLI `--model` flag
4. Update API documentation

### Phase 4: Documentation & Examples (Week 2-3)

1. Update configuration docs with multi-model examples
2. Create migration guide for existing users
3. Add examples for common use cases
4. Update CLI help text

---

## 🧪 Testing Strategy

### Unit Tests

```python
# test_model_service.py
def test_legacy_config_normalized():
    """Legacy single-model config creates default model."""
    config = load_config("test", "legacy-project")
    assert "default" in config.runtime.models
    assert config.runtime.default_model == "default"

def test_multi_model_selection():
    """Model selection works with multi-model config."""
    config = load_config("test", "multi-model-project")
    model_config = ModelService.get_model_config(config, "fast-model")
    assert model_config.provider == Provider.ollama
    assert model_config.model == "gemma3:1b"

def test_model_not_found_raises():
    """Selecting non-existent model raises error."""
    config = load_config("test", "multi-model-project")
    with pytest.raises(ValueError, match="not found"):
        ModelService.get_model_config(config, "nonexistent")
```

### Integration Tests

```python
# test_multi_model_chat.py
async def test_chat_with_model_selection():
    """API accepts model parameter and uses correct config."""
    response = await client.post(
        "/api/v1/projects/test/my-project/chat",
        json={"message": "Hello", "model": "fast-model"}
    )
    assert response.status_code == 200
    # Verify correct model was used (check logs or response metadata)

async def test_list_models_endpoint():
    """GET /models returns available models."""
    response = await client.get("/api/v1/projects/test/my-project/models")
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) > 0
    assert any(m["is_default"] for m in data["data"])
```

### CLI Tests

```bash
# Test model selection
lf chat --model fast-model "What is 2+2?"

# Test default model
lf chat "What is 2+2?"

# Test listing models
lf models list
```

---

## 🚀 Provider-Specific Considerations

### Ollama: Multiple Models (Same Provider)

**Hot-loading is NOT needed** - Ollama manages multiple models natively:

```yaml
runtime:
  models:
    small:
      provider: ollama
      model: gemma3:1b        # Model 1
    large:
      provider: ollama
      model: qwen2.5:7b       # Model 2
```

Ollama loads models on-demand. No code changes needed.

### Lemonade: Multi-Instance Support

**Current limitation**: Lemonade can only run ONE model per server instance.

**Solution**: Run multiple Lemonade instances on different ports.

```yaml
runtime:
  default_model: lemonade-small

  models:
    lemonade-small:
      provider: lemonade
      model: Qwen3-0.6B-GGUF
      lemonade:
        port: 11534           # Instance 1
        backend: llamacpp
        context_size: 32768

    lemonade-large:
      provider: lemonade
      model: Llama-3.2-1B-GGUF
      lemonade:
        port: 11535           # Instance 2 (different port)
        backend: llamacpp
        context_size: 65536
```

#### NX Start Command Enhancement

**Current behavior**: `nx start lemonade` starts a single instance

**New behavior**: `nx start lemonade` detects all Lemonade models in config and starts multiple instances

**Implementation**:

1. **Update `runtimes/lemonade/start.sh`** to accept instance parameters:

```bash
#!/bin/bash
# New parameters
LEMONADE_INSTANCE_NAME="${LEMONADE_INSTANCE_NAME:-default}"
LEMONADE_PORT="${LEMONADE_PORT:-11534}"
LEMONADE_MODEL="${LEMONADE_MODEL:-}"
LEMONADE_BACKEND="${LEMONADE_BACKEND:-onnx}"
LEMONADE_CONTEXT_SIZE="${LEMONADE_CONTEXT_SIZE:-32768}"

# Allow instance-specific configuration
echo "Starting Lemonade instance: $LEMONADE_INSTANCE_NAME"
echo "Model: $LEMONADE_MODEL on port $LEMONADE_PORT"

# Start with instance-specific settings
LEMONADE_CMD="uv run lemonade-server-dev run $LEMONADE_MODEL --port $LEMONADE_PORT --host $LEMONADE_HOST --no-tray"
# ... rest of startup logic
```

2. **Create wrapper script `runtimes/lemonade/start-multi.sh`**:

```bash
#!/bin/bash
# Multi-instance Lemonade startup script
# Reads llamafarm.yaml and starts all Lemonade model instances

set -e

CONFIG_FILE="../llamafarm.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "No llamafarm.yaml found, starting single default instance"
    bash start.sh
    exit 0
fi

# Parse config and extract Lemonade models
LEMONADE_MODELS=$(uv run python -c "
import yaml
import sys

try:
    with open('$CONFIG_FILE') as f:
        config = yaml.safe_load(f)

    runtime = config.get('runtime', {})
    models = runtime.get('models', {})

    # Find all Lemonade models
    lemonade_models = []
    for name, model_config in models.items():
        if model_config.get('provider') == 'lemonade':
            lemonade_config = model_config.get('lemonade', {})
            lemonade_models.append({
                'name': name,
                'model': model_config.get('model'),
                'port': lemonade_config.get('port', 11534),
                'backend': lemonade_config.get('backend', 'onnx'),
                'context_size': lemonade_config.get('context_size', 32768)
            })

    # Print as JSON for bash to parse
    import json
    print(json.dumps(lemonade_models))
except Exception as e:
    print('[]', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null)

if [ -z "$LEMONADE_MODELS" ] || [ "$LEMONADE_MODELS" = "[]" ]; then
    echo "No Lemonade models found in config, starting single default instance"
    bash start.sh
    exit 0
fi

# Start each Lemonade instance
echo "Found multiple Lemonade models, starting instances..."
echo "$LEMONADE_MODELS" | jq -c '.[]' | while read -r model; do
    NAME=$(echo "$model" | jq -r '.name')
    MODEL=$(echo "$model" | jq -r '.model')
    PORT=$(echo "$model" | jq -r '.port')
    BACKEND=$(echo "$model" | jq -r '.backend')
    CONTEXT_SIZE=$(echo "$model" | jq -r '.context_size')

    echo ""
    echo "==================================="
    echo "Starting Lemonade instance: $NAME"
    echo "Model: $MODEL"
    echo "Port: $PORT"
    echo "Backend: $BACKEND"
    echo "==================================="
    echo ""

    # Start instance in background
    LEMONADE_INSTANCE_NAME="$NAME" \
    LEMONADE_MODEL="$MODEL" \
    LEMONADE_PORT="$PORT" \
    LEMONADE_BACKEND="$BACKEND" \
    LEMONADE_CONTEXT_SIZE="$CONTEXT_SIZE" \
    bash start.sh &

    # Store PID for cleanup
    echo $! >> /tmp/lemonade_pids_$$.txt

    # Wait a bit between starts
    sleep 2
done

echo ""
echo "All Lemonade instances started!"
echo "PIDs saved to /tmp/lemonade_pids_$$.txt"

# Keep script running
wait
```

3. **Update `runtimes/lemonade/project.json`**:

```json
{
  "name": "lemonade",
  "$schema": "../../node_modules/nx/schemas/project-schema.json",
  "projectType": "application",
  "sourceRoot": "runtimes/lemonade",
  "targets": {
    "start": {
      "executor": "nx:run-commands",
      "options": {
        "command": "bash runtimes/lemonade/start-multi.sh",
        "cwd": "{projectRoot}"
      }
    },
    "start-single": {
      "executor": "nx:run-commands",
      "options": {
        "command": "bash runtimes/lemonade/start.sh",
        "cwd": "{projectRoot}"
      },
      "description": "Start a single Lemonade instance (legacy mode)"
    },
    "stop": {
      "executor": "nx:run-commands",
      "options": {
        "command": "bash runtimes/lemonade/stop.sh",
        "cwd": "{projectRoot}"
      }
    }
  }
}
```

4. **Create stop script `runtimes/lemonade/stop.sh`**:

```bash
#!/bin/bash
# Stop all Lemonade instances

echo "Stopping all Lemonade instances..."

# Kill processes on known Lemonade ports (11534-11544)
for port in {11534..11544}; do
    if lsof -ti :$port >/dev/null 2>&1; then
        echo "Stopping Lemonade on port $port..."
        lsof -ti :$port | xargs -r kill -9 2>/dev/null
    fi
done

# Also kill using PID file if it exists
if [ -f /tmp/lemonade_pids_*.txt ]; then
    for pid_file in /tmp/lemonade_pids_*.txt; do
        while read -r pid; do
            if ps -p "$pid" > /dev/null 2>&1; then
                echo "Stopping Lemonade process $pid..."
                kill -9 "$pid" 2>/dev/null
            fi
        done < "$pid_file"
        rm "$pid_file"
    done
fi

echo "All Lemonade instances stopped."
```

#### Port Allocation Strategy

**Automatic port assignment**: If user doesn't specify ports, auto-assign from pool

**Port range**: 11534-11544 (supports up to 10 Lemonade instances)

```yaml
# User can omit ports - system auto-assigns
runtime:
  models:
    lemonade-1:
      provider: lemonade
      model: Qwen3-0.6B-GGUF
      # Port auto-assigned: 11534

    lemonade-2:
      provider: lemonade
      model: Llama-3.2-1B-GGUF
      # Port auto-assigned: 11535
```

**Port assignment logic in `start-multi.sh`**:

```bash
# Auto-assign ports if not specified
NEXT_PORT=11534
for model in $LEMONADE_MODELS; do
    if [ -z "$(echo $model | jq -r '.port')" ]; then
        # Find next available port
        while lsof -ti :$NEXT_PORT >/dev/null 2>&1; do
            NEXT_PORT=$((NEXT_PORT + 1))
        done
        # Assign port to model
        PORT=$NEXT_PORT
        NEXT_PORT=$((NEXT_PORT + 1))
    else
        PORT=$(echo $model | jq -r '.port')
    fi
    # ... start instance with $PORT
done
```

#### Validation Rules

**Config validation** (in `ModelService` or startup):

1. **Unique ports required**: Each Lemonade model must have a unique port
2. **Port range check**: Ports should be in range 11534-11544 (or configurable range)
3. **Model required**: Each Lemonade config must specify a model name

```python
# In ModelService
def validate_lemonade_models(config: LlamaFarmConfig):
    """Validate Lemonade model configurations."""
    lemonade_models = [
        (name, model)
        for name, model in config.runtime.models.items()
        if model.provider == Provider.lemonade
    ]

    if len(lemonade_models) == 0:
        return  # No Lemonade models, nothing to validate

    # Check for duplicate ports
    ports = [m[1].lemonade.port for m in lemonade_models if m[1].lemonade]
    if len(ports) != len(set(ports)):
        raise ValueError(
            "Duplicate Lemonade ports detected. Each Lemonade model must use a unique port."
        )

    # Check for missing models
    for name, model in lemonade_models:
        if not model.model:
            raise ValueError(f"Lemonade model '{name}' missing model name")
```

#### Health Check Updates

**Update `health_service.py`** to check all Lemonade instances:

```python
def _check_all_lemonade_instances(config: LlamaFarmConfig) -> list[dict]:
    """Check health of all configured Lemonade instances."""
    lemonade_models = [
        (name, model)
        for name, model in config.runtime.models.items()
        if model.provider == Provider.lemonade
    ]

    health_results = []
    for name, model in lemonade_models:
        port = model.lemonade.port if model.lemonade else 11534
        provider = get_provider(Provider.lemonade)

        health = provider.check_health({"port": port})
        health["instance_name"] = name
        health["model_name"] = model.model
        health_results.append(health)

    return health_results
```

#### Example Multi-Instance Configuration

```yaml
version: v1
name: multi-lemonade-project
namespace: default

runtime:
  default_model: tiny

  models:
    # Ollama models (single instance, multiple models)
    ollama-small:
      provider: ollama
      model: gemma3:1b

    ollama-large:
      provider: ollama
      model: qwen2.5:7b

    # Lemonade models (multiple instances, different ports)
    tiny:
      description: "Tiny model for testing (600MB)"
      provider: lemonade
      model: Qwen3-0.6B-GGUF
      lemonade:
        port: 11534
        backend: llamacpp
        context_size: 32768

    small:
      description: "Small model for general use (1.2GB)"
      provider: lemonade
      model: Llama-3.2-1B-GGUF
      lemonade:
        port: 11535
        backend: llamacpp
        context_size: 65536

    vision:
      description: "Vision-capable model (1.5GB)"
      provider: lemonade
      model: Qwen2-VL-2B-GGUF
      lemonade:
        port: 11536
        backend: transformers  # Different backend
        context_size: 32768

prompts:
  - role: system
    content: "You are a helpful assistant."
```

**Command to start all instances**:
```bash
nx start lemonade
```

**Output**:
```
Found multiple Lemonade models, starting instances...

===================================
Starting Lemonade instance: tiny
Model: Qwen3-0.6B-GGUF
Port: 11534
Backend: llamacpp
===================================

===================================
Starting Lemonade instance: small
Model: Llama-3.2-1B-GGUF
Port: 11535
Backend: llamacpp
===================================

===================================
Starting Lemonade instance: vision
Model: Qwen2-VL-2B-GGUF
Port: 11536
Backend: transformers
===================================

All Lemonade instances started!
```

#### Files to Change (Lemonade Multi-Instance)

- ✏️ `runtimes/lemonade/start.sh` - Add instance name parameter support
- ✨ `runtimes/lemonade/start-multi.sh` - NEW: Multi-instance orchestrator
- ✨ `runtimes/lemonade/stop.sh` - NEW: Stop all instances
- ✏️ `runtimes/lemonade/project.json` - Update NX targets
- ✏️ `server/services/model_service.py` - Add Lemonade validation
- ✏️ `server/services/health_service.py` - Check all instances
- ✏️ `docs/runtimes/lemonade/README.md` - Document multi-instance setup

**Included in Phase 2** (after basic multi-model support is working)

---

## 📊 Example Configurations

### Example 1: Ollama Multi-Model

```yaml
version: v1
name: my-app
namespace: default

runtime:
  default_model: fast

  models:
    fast:
      description: "Fast responses for simple queries"
      provider: ollama
      model: gemma3:1b
      prompt_format: unstructured

    smart:
      description: "Complex reasoning tasks"
      provider: ollama
      model: qwen2.5:7b
      prompt_format: structured
      instructor_mode: md_json
```

### Example 2: Mixed Providers

```yaml
runtime:
  default_model: local

  models:
    local:
      description: "Local Ollama for privacy"
      provider: ollama
      model: gemma3:1b

    lemonade:
      description: "Tiny model for testing"
      provider: lemonade
      model: Qwen3-0.6B-GGUF
      lemonade:
        backend: llamacpp
```

### Example 3: Legacy Compatibility

```yaml
# Old config - still works!
runtime:
  provider: ollama
  model: gemma3:1b
  base_url: http://localhost:11434/v1
```

Internally normalized to:

```yaml
runtime:
  default_model: default
  models:
    default:
      provider: ollama
      model: gemma3:1b
      base_url: http://localhost:11434/v1
```

---

## 🔮 Future Enhancements (Not in Scope)

### 1. Model Routing & Fallback

```yaml
runtime:
  routing:
    strategy: fallback  # or: round-robin, load-balance
    models:
      - fast-model
      - smart-model  # Fallback if fast fails
```

### 2. Cloud Models

```yaml
runtime:
  models:
    cloud-gpt4:
      provider: openai
      model: gpt-4
      api_key: sk-...

    local-backup:
      provider: ollama
      model: qwen2.5:7b
```

### 3. Cost-Based Routing

```yaml
runtime:
  models:
    cheap:
      provider: ollama
      model: gemma3:1b
      cost_per_1k_tokens: 0.0

    expensive:
      provider: openai
      model: gpt-4
      cost_per_1k_tokens: 0.03
```

---

## ✅ Success Criteria

- [ ] Legacy single-model configs work without changes
- [ ] Multi-model configs can define 2+ models
- [ ] API accepts `model` parameter (OpenAI-compatible)
- [ ] CLI accepts `--model` flag
- [ ] `GET /models` endpoint returns available models
- [ ] Default model is used when no model specified
- [ ] Provider registry works with multi-model setup
- [ ] All existing tests pass
- [ ] New tests for multi-model scenarios pass
- [ ] Documentation updated with examples

---

## 📝 Files to Change

### Configuration
- ✏️ `config/schema.yaml` - Add `models`, `default_model`
- 🔄 `config/datamodel.py` - Auto-generated via `generate-types.sh`

### Backend (Python)
- ✏️ `server/services/project_service.py` - Add config normalization
- ✨ `server/services/model_service.py` - NEW: Model resolution service
- ✏️ `server/agents/project_chat_orchestrator.py` - Accept `model_name`
- ✏️ `server/api/routers/projects/projects.py` - Add `model` parameter, `/models` endpoint
- ✏️ `server/api/routers/inference/models.py` - Update `ChatRequest`

### CLI (Go)
- ✏️ `cli/cmd/chat.go` - Add `--model` flag
- ✏️ `cli/cmd/chat_client.go` - Send model in request
- ✨ `cli/cmd/models.go` - NEW: `lf models list` command

### Tests
- ✨ `server/tests/test_model_service.py` - NEW: Model service tests
- ✏️ `server/tests/test_chat_api.py` - Add multi-model tests
- ✏️ `cli/cmd/chat_test.go` - Add model selection tests

### Documentation
- ✏️ `docs/website/docs/configuration/index.md` - Multi-model examples
- ✏️ `docs/website/docs/cli/index.md` - Document `--model` flag
- ✨ `docs/website/docs/configuration/multi-model.md` - NEW: Multi-model guide
- ✏️ `README.md` - Add multi-model example

**Legend**: ✏️ = Modify, ✨ = New file, 🔄 = Auto-generated

---

## 🎯 Implementation Order

1. **Schema changes** (Day 1)
   - Update `config/schema.yaml`
   - Run `generate-types.sh`
   - Verify schema validation

2. **Backend foundation** (Day 2-3)
   - Create `ModelService`
   - Update `ProjectService` with normalization
   - Add unit tests

3. **Agent updates** (Day 3-4)
   - Update `ProjectChatOrchestratorAgent`
   - Update factory pattern
   - Test with single and multi-model configs

4. **API layer** (Day 4-5)
   - Add `model` to `ChatRequest`
   - Add `/models` endpoint
   - Integration tests

5. **CLI** (Day 5-6)
   - Add `--model` flag
   - Add `lf models list`
   - CLI tests

6. **Documentation** (Day 6-7)
   - Configuration guide
   - Migration guide
   - Examples

---

## 🔒 Backward Compatibility Guarantees

✅ **These will continue to work:**
- Existing `llamafarm.yaml` files with single `runtime.provider` + `runtime.model`
- All existing CLI commands without `--model` flag
- All existing API calls without `model` parameter

✅ **Migration is optional:**
- Users can keep using single-model configs indefinitely
- No forced migration to multi-model format
- Legacy format is auto-normalized internally

✅ **No breaking changes:**
- Schema adds optional fields only
- All new fields are nullable
- Default behavior unchanged for legacy configs

---

**End of Plan**
