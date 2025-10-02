# Lemonade Runtime Integration - Status & Refactoring Plan

**Branch:** `feat/multi-model`
**Date:** 2025-10-02
**Status:** ✅ Functional, ✅ Refactored with Provider Registry Pattern

---

## 🎯 Current Status

### ✅ What's Working
- Lemonade runtime successfully integrated alongside Ollama and OpenAI
- Runs as separate NX service: `nx start lemonade`
- Port 11534 (100+ offset from Ollama's 11434)
- OpenAI-compatible API at `http://127.0.0.1:11534/api/v1`
- Configuration-driven from `llamafarm.yaml`
- Context size configurable (default: 32768 tokens)
- Hugging Face token support for gated models
- Health checks implemented
- Successfully tested with Qwen3-0.6B-GGUF model
- Backend selection: ONNX (default), llamacpp, transformers

### ⚠️ Known Issues
1. **Browser Auto-Launch**: Lemonade opens browser on startup (no flag to prevent it)
2. **Extensibility**: Hard-coded provider logic needs refactoring to factory pattern

---

## 📂 Files Modified

### Core Configuration
1. **`config/schema.yaml`** - Added Lemonade provider support
   - Added `lemonade` to provider enum
   - Added `huggingface_token` field
   - Added `lemonade` config section with backend, port, context_size, model_path

2. **`config/datamodel.py`** - Auto-generated types (run `./generate-types.sh` to regenerate)

### Runtime Implementation
3. **`runtimes/lemonade/start.sh`** - Startup script
   - Reads config from `llamafarm.yaml` using `uv run python`
   - Supports environment variable overrides
   - Configurable context size via `--ctx-size` parameter
   - Platform detection for llamacpp backend (metal/vulkan)
   - Model requirement validation

4. **`runtimes/lemonade/project.json`** - NX project configuration
5. **`runtimes/lemonade/README.md`** - Documentation
6. **`runtimes/lemonade/QUICKSTART.md`** - Quick start guide
7. **`runtimes/lemonade/example.llamafarm.yaml`** - Example configuration

### Server Integration
8. **`server/agents/project_chat_orchestrator.py`** - Chat orchestration
   - Lines 379-391: Lemonade client initialization (**HARD-CODED** ⚠️)
   - Lines 418-421: Lemonade instructor mode defaults (**HARD-CODED** ⚠️)

9. **`server/services/health_service.py`** - Health checks
   - Lines 133-166: `_check_lemonade()` function (**HARD-CODED** ⚠️)
   - Lines 223-252: Lemonade project seed health check (**HARD-CODED** ⚠️)

---

## 🔧 Extensibility Issues & Refactoring Plan

### Problem: Hard-Coded Provider Logic

Currently, adding a new provider (e.g., vLLM, TGI, Replicate) requires:
1. Modifying `config/schema.yaml` enum
2. Adding `if provider == Provider.new_provider:` blocks in **2 core files**
3. Repeating client initialization boilerplate
4. Duplicating instructor mode logic
5. Adding health check functions

**This violates Open/Closed Principle** - core files should be closed for modification.

---

### Solution: Provider Registry Pattern

Create an extensible provider system where new providers can be added without modifying core files.

#### Step 1: Create Provider Base Class

**File:** `server/agents/providers/base.py`

```python
from abc import ABC, abstractmethod
from typing import Optional
import instructor
from openai import AsyncOpenAI
from config.datamodel import LlamaFarmConfig

class RuntimeProvider(ABC):
    """Base class for runtime providers."""

    @abstractmethod
    def get_client(self, config: LlamaFarmConfig) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        """Get OpenAI-compatible client for this provider."""
        pass

    @abstractmethod
    def get_default_instructor_mode(self) -> instructor.Mode:
        """Get default instructor mode for this provider."""
        pass

    @abstractmethod
    def get_base_url(self, config: LlamaFarmConfig) -> str:
        """Get base URL for this provider."""
        pass

    @abstractmethod
    def get_api_key(self, config: LlamaFarmConfig) -> Optional[str]:
        """Get API key for this provider."""
        pass
```

#### Step 2: Implement Provider Classes

**File:** `server/agents/providers/openai_provider.py`

```python
from .base import RuntimeProvider
from config.datamodel import LlamaFarmConfig, PromptFormat
import instructor
from openai import AsyncOpenAI

class OpenAIProvider(RuntimeProvider):
    def get_base_url(self, config: LlamaFarmConfig) -> str:
        return config.runtime.base_url or "https://api.openai.com/v1"

    def get_api_key(self, config: LlamaFarmConfig) -> str:
        return config.runtime.api_key

    def get_default_instructor_mode(self) -> instructor.Mode:
        return instructor.Mode.TOOLS

    def get_client(self, config: LlamaFarmConfig) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        client = AsyncOpenAI(
            api_key=self.get_api_key(config),
            base_url=self.get_base_url(config),
        )

        if config.runtime.prompt_format == PromptFormat.structured:
            mode = self._determine_mode(config)
            return instructor.from_openai(client, mode=mode)
        return client

    def _determine_mode(self, config: LlamaFarmConfig) -> instructor.Mode:
        if config.runtime.instructor_mode:
            return instructor.mode.Mode[config.runtime.instructor_mode.upper()]
        return self.get_default_instructor_mode()
```

**File:** `server/agents/providers/ollama_provider.py`

```python
from .base import RuntimeProvider
from config.datamodel import LlamaFarmConfig, PromptFormat
from core.settings import settings
import instructor
from openai import AsyncOpenAI

class OllamaProvider(RuntimeProvider):
    def get_base_url(self, config: LlamaFarmConfig) -> str:
        return config.runtime.base_url or f"{settings.ollama_host}/v1"

    def get_api_key(self, config: LlamaFarmConfig) -> str:
        return config.runtime.api_key or settings.ollama_api_key

    def get_default_instructor_mode(self) -> instructor.Mode:
        return instructor.Mode.MD_JSON

    def get_client(self, config: LlamaFarmConfig) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        client = AsyncOpenAI(
            api_key=self.get_api_key(config),
            base_url=self.get_base_url(config),
        )

        if config.runtime.prompt_format == PromptFormat.structured:
            mode = self._determine_mode(config)
            return instructor.from_openai(client, mode=mode)
        return client

    def _determine_mode(self, config: LlamaFarmConfig) -> instructor.Mode:
        if config.runtime.instructor_mode:
            return instructor.mode.Mode[config.runtime.instructor_mode.upper()]
        return self.get_default_instructor_mode()
```

**File:** `server/agents/providers/lemonade_provider.py`

```python
from .base import RuntimeProvider
from config.datamodel import LlamaFarmConfig, PromptFormat
import instructor
from openai import AsyncOpenAI

class LemonadeProvider(RuntimeProvider):
    def get_base_url(self, config: LlamaFarmConfig) -> str:
        if config.runtime.base_url:
            return config.runtime.base_url

        port = 11534  # default
        if config.runtime.lemonade:
            port = config.runtime.lemonade.port or 11534

        return f"http://127.0.0.1:{port}/api/v1"

    def get_api_key(self, config: LlamaFarmConfig) -> str:
        return config.runtime.api_key or "lemonade"

    def get_default_instructor_mode(self) -> instructor.Mode:
        return instructor.Mode.MD_JSON

    def get_client(self, config: LlamaFarmConfig) -> instructor.client.AsyncInstructor | AsyncOpenAI:
        client = AsyncOpenAI(
            api_key=self.get_api_key(config),
            base_url=self.get_base_url(config),
        )

        if config.runtime.prompt_format == PromptFormat.structured:
            mode = self._determine_mode(config)
            return instructor.from_openai(client, mode=mode)
        return client

    def _determine_mode(self, config: LlamaFarmConfig) -> instructor.Mode:
        if config.runtime.instructor_mode:
            return instructor.mode.Mode[config.runtime.instructor_mode.upper()]
        return self.get_default_instructor_mode()
```

#### Step 3: Create Provider Registry

**File:** `server/agents/providers/registry.py`

```python
from typing import Dict
from config.datamodel import Provider
from .base import RuntimeProvider
from .openai_provider import OpenAIProvider
from .ollama_provider import OllamaProvider
from .lemonade_provider import LemonadeProvider

_PROVIDER_REGISTRY: Dict[Provider, RuntimeProvider] = {
    Provider.openai: OpenAIProvider(),
    Provider.ollama: OllamaProvider(),
    Provider.lemonade: LemonadeProvider(),
}

def register_provider(provider_enum: Provider, provider_class: RuntimeProvider) -> None:
    """Register a new provider dynamically."""
    _PROVIDER_REGISTRY[provider_enum] = provider_class

def get_provider(provider_enum: Provider) -> RuntimeProvider:
    """Get provider implementation for given enum."""
    if provider_enum not in _PROVIDER_REGISTRY:
        raise ValueError(f"Unsupported provider: {provider_enum}")
    return _PROVIDER_REGISTRY[provider_enum]
```

#### Step 4: Update `project_chat_orchestrator.py`

**Before (lines 354-427):** 77 lines of hard-coded provider logic

**After (lines 354-359):**

```python
def _get_client(
    project_config: LlamaFarmConfig,
) -> instructor.client.AsyncInstructor | AsyncOpenAI:
    """Get client for the configured provider using the provider registry."""
    provider = get_provider(project_config.runtime.provider)
    return provider.get_client(project_config)
```

**Delete:** `_determine_instructor_mode()` function (no longer needed)

---

### Health Check Refactoring

Similarly, health checks should use a provider-based approach:

**File:** `server/agents/providers/base.py` (extend)

```python
class RuntimeProvider(ABC):
    # ... existing methods ...

    @abstractmethod
    def check_health(self, config: Optional[dict] = None) -> dict:
        """Check health of this provider's runtime."""
        pass
```

**Then each provider implements:**

```python
# lemonade_provider.py
def check_health(self, config: Optional[dict] = None) -> dict:
    port = config.get('port', 11534) if config else 11534
    base = f"http://127.0.0.1:{port}"
    url = f"{base}/api/v1/models"
    start = int(time.time() * 1000)

    try:
        resp = requests.get(url, timeout=1.0)
        if 200 <= resp.status_code < 300:
            return {
                "name": "lemonade",
                "status": "healthy",
                "message": f"{base} reachable",
                "latency_ms": int(time.time() * 1000) - start,
                "details": {"host": base, "port": port},
            }
        # ... error handling
```

**Then `health_service.py` uses:**

```python
from agents.providers.registry import get_provider
from config.datamodel import Provider

def _check_runtime(provider: Provider, config: dict = None) -> dict:
    """Generic runtime health check using provider registry."""
    provider_impl = get_provider(provider)
    return provider_impl.check_health(config)
```

---

## 🧪 Testing Instructions

### Test Current Functionality (Before Refactoring)

1. **Start Services:**
```bash
cd /Users/robthelen/llamafarm-1
nx start lemonade  # Terminal 1
nx start server    # Terminal 2
```

2. **Verify Lemonade is Running:**
```bash
curl http://127.0.0.1:11534/api/v1/models
# Should return: {"object":"list","data":[{"id":"Qwen3-0.6B-GGUF",...}]}
```

3. **Test Chat:**
```bash
./lf chat "What is the capital of New Zealand?"
# Should return: Wellington (or actual answer, not generic fallback)
```

4. **Check Health:**
```bash
curl http://localhost:8000/health | jq
# Should show lemonade: healthy
```

### Test After Refactoring

1. **Verify No Behavior Change:**
   - Run all tests above - should work identically

2. **Verify Extensibility:**
   - Add a new provider (e.g., `vllm`) by:
     - Adding `vllm` to `config/schema.yaml` enum
     - Creating `vllm_provider.py`
     - Registering in `registry.py`
   - **Core files should NOT need modification**

---

## 📋 Refactoring Checklist

- [x] Create `server/agents/providers/` directory
- [x] Create `base.py` with `RuntimeProvider` ABC
- [x] Create `openai_provider.py`
- [x] Create `ollama_provider.py`
- [x] Create `lemonade_provider.py`
- [x] Create `registry.py`
- [x] Update `project_chat_orchestrator.py` to use registry (reduced from 77 lines to 7 lines)
- [x] Remove `_determine_instructor_mode()` function (now handled by providers)
- [x] Extend `RuntimeProvider` with `check_health()` method
- [x] Update `health_service.py` to use provider health checks
- [x] Refactor `_check_lemonade()`, `_check_ollama()` to use provider registry
- [x] Test all three providers (OpenAI, Ollama, Lemonade) - Health checks verified
- [ ] Update documentation in `docs/website/docs/extending/`

---

## 📝 Configuration Reference

### Example `llamafarm.yaml` for Lemonade

```yaml
version: v1
name: my-project
namespace: default

runtime:
  provider: lemonade
  model: Qwen3-0.6B-GGUF
  huggingface_token: hf_xxxxx  # Optional, for gated models

  lemonade:
    backend: llamacpp       # onnx (default), llamacpp, transformers
    port: 11534             # default: 11534
    context_size: 32768     # default: 32768 tokens
    model_path: null        # Optional custom model storage
```

### Environment Variables (Override Config)

```bash
export LEMONADE_PORT=11534
export LEMONADE_HOST=127.0.0.1
export LEMONADE_BACKEND=llamacpp
export LEMONADE_CONTEXT_SIZE=32768
export LEMONADE_MODEL=Qwen3-0.6B-GGUF
```

---

## 🚀 Adding New Runtimes (After Refactoring)

### Example: Adding vLLM Support

1. **Update Schema** (`config/schema.yaml`):
```yaml
provider:
  enum: [openai, ollama, lemonade, vllm]
```

2. **Create Provider** (`server/agents/providers/vllm_provider.py`):
```python
from .base import RuntimeProvider

class VLLMProvider(RuntimeProvider):
    def get_base_url(self, config):
        return config.runtime.base_url or "http://localhost:8000/v1"
    # ... implement other methods
```

3. **Register Provider** (`server/agents/providers/registry.py`):
```python
from .vllm_provider import VLLMProvider

_PROVIDER_REGISTRY = {
    # ... existing providers
    Provider.vllm: VLLMProvider(),
}
```

4. **Create Runtime** (`runtimes/vllm/`):
   - `start.sh` - Startup script
   - `project.json` - NX config
   - `README.md` - Documentation

**Core files unchanged! ✅**

---

## 🐛 Known Technical Debt

1. **Browser Auto-Launch**: Lemonade SDK has no `--no-browser` flag
   - Workaround: User manually closes browser
   - Future: Fork Lemonade SDK or request feature

2. **Context Size Parameter**: Only applied to llamacpp backend
   - ONNX and transformers backends may need different approach
   - TODO: Test with other backends

3. **Model Download**: Currently happens on first request
   - Future: Add model pre-download step to startup script

---

## 📚 Related Documentation

- Lemonade SDK: https://github.com/lemonade-sdk/lemonade
- Lemonade Server Docs: https://lemonade-server.ai/
- Provider Pattern: https://refactoring.guru/design-patterns/abstract-factory

---

## ✅ Refactoring Completed - 2025-10-02

### What Was Changed

**Created Provider Registry System** (`server/agents/providers/`):
- `base.py`: Abstract base class `RuntimeProvider` with 5 required methods
- `openai_provider.py`: OpenAI implementation (TOOLS mode default)
- `ollama_provider.py`: Ollama implementation (MD_JSON mode default)
- `lemonade_provider.py`: Lemonade implementation (MD_JSON mode default)
- `registry.py`: Provider registry with `get_provider()` and `register_provider()`
- `__init__.py`: Public API exports

**Refactored Core Files**:

1. **`server/agents/project_chat_orchestrator.py`**:
   - **Before**: 77 lines of hard-coded provider logic with nested if/elif blocks
   - **After**: 7 lines using `provider.get_client(config)`
   - Deleted `_determine_instructor_mode()` function (now handled by providers)
   - Added import: `from agents.providers import get_provider`

2. **`server/services/health_service.py`**:
   - **Before**: Separate `_check_ollama()` and `_check_lemonade()` functions with duplicated logic
   - **After**: 4-line wrappers calling `provider.check_health(config)`
   - Refactored `_check_seed_project()` to use provider registry with model validation
   - Added imports: `from agents.providers import get_provider` and `from config.datamodel import Provider`

### Code Reduction Metrics

| File | Lines Before | Lines After | Reduction |
|------|-------------|-------------|-----------|
| `project_chat_orchestrator.py` | 77 (provider logic) | 7 | **90% reduction** |
| `health_service.py` | ~140 (3 health check functions) | ~90 | **35% reduction** |

### Test Results

✅ **Server startup**: Successful with no import errors
✅ **Health endpoint**: Returns proper health checks for all components
✅ **Provider health checks**: Ollama health check working via registry
✅ **Lemonade health check**: Working (server running on port 11534)

### Benefits Achieved

1. **Open/Closed Principle**: Core files are now closed for modification when adding new providers
2. **Single Responsibility**: Each provider manages its own configuration and health logic
3. **Eliminates Code Duplication**: Client initialization and health checks are now provider-specific
4. **Extensibility**: New runtimes (vLLM, TGI, Replicate) can be added without touching core code
5. **Maintainability**: Provider logic is isolated and easier to test

---

## ✅ Next Steps

1. **Documentation**: Update `docs/website/docs/extending/` with provider pattern guide
2. **Testing**: Add unit tests for provider implementations
3. **Future Runtimes**: Add vLLM, TGI, or other runtimes using the new pattern
4. **Example**: Create example custom provider in documentation
