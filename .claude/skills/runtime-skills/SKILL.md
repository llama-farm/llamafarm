---
name: runtime-skills
description: Universal Runtime best practices for PyTorch inference, Transformers models, and FastAPI serving. Covers device management, model loading, memory optimization, and performance tuning.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# Universal Runtime Skills

Best practices and code review checklists for the Universal Runtime - LlamaFarm's local ML inference server.

## Overview

The Universal Runtime provides OpenAI-compatible endpoints for HuggingFace models:
- Text generation (Causal LMs: GPT, Llama, Mistral, Qwen)
- Text embeddings (BERT, sentence-transformers, ModernBERT)
- Classification, NER, and reranking
- OCR and document understanding
- Anomaly detection

**Directory**: `runtimes/universal/`
**Python**: 3.11+
**Key Dependencies**: PyTorch, Transformers, FastAPI, llama-cpp-python

## Links to Shared Skills

This skill extends the shared Python practices. Always apply these first:

| Topic | File | Priority |
|-------|------|----------|
| Patterns | [python-skills/patterns.md](../python-skills/patterns.md) | Medium |
| Async | [python-skills/async.md](../python-skills/async.md) | High |
| Typing | [python-skills/typing.md](../python-skills/typing.md) | Medium |
| Testing | [python-skills/testing.md](../python-skills/testing.md) | Medium |
| Errors | [python-skills/error-handling.md](../python-skills/error-handling.md) | High |
| Security | [python-skills/security.md](../python-skills/security.md) | Critical |

## Runtime-Specific Checklists

| Topic | File | Key Points |
|-------|------|------------|
| PyTorch | [pytorch.md](pytorch.md) | Device management, dtype, memory cleanup |
| Transformers | [transformers.md](transformers.md) | Model loading, tokenization, inference |
| FastAPI | [fastapi.md](fastapi.md) | API design, streaming, lifespan |
| Performance | [performance.md](performance.md) | Batching, caching, optimizations |

## Architecture

```
runtimes/universal/
├── server.py              # FastAPI app, model caching, endpoints
├── core/
│   └── logging.py         # UniversalRuntimeLogger (structlog)
├── models/
│   ├── base.py            # BaseModel ABC with device management
│   ├── language_model.py  # Transformers text generation
│   ├── gguf_language_model.py  # llama-cpp-python for GGUF
│   ├── encoder_model.py   # Embeddings, classification, NER, reranking
│   └── ...                # OCR, anomaly, document models
├── routers/
│   └── chat_completions/  # Chat completions with streaming
├── utils/
│   ├── device.py          # Device detection (CUDA/MPS/CPU)
│   ├── model_cache.py     # TTL-based model caching
│   ├── model_format.py    # GGUF vs transformers detection
│   └── context_calculator.py  # GGUF context size computation
└── tests/
```

## Key Patterns

### 1. Model Loading with Double-Checked Locking

```python
_model_load_lock = asyncio.Lock()

async def load_encoder(model_id: str, task: str = "embedding"):
    cache_key = f"encoder:{task}:{model_id}"
    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check after acquiring lock
            if cache_key not in _models:
                model = EncoderModel(model_id, device, task=task)
                await model.load()
                _models[cache_key] = model
    return _models.get(cache_key)
```

### 2. Device-Aware Tensor Operations

```python
class BaseModel(ABC):
    def get_dtype(self, force_float32: bool = False):
        if force_float32:
            return torch.float32
        if self.device in ("cuda", "mps"):
            return torch.float16
        return torch.float32

    def to_device(self, tensor: torch.Tensor, dtype=None):
        # Don't change dtype for integer tensors
        if tensor.dtype in (torch.int32, torch.int64, torch.long):
            return tensor.to(device=self.device)
        dtype = dtype or self.get_dtype()
        return tensor.to(device=self.device, dtype=dtype)
```

### 3. TTL-Based Model Caching

```python
_models: ModelCache[BaseModel] = ModelCache(ttl=300)  # 5 min TTL

async def _cleanup_idle_models():
    while True:
        await asyncio.sleep(CLEANUP_CHECK_INTERVAL)
        for cache_key, model in _models.pop_expired():
            await model.unload()
```

### 4. Async Generation with Thread Pools

```python
# GGUF models use blocking llama-cpp, run in executor
self._executor = ThreadPoolExecutor(max_workers=1)

async def generate(self, messages, max_tokens=512, ...):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(self._executor, self._generate_sync)
```

## Review Priority

When reviewing Universal Runtime code:

1. **Critical** - Security
   - Path traversal prevention in file endpoints
   - Input sanitization for model IDs

2. **High** - Memory & Device
   - Proper CUDA/MPS cache clearing on unload
   - torch.no_grad() for inference
   - Correct dtype for device

3. **Medium** - Performance
   - Model caching patterns
   - Batch processing where applicable
   - Streaming implementation

4. **Low** - Code Style
   - Consistent with patterns.md
   - Proper type hints
