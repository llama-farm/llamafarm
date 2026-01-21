# Inference Performance Checklist

Optimization patterns for model caching, batching, memory management, and inference speed.

---

## Category: Model Caching

### TTL-Based Model Cache

**What to check**: Use TTL cache to auto-unload idle models

**Search pattern**:
```bash
rg "ModelCache|TTLCache" --type py runtimes/universal/
```

**Pass criteria**:
- Models cached with configurable TTL (default: 5 minutes)
- TTL refreshed on each access
- Background task cleans up expired models

**Severity**: High

**Good pattern** (from utils/model_cache.py):
```python
class ModelCache(Generic[T]):
    def __init__(self, ttl: float, maxsize: int = 1000):
        self._ttl = ttl
        self._cache: TTLCache[str, T] = TTLCache(maxsize=maxsize, ttl=ttl * 10)
        self._access: dict[str, float] = {}

    def get(self, key: str, default: T | None = None) -> T | None:
        if key not in self._cache:
            return default
        self._access[key] = self._timer()  # Refresh TTL on read
        return self._cache[key]
```

---

### Configurable Unload Timeout

**What to check**: Allow environment variable control of model timeout

**Search pattern**:
```bash
rg "MODEL_UNLOAD_TIMEOUT|CLEANUP_CHECK_INTERVAL" --type py runtimes/universal/
```

**Pass criteria**:
- MODEL_UNLOAD_TIMEOUT configurable (default: 300 seconds)
- CLEANUP_CHECK_INTERVAL configurable (default: 30 seconds)
- Values read from environment at startup

**Severity**: Medium

**Good pattern**:
```python
MODEL_UNLOAD_TIMEOUT = int(os.getenv("MODEL_UNLOAD_TIMEOUT", "300"))
CLEANUP_CHECK_INTERVAL = int(os.getenv("CLEANUP_CHECK_INTERVAL", "30"))

_models: ModelCache[BaseModel] = ModelCache(ttl=MODEL_UNLOAD_TIMEOUT)
```

---

### Cache Key Design for Model Variants

**What to check**: Include all configuration in cache keys

**Search pattern**:
```bash
rg "_make_.*_cache_key" --type py runtimes/universal/
```

**Pass criteria**:
- Cache key includes model_id
- Cache key includes task/mode (embedding, classification)
- Cache key includes quantization preference
- Cache key includes context size (for GGUF)

**Severity**: High

**Good pattern**:
```python
def _make_encoder_cache_key(
    model_id: str,
    task: str,
    model_format: str,
    preferred_quantization: str | None = None,
    max_length: int | None = None,
) -> str:
    quant_key = preferred_quantization or "default"
    len_key = max_length if max_length is not None else "auto"
    return f"encoder:{task}:{model_format}:{model_id}:quant{quant_key}:len{len_key}"
```

---

## Category: Lazy Loading

### Double-Checked Locking for Model Loading

**What to check**: Prevent duplicate model loading with locking

**Search pattern**:
```bash
rg "_model_load_lock|async with.*lock" --type py runtimes/universal/
```

**Pass criteria**:
- Check cache before acquiring lock
- Double-check cache after acquiring lock
- Only one instance of each model loaded

**Severity**: Critical

**Good pattern**:
```python
_model_load_lock = asyncio.Lock()

async def load_encoder(model_id: str, task: str = "embedding"):
    cache_key = _make_encoder_cache_key(model_id, task, model_format)

    if cache_key not in _models:
        async with _model_load_lock:
            # Double-check after acquiring lock
            if cache_key not in _models:
                model = EncoderModel(model_id, device, task=task)
                await model.load()
                _models[cache_key] = model

    return _models.get(cache_key)  # get() refreshes TTL
```

---

### Load on First Request

**What to check**: Models loaded lazily, not at startup

**Search pattern**:
```bash
rg "await.*load\(\)" --type py runtimes/universal/server.py
```

**Pass criteria**:
- No models loaded in lifespan startup
- Models loaded when first endpoint is called
- Fast server startup time

**Severity**: Medium

---

## Category: Memory Optimization

### Periodic Cleanup Task

**What to check**: Background task unloads idle models

**Search pattern**:
```bash
rg "_cleanup_idle_models|pop_expired" --type py runtimes/universal/
```

**Pass criteria**:
- Cleanup task runs periodically
- Uses pop_expired() to get idle models
- Calls await model.unload() for each
- Continues running despite individual errors

**Severity**: High

**Good pattern**:
```python
async def _cleanup_idle_models() -> None:
    while True:
        try:
            await asyncio.sleep(CLEANUP_CHECK_INTERVAL)

            for cache, cache_name in [(_models, "models"), (_classifiers, "classifiers")]:
                expired_items = cache.pop_expired()
                for cache_key, model in expired_items:
                    try:
                        await model.unload()
                        logger.info(f"Unloaded: {cache_key}")
                    except Exception as e:
                        logger.error(f"Error unloading {cache_key}: {e}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            # Continue running
```

---

### Graceful Shutdown Model Cleanup

**What to check**: Unload all models on server shutdown

**Search pattern**:
```bash
rg "Shutting down|Unloading.*remaining" --type py runtimes/universal/
```

**Pass criteria**:
- All cached models unloaded
- GPU memory freed
- Log each model unload

**Severity**: High

**Good pattern**:
```python
# In lifespan shutdown:
if _models:
    logger.info(f"Unloading {len(_models)} remaining model(s)")
    for cache_key, model in list(_models.items()):
        try:
            await model.unload()
            logger.info(f"Unloaded: {cache_key}")
        except Exception as e:
            logger.error(f"Error unloading {cache_key}: {e}")
    _models.clear()
```

---

## Category: GGUF Optimizations

### Compute Optimal Context Size

**What to check**: Auto-calculate context size based on available memory

**Search pattern**:
```bash
rg "get_default_context_size|context_calculator" --type py runtimes/universal/
```

**Pass criteria**:
- Consider available GPU/system memory
- Read model's default from GGUF metadata
- Allow user override via parameter
- Log warnings if reduced

**Severity**: High

**Good pattern** (conceptual):
```python
def get_default_context_size(
    model_id: str,
    gguf_path: str,
    device: str,
    config_n_ctx: int | None = None,
) -> tuple[int, list[str]]:
    warnings = []

    # User-specified takes priority
    if config_n_ctx is not None:
        return config_n_ctx, warnings

    # Read model's default from GGUF
    model_default = read_gguf_context_length(gguf_path)

    # Calculate based on available memory
    available_memory = get_available_memory(device)
    computed_ctx = compute_safe_context(model_default, available_memory)

    if computed_ctx < model_default:
        warnings.append(f"Reduced context from {model_default} to {computed_ctx}")

    return computed_ctx, warnings
```

---

### Single-File Quantization Download

**What to check**: Only download the requested quantization

**Search pattern**:
```bash
rg "preferred_quantization|get_gguf_file_path" --type py runtimes/universal/
```

**Pass criteria**:
- Accept quantization preference (Q4_K_M, Q8_0, etc.)
- Download only the specified file
- Default to Q4_K_M (good balance of size/quality)

**Severity**: Medium

**Good pattern**:
```python
async def load_language(
    model_id: str,
    n_ctx: int | None = None,
    preferred_quantization: str | None = None,
):
    model = GGUFLanguageModel(
        model_id,
        device,
        n_ctx=n_ctx,
        preferred_quantization=preferred_quantization,  # Only downloads this one
    )
```

---

## Category: Batch Processing

### Batch Tokenization

**What to check**: Tokenize multiple inputs together for efficiency

**Search pattern**:
```bash
rg "self\.tokenizer\(" --type py runtimes/universal/models/ -A 5
```

**Pass criteria**:
- Pass list of texts to tokenizer (not one at a time)
- Use padding=True for batch processing
- Reduces tokenizer overhead

**Severity**: Medium

**Good pattern**:
```python
async def embed(self, texts: list[str], normalize: bool = True):
    # Batch tokenization - single call for all texts
    encoded = self.tokenizer(
        texts,  # List of texts
        padding=True,
        truncation=True,
        max_length=self.max_length,
        return_tensors="pt",
    )
```

---

### Batch Inference Where Possible

**What to check**: Process multiple inputs in single forward pass

**Search pattern**:
```bash
rg "model\(\*\*encoded\)" --type py runtimes/universal/models/
```

**Pass criteria**:
- Embedding: batch multiple texts
- Classification: batch multiple texts
- Generation: typically single request (streaming)

**Severity**: Medium

**Good pattern**:
```python
# Good: Batch processing for embeddings
async def embed(self, texts: list[str], normalize: bool = True):
    encoded = self.tokenizer(texts, padding=True, ...)
    with torch.no_grad():
        model_output = self.model(**encoded)  # All texts at once
    embeddings = self._mean_pooling(model_output, encoded["attention_mask"])
    return embeddings.cpu().tolist()
```

---

## Category: Streaming Performance

### Avoid Buffering in Streams

**What to check**: Yield tokens immediately without buffering

**Search pattern**:
```bash
rg "async for.*in.*stream" --type py runtimes/universal/
```

**Pass criteria**:
- Yield each token as it's generated
- Use asyncio.sleep(0) to force flush
- Set no-cache headers on response

**Severity**: High

**Good pattern**:
```python
async for token in model.generate_stream(messages=messages, ...):
    chunk = ChatCompletionChunk(...)
    yield f"data: {chunk.model_dump_json()}\n\n".encode()
    await asyncio.sleep(0)  # Force immediate delivery
```

---

### Queue-Based Streaming for Blocking Backends

**What to check**: Use asyncio.Queue for thread-based generation

**Search pattern**:
```bash
rg "asyncio\.Queue" --type py runtimes/universal/
```

**Pass criteria**:
- Create queue for producer/consumer pattern
- Producer runs in thread pool
- Consumer yields from queue in async context

**Severity**: High

**Good pattern** (from gguf_language_model.py):
```python
async def generate_stream(self, messages, ...):
    queue: asyncio.Queue[str | Exception | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def _generate_stream():
        for chunk in self.llama.create_chat_completion(stream=True, ...):
            content = chunk["choices"][0].get("delta", {}).get("content", "")
            if content:
                future = asyncio.run_coroutine_threadsafe(queue.put(content), loop)
                future.result()
        asyncio.run_coroutine_threadsafe(queue.put(None), loop).result()

    loop.run_in_executor(self._executor, _generate_stream)

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item
```

---

## Category: Model Format Detection

### Automatic GGUF vs Transformers Detection

**What to check**: Detect model format automatically

**Search pattern**:
```bash
rg "detect_model_format|model_format" --type py runtimes/universal/
```

**Pass criteria**:
- Check HuggingFace repo for .gguf files
- Use GGUF loader for quantized models
- Use Transformers loader for standard models

**Severity**: High

**Good pattern**:
```python
# Detect model format
model_format = detect_model_format(model_id)

if model_format == "gguf":
    model = GGUFLanguageModel(model_id, device, n_ctx=n_ctx)
else:
    model = LanguageModel(model_id, device)
```

---

### Parse Model:Quantization Syntax

**What to check**: Support model:Q4_K_M syntax for quantization selection

**Search pattern**:
```bash
rg "parse_model_with_quantization" --type py runtimes/universal/
```

**Pass criteria**:
- Parse "model:Q4_K_M" into (model, Q4_K_M)
- Handle models without quantization suffix
- Support HuggingFace model IDs with slashes

**Severity**: Medium

**Good pattern**:
```python
# Parse model name to extract quantization if present
model_id, gguf_quantization = parse_model_with_quantization(request.model)

model = await self.load_language(
    model_id,
    n_ctx=n_ctx,
    preferred_quantization=gguf_quantization,
)
```
