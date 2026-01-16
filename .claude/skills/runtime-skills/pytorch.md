# PyTorch Patterns Checklist

Device management, dtype handling, and memory optimization for PyTorch inference.

---

## Category: Device Management

### Use get_optimal_device() for Device Selection

**What to check**: Centralized device detection with proper fallbacks

**Search pattern**:
```bash
rg "get_optimal_device|get_device_info" --type py runtimes/universal/
```

**Pass criteria**:
- Device detected once and cached
- Supports CUDA, MPS (Apple Silicon), and CPU
- Environment variable overrides (TRANSFORMERS_FORCE_CPU, TRANSFORMERS_SKIP_MPS)

**Severity**: High

**Good pattern** (from utils/device.py):
```python
def get_optimal_device() -> str:
    if os.environ.get("TRANSFORMERS_FORCE_CPU", "").lower() in ("1", "true"):
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
```

---

### Cache Device Selection at Module Level

**What to check**: Device should be detected once, not on every request

**Search pattern**:
```bash
rg "get_device\(\)|get_optimal_device\(\)" --type py runtimes/universal/server.py
```

**Pass criteria**:
- Global `_current_device` cached after first detection
- get_device() function returns cached value

**Severity**: Medium

**Good pattern**:
```python
_current_device = None

def get_device():
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device
```

---

## Category: Dtype Handling

### Device-Appropriate Dtype Selection

**What to check**: Use float16 on GPU, float32 on CPU

**Search pattern**:
```bash
rg "get_dtype|torch\.float16|torch\.float32" --type py runtimes/universal/models/
```

**Pass criteria**:
- float16 for CUDA and MPS (memory efficient)
- float32 for CPU (better compatibility)
- force_float32 option for MPS compatibility issues

**Severity**: High

**Good pattern** (from models/base.py):
```python
def get_dtype(self, force_float32: bool = False):
    if force_float32:
        return torch.float32
    if self.device == "cuda" or self.device == "mps":
        return torch.float16
    else:
        return torch.float32
```

---

### Preserve Integer Tensor Dtypes

**What to check**: Don't convert input_ids and attention_mask to float

**Search pattern**:
```bash
rg "\.to\(device" --type py runtimes/universal/models/
```

**Pass criteria**:
- Integer tensors (input_ids, attention_mask) only moved to device
- Float tensors get appropriate dtype conversion

**Severity**: High

**Good pattern**:
```python
def to_device(self, tensor: torch.Tensor, dtype: torch.dtype | None = None):
    # Don't change dtype for integer tensors
    if tensor.dtype in (torch.int32, torch.int64, torch.long, torch.bool):
        return tensor.to(device=self.device)
    dtype = dtype or self.get_dtype()
    return tensor.to(device=self.device, dtype=dtype)
```

---

## Category: Memory Management

### Clear GPU Cache on Model Unload

**What to check**: Free GPU memory when unloading models

**Search pattern**:
```bash
rg "empty_cache|cuda\.empty_cache|mps\.empty_cache" --type py runtimes/universal/
```

**Pass criteria**:
- torch.cuda.empty_cache() called after unload
- torch.mps.empty_cache() called on Apple Silicon
- Model moved to CPU before clearing references

**Severity**: Critical

**Good pattern** (from models/base.py):
```python
async def unload(self) -> None:
    # Move model to CPU to free GPU memory
    if self.model is not None and hasattr(self.model, "to"):
        self.model = self.model.to("cpu")

    # Clear references
    self.model = None
    self.tokenizer = None

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Clear MPS cache
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
```

---

### Use torch.no_grad() for Inference

**What to check**: Disable gradient computation during inference

**Search pattern**:
```bash
rg "with torch\.no_grad\(\):" --type py runtimes/universal/models/
```

**Pass criteria**:
- All inference code wrapped in torch.no_grad()
- No gradient computation for embeddings, generation, classification

**Severity**: Critical

**Good pattern**:
```python
async def embed(self, texts: list[str], normalize: bool = True):
    with torch.no_grad():
        model_output = self.model(**encoded)
    embeddings = self._mean_pooling(model_output, attention_mask)
    return embeddings.cpu().tolist()
```

**Recommendation**: torch.no_grad() reduces memory usage and speeds up inference

---

### Return Tensors to CPU Before Python Conversion

**What to check**: Move tensors to CPU before .tolist() or numpy conversion

**Search pattern**:
```bash
rg "\.tolist\(\)|\.numpy\(\)" --type py runtimes/universal/models/
```

**Pass criteria**:
- Always call .cpu() before .tolist() or .numpy()
- Prevents GPU memory from being held by Python objects

**Severity**: High

**Good pattern**:
```python
# Correct - move to CPU first
embeddings = F.normalize(embeddings, p=2, dim=1)
return embeddings.cpu().tolist()

# Incorrect - GPU tensor leaked to Python
return embeddings.tolist()  # May fail on MPS/CUDA
```

---

## Category: Model Loading

### Specify device_map for CUDA

**What to check**: Use device_map="auto" for multi-GPU and efficient loading

**Search pattern**:
```bash
rg "device_map=" --type py runtimes/universal/models/
```

**Pass criteria**:
- device_map="auto" for CUDA devices
- None for CPU/MPS (handle manually)

**Severity**: Medium

**Good pattern**:
```python
self.model = AutoModelForCausalLM.from_pretrained(
    self.model_id,
    dtype=dtype,
    device_map="auto" if self.device == "cuda" else None,
    trust_remote_code=True,
)

if self.device != "cuda":
    self.model = self.model.to(self.device)
```

---

### Use torch_dtype Parameter

**What to check**: Pass dtype to from_pretrained() for efficient loading

**Search pattern**:
```bash
rg "torch_dtype=|dtype=" --type py runtimes/universal/models/
```

**Pass criteria**:
- torch_dtype set in from_pretrained() for GPU devices
- Omit for CPU to use full precision

**Severity**: Medium

**Good pattern**:
```python
model_kwargs = {
    "trust_remote_code": True,
    "token": self.token,
}

# Only set torch_dtype for GPU devices
if self.device != "cpu":
    model_kwargs["torch_dtype"] = dtype

self.model = AutoModel.from_pretrained(self.model_id, **model_kwargs)
```

---

## Category: Platform Optimizations

### Apply Platform-Specific Optimizations

**What to check**: Enable optimizations for each platform

**Search pattern**:
```bash
rg "enable_attention_slicing|enable_xformers|enable_model_cpu_offload" --type py runtimes/universal/
```

**Pass criteria**:
- MPS: attention_slicing (reduces memory pressure)
- CUDA: xformers memory efficient attention
- CUDA: model CPU offload for large models

**Severity**: Medium

**Good pattern**:
```python
def apply_optimizations(self):
    if self.device == "mps":
        self.pipe.enable_attention_slicing()
    elif self.device == "cuda":
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers not available
```

---

### Handle MPS Limitations

**What to check**: Work around MPS 4GB buffer limit and other issues

**Search pattern**:
```bash
rg "TRANSFORMERS_SKIP_MPS|mps.*limit|4GB" --type py runtimes/universal/
```

**Pass criteria**:
- Environment variable to skip MPS if needed
- Warning about MPS limitations logged
- force_float32 option for incompatible models

**Severity**: Medium

**Good pattern**:
```python
if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    skip_mps = os.environ.get("TRANSFORMERS_SKIP_MPS", "").lower() in ("1", "true")
    if skip_mps:
        return "cpu"
    logger.warning(
        "MPS has a 4GB temporary buffer limit. "
        "Set TRANSFORMERS_SKIP_MPS=1 to use CPU if you encounter errors."
    )
    return "mps"
```

---

## Category: Thread Safety

### ThreadPoolExecutor for Blocking Operations

**What to check**: Run blocking llama-cpp operations in thread pool

**Search pattern**:
```bash
rg "ThreadPoolExecutor|run_in_executor" --type py runtimes/universal/
```

**Pass criteria**:
- Thread pool created with max_workers=1 (serialized inference)
- Blocking Llama() calls run via executor
- Executor shutdown on model unload

**Severity**: High

**Good pattern** (from models/gguf_language_model.py):
```python
def __init__(self, model_id: str, device: str, ...):
    self._executor = ThreadPoolExecutor(max_workers=1)

async def load(self) -> None:
    loop = asyncio.get_running_loop()
    self.llama = await loop.run_in_executor(self._executor, _load_model)

async def unload(self) -> None:
    self.llama = None
    if hasattr(self, "_executor"):
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._executor = ThreadPoolExecutor(max_workers=1)
```

---

### Proper Executor Cleanup

**What to check**: Shutdown thread pool executors properly

**Search pattern**:
```bash
rg "shutdown\(wait=" --type py runtimes/universal/
```

**Pass criteria**:
- shutdown(wait=True) for graceful cleanup
- cancel_futures=True to stop pending work
- Create new executor if model might be reloaded

**Severity**: High

**Good pattern**:
```python
async def unload(self) -> None:
    if hasattr(self, "_executor"):
        self._executor.shutdown(wait=True, cancel_futures=True)
        # Recreate for potential future use
        self._executor = ThreadPoolExecutor(max_workers=1)
```
