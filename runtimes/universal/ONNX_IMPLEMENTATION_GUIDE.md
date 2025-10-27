# ONNX Implementation Guide - Step by Step

This guide walks through implementing ONNX support in the Universal Runtime.

## Quick Answer: What Changes?

### For Production Deployment

**99% of your code stays the same.** You just need to:

1. Add ONNX dependencies
2. Modify model loading (5-10 lines per model class)
3. Set environment variable

That's it! The API interface remains identical.

## Implementation Steps

### Step 1: Add Dependencies (5 minutes)

```toml
# pyproject.toml
[project.dependencies]
...
# Add these:
onnxruntime-gpu = ">=1.16.0"  # or onnxruntime for CPU-only
optimum = {version = ">=1.15.0", extras = ["onnxruntime"]}
```

```bash
cd runtimes/universal
uv sync
```

### Step 2: Modify BaseModel (10 minutes)

```python
# models/base.py
class BaseModel(ABC):
    def __init__(self, model_id: str, device: str):
        self.model_id = model_id
        self.device = device
        self.backend = os.getenv("RUNTIME_BACKEND", "pytorch")  # ← Add this
        ...
```

### Step 3: Update EncoderModel (30 minutes)

This is your highest-value target for ONNX.

```python
# models/encoder_model.py
from optimum.onnxruntime import ORTModelForFeatureExtraction

class EncoderModel(BaseModel):
    async def load(self):
        if self.backend == "onnx":
            await self._load_onnx()
        else:
            await self._load_pytorch()

    async def _load_pytorch(self):
        # Your existing load code here
        logger.info(f"Loading PyTorch encoder: {self.model_id}")
        self.model = AutoModel.from_pretrained(...)
        ...

    async def _load_onnx(self):
        # New ONNX loading code
        logger.info(f"Loading ONNX encoder: {self.model_id}")

        provider = "CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_id,
            export=True,  # Auto-converts to ONNX
            provider=provider,
        )
        ...
```

### Step 4: Test Both Backends (10 minutes)

```bash
# Test PyTorch (default)
RUNTIME_BACKEND=pytorch uv run python -m pytest tests/test_encoder_model.py -v

# Test ONNX
RUNTIME_BACKEND=onnx uv run python -m pytest tests/test_encoder_model.py -v
```

### Step 5: Benchmark Performance (15 minutes)

```python
# Create benchmark script
import asyncio
import time
from models.encoder_model import EncoderModel

async def benchmark():
    texts = ["Hello world"] * 100

    for backend in ["pytorch", "onnx"]:
        model = EncoderModel("sentence-transformers/all-MiniLM-L6-v2", "cpu")
        model.backend = backend
        await model.load()

        start = time.perf_counter()
        for _ in range(10):
            await model.embed(texts)
        elapsed = time.perf_counter() - start

        print(f"{backend}: {elapsed:.2f}s")

asyncio.run(benchmark())
```

Expected results:
- PyTorch: ~2.5s
- ONNX: ~0.8s
- **Speedup: 3x** 🚀

### Step 6: Deploy with Feature Flag (5 minutes)

```yaml
# docker-compose.yml
services:
  universal-runtime:
    environment:
      - RUNTIME_BACKEND=onnx  # ← Just set this!
      - ONNX_PROVIDER=CUDAExecutionProvider
```

## What About Other Model Types?

### CausalLM (Text Generation)

**Effort:** Medium (1-2 hours)
**Speedup:** 1.3-1.5x
**Worth it?** Yes if you do high-volume text generation

```python
from optimum.onnxruntime import ORTModelForCausalLM

async def _load_onnx(self):
    self.model = ORTModelForCausalLM.from_pretrained(
        self.model_id,
        export=True,
        provider=self._get_provider(),
    )
```

### VisionModel (Image Classification)

**Effort:** Medium (1-2 hours)
**Speedup:** 2x
**Worth it?** Yes for high-throughput classification

```python
from optimum.onnxruntime import ORTModelForImageClassification

async def _load_onnx(self):
    self.model = ORTModelForImageClassification.from_pretrained(
        self.model_id,
        export=True,
        provider=self._get_provider(),
    )
```

### DiffusionModel (Image Generation)

**Effort:** Very High (days)
**Speedup:** Variable
**Worth it?** **NO - Keep PyTorch**

Diffusion models are incredibly complex for ONNX:
- Multiple sub-models (UNet, VAE, text encoder)
- Dynamic execution graphs
- Custom schedulers
- Limited tooling support

**Recommendation:** Use PyTorch + `torch.compile()` instead:

```python
# Better approach for diffusion
self.pipe = StableDiffusionPipeline.from_pretrained(...)
self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead")
# 1.5-2x speedup with zero conversion effort!
```

### AudioModel (Whisper)

**Effort:** Medium-High (2-4 hours)
**Speedup:** 1.3x
**Worth it?** Maybe - depends on audio volume

Some Whisper ONNX models exist but conversion can be tricky.

### MultimodalModel (BLIP, LLaVA)

**Effort:** Very High
**Speedup:** Unknown
**Worth it?** **NO - Keep PyTorch**

Limited tooling support, complex architecture.

## Production Deployment Checklist

### For Immediate Impact (Start Here)

1. ✅ **EncoderModel → ONNX** (Critical for RAG, easy win)
   - Effort: 30 minutes
   - Speedup: 3x
   - Impact: High

2. ✅ **Add Backend Environment Variable**
   ```bash
   export RUNTIME_BACKEND=onnx
   ```

3. ✅ **Monitor Performance**
   - Track latency before/after
   - Verify accuracy (embeddings should match within 1e-4)

### Phase 2 (If Needed)

4. ⚠️ **CausalLM → ONNX** (If doing lots of text generation)
   - Effort: 1-2 hours
   - Speedup: 1.4x
   - Impact: Medium

5. ⚠️ **VisionModel → ONNX** (If doing high-volume classification)
   - Effort: 1-2 hours
   - Speedup: 2x
   - Impact: Medium

### Don't Bother

6. ❌ **DiffusionModel** - Stay PyTorch, use `torch.compile()`
7. ❌ **MultimodalModel** - Stay PyTorch

## Common Issues & Solutions

### Issue 1: "Model export failed"

**Cause:** Model has custom operations not supported by ONNX

**Solution:**
```python
# Add fallback in load method
try:
    await self._load_onnx()
except Exception as e:
    logger.warning(f"ONNX load failed: {e}, falling back to PyTorch")
    await self._load_pytorch()
```

### Issue 2: "Accuracy differs between backends"

**Cause:** Numerical precision differences

**Solution:**
```python
# Test both backends produce similar results
pytorch_emb = await pytorch_model.embed(texts)
onnx_emb = await onnx_model.embed(texts)

diff = np.abs(np.array(pytorch_emb) - np.array(onnx_emb))
assert diff.max() < 1e-3, f"Embeddings differ too much: {diff.max()}"
```

### Issue 3: "ONNX is slower than PyTorch"

**Causes:**
1. Wrong provider (using CPU instead of CUDA)
2. Not enough warmup iterations
3. Model too small to benefit

**Solutions:**
```python
# 1. Check provider
assert "CUDA" in str(self.model.providers), "Should use CUDA"

# 2. Add warmup
for _ in range(5):
    await model.embed(["warmup"])

# 3. Benchmark properly
# Only convert if you see >1.5x speedup in your workload
```

### Issue 4: "MPS (Apple Silicon) support?"

**Answer:** ONNX Runtime doesn't support MPS yet.

**Workaround:**
```python
if self.device == "mps" and self.backend == "onnx":
    logger.warning("ONNX doesn't support MPS, using CPU")
    self.device = "cpu"
```

For Apple Silicon, PyTorch with MPS is often faster than ONNX CPU anyway.

## Performance Monitoring

Add telemetry to track backend performance:

```python
import time

async def embed(self, texts: List[str]) -> List[List[float]]:
    start = time.perf_counter()
    result = await self._embed_impl(texts)
    latency = time.perf_counter() - start

    # Log metrics
    logger.info(
        f"Embedding latency: {latency*1000:.2f}ms "
        f"(backend={self.backend}, batch_size={len(texts)})"
    )

    return result
```

## Cost-Benefit Analysis

### EncoderModel (Embeddings)

**Investment:** 30 minutes
**Return:** 3x faster, critical for RAG
**ROI:** 🌟🌟🌟🌟🌟 **Excellent**

**Do it immediately.**

### CausalLM (Text Generation)

**Investment:** 1-2 hours
**Return:** 1.4x faster
**ROI:** 🌟🌟🌟 **Good if high volume**

**Do it if:**
- Processing >1000 requests/day
- Latency is critical
- Running on GPU

### VisionModel (Classification)

**Investment:** 1-2 hours
**Return:** 2x faster
**ROI:** 🌟🌟🌟 **Good if high volume**

Similar to CausalLM.

### DiffusionModel (Image Gen)

**Investment:** Days
**Return:** Unpredictable
**ROI:** 🌟 **Poor**

**Don't do it.** Use `torch.compile()` instead for easy wins.

## Summary: What You Should Do

### For LlamaFarm (RAG-focused)

**Priority 1: EncoderModel**
```bash
# 1. Add dependencies
uv pip install optimum[onnxruntime] onnxruntime-gpu

# 2. Modify EncoderModel (see example above)

# 3. Deploy
export RUNTIME_BACKEND=onnx
docker-compose up
```

**Expected impact:**
- Embedding latency: 15ms → 5ms
- RAG query throughput: 3x higher
- Cost savings: Run on smaller GPU or more instances on same hardware

**Time investment:** 1-2 hours
**Impact:** Immediate and significant

### Priority 2-3 (Optional)

Only do these if you have high volume and latency requirements:
- CausalLM for text generation
- VisionModel for classification

### Don't Bother

- Diffusion models → Use PyTorch + `torch.compile()`
- Multimodal models → Stay PyTorch
- Audio models → Stay PyTorch (unless very high volume)

## Next Steps

1. ✅ Read `ONNX_STRATEGY.md` for architectural overview
2. ✅ Review `encoder_model_onnx_example.py` for working code
3. ✅ Start with EncoderModel (30 min)
4. ✅ Benchmark and verify (15 min)
5. ✅ Deploy to staging with `RUNTIME_BACKEND=onnx`
6. ✅ Monitor performance for 1-2 days
7. ✅ Roll out to production

## Questions?

Common questions answered:

**Q: Will my API change?**
A: No, it's 100% transparent to users.

**Q: Can I mix backends?**
A: Yes! `EncoderModel=onnx`, `DiffusionModel=pytorch`

**Q: What about Apple Silicon?**
A: ONNX CPU mode works, but PyTorch MPS might be faster.

**Q: Does this work with quantization?**
A: Yes! ONNX makes quantization even easier with Optimum.

**Q: Can I pre-convert models?**
A: Yes: `optimum-cli export onnx --model <model_id> <output_dir>`

**Q: What if model conversion fails?**
A: Add automatic fallback to PyTorch (see Issue 1 above).
