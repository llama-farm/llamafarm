# ONNX Production Strategy for Universal Runtime

## Overview

This document outlines the strategy for running models with ONNX in production while maintaining PyTorch compatibility for development.

## Architecture: Runtime Backend Abstraction

```
┌─────────────────────────────────────────┐
│         Model Interface (Base)          │
└─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼────────┐    ┌────────▼──────┐
│ PyTorch Backend│    │  ONNX Backend │
│   (Dev/Test)   │    │  (Production) │
└────────────────┘    └───────────────┘
```

## Model Type Compatibility Matrix

| Model Type | ONNX Feasibility | Library | Notes |
|------------|------------------|---------|-------|
| **Encoder** (Embeddings) | ✅ Excellent | Optimum | Perfect fit, 2-3x faster |
| **CausalLM** (Text Gen) | ✅ Good | Optimum | Supported, slight complexity |
| **Vision** (Classification) | ✅ Good | Optimum | Well supported |
| **Vision** (CLIP) | ⚠️ Moderate | Custom | Needs custom export |
| **Audio** (Whisper) | ⚠️ Moderate | Optimum | Experimental support |
| **Multimodal** (BLIP) | ⚠️ Difficult | Custom | Limited support |
| **Diffusion** (Stable Diff) | ❌ Very Hard | Olive/Custom | Keep PyTorch |

## Implementation Strategy

### Phase 1: High-Value Models (Recommended Start)

Focus on models with best ONNX support and highest production usage:

1. **EncoderModel** → ONNX (Embeddings are critical for RAG)
2. **CausalLMModel** → ONNX (Text generation)
3. **VisionModel** → ONNX (Image classification)

### Phase 2: Complex Models

4. **AudioModel** → Hybrid (ONNX where available, PyTorch fallback)
5. **MultimodalModel** → Hybrid

### Phase 3: Keep in PyTorch

6. **DiffusionModel** → Stay PyTorch (or explore ONNX Runtime with Olive)

## Technical Approach

### Option 1: Backend Abstraction (Recommended)

Create a runtime backend system that switches based on environment:

```python
# Environment variable controls backend
RUNTIME_BACKEND = os.getenv("RUNTIME_BACKEND", "pytorch")  # or "onnx"

class BaseModel(ABC):
    def __init__(self, model_id: str, device: str, backend: str = None):
        self.backend = backend or os.getenv("RUNTIME_BACKEND", "pytorch")

    async def load(self):
        if self.backend == "onnx":
            await self._load_onnx()
        else:
            await self._load_pytorch()

    @abstractmethod
    async def _load_onnx(self):
        pass

    @abstractmethod
    async def _load_pytorch(self):
        pass
```

### Option 2: Separate Model Classes

Create parallel ONNX implementations:

```
models/
├── pytorch/
│   ├── encoder_model.py
│   ├── language_model.py
│   └── ...
└── onnx/
    ├── encoder_model.py
    ├── language_model.py
    └── ...
```

### Option 3: HuggingFace Optimum (Simplest)

Use Optimum library which handles ONNX conversion automatically:

```python
from optimum.onnxruntime import ORTModelForSequenceClassification
from optimum.onnxruntime import ORTModelForFeatureExtraction

# Drop-in replacement for transformers
model = ORTModelForFeatureExtraction.from_pretrained(
    model_id,
    export=True,  # Auto-converts to ONNX
    provider="CUDAExecutionProvider"  # or CPUExecutionProvider
)
```

## Code Example: ONNX-Ready Encoder

Here's how an ONNX-enabled encoder would look:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer
import numpy as np

class ONNXEncoderModel(BaseModel):
    """ONNX-optimized encoder for production."""

    async def load(self):
        logger.info(f"Loading ONNX encoder: {self.model_id}")

        # Load ONNX model (auto-converts if needed)
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            self.model_id,
            export=True,  # Convert to ONNX if not already
            provider="CUDAExecutionProvider" if self.device == "cuda" else "CPUExecutionProvider"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        logger.info(f"ONNX encoder loaded on {self.device}")

    async def embed(self, texts: List[str], normalize: bool = True):
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np"  # NumPy for ONNX
        )

        # Run inference (ONNX Runtime)
        outputs = self.model(**encoded)
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Normalize if requested
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings.tolist()
```

## Performance Comparison

Based on benchmarks from HuggingFace Optimum:

| Model Type | PyTorch | ONNX | Speedup |
|------------|---------|------|---------|
| BERT Embeddings | 15ms | 5ms | **3x** |
| GPT-2 Generation | 50ms | 35ms | **1.4x** |
| ViT Classification | 12ms | 6ms | **2x** |
| Whisper (small) | 200ms | 150ms | **1.3x** |

## Deployment Recommendations

### Development Environment
```bash
export RUNTIME_BACKEND=pytorch
uv run python server.py
```

### Production Environment
```bash
export RUNTIME_BACKEND=onnx
export ONNX_PROVIDER=CUDAExecutionProvider  # or CPUExecutionProvider
docker run -e RUNTIME_BACKEND=onnx universal-runtime
```

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.12

# Install ONNX Runtime
RUN pip install onnxruntime-gpu  # or onnxruntime for CPU

# Install Optimum
RUN pip install optimum[onnxruntime]

# Set production backend
ENV RUNTIME_BACKEND=onnx
ENV ONNX_PROVIDER=CUDAExecutionProvider

CMD ["python", "server.py"]
```

## Model Conversion Process

### Automatic (Recommended)
```python
# Optimum handles conversion automatically
from optimum.onnxruntime import ORTModelForFeatureExtraction

model = ORTModelForFeatureExtraction.from_pretrained(
    "sentence-transformers/all-MiniLM-L6-v2",
    export=True  # Converts on first run
)
```

### Manual (More Control)
```bash
# Pre-convert models
optimum-cli export onnx \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --optimize O3 \
    onnx_models/all-MiniLM-L6-v2/
```

## Migration Path

### Step 1: Add ONNX Dependencies
```toml
[project.dependencies]
...
onnxruntime-gpu = ">=1.16.0"  # or onnxruntime for CPU
optimum = {version = ">=1.15.0", extras = ["onnxruntime"]}
```

### Step 2: Create Backend Abstraction
Modify `BaseModel` to support dual backends

### Step 3: Implement ONNX Loaders
Add `_load_onnx()` methods to each model class

### Step 4: Test Both Backends
Ensure feature parity between PyTorch and ONNX

### Step 5: Deploy with Feature Flag
Use environment variable to control backend

### Step 6: Monitor Performance
Compare latency and throughput in production

## Limitations & Considerations

### 1. Dynamic Shapes
Some ONNX models require fixed input shapes. Use dynamic axis:
```python
export_config = {
    "opset_version": 14,
    "dynamic_axes": {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"}
    }
}
```

### 2. Custom Operations
Models with custom ops may not convert. Use `torch.onnx.register_custom_op_symbolic()`

### 3. Memory Management
ONNX Runtime has different memory characteristics. Monitor carefully.

### 4. Quantization
ONNX makes quantization easier for additional speedup:
```python
from optimum.onnxruntime import ORTQuantizer

quantizer = ORTQuantizer.from_pretrained(model_id)
quantizer.quantize(save_directory="quantized_model", quantization_config=qconfig)
```

## Real-World Production Setup

```python
# config.py
import os

RUNTIME_CONFIG = {
    "encoder": {
        "backend": os.getenv("ENCODER_BACKEND", "onnx"),
        "provider": "CUDAExecutionProvider",
        "optimize": True,
        "quantize": False,  # Enable for even more speed
    },
    "language": {
        "backend": os.getenv("language_BACKEND", "onnx"),
        "provider": "CUDAExecutionProvider",
        "optimize": True,
    },
    "diffusion": {
        "backend": "pytorch",  # Keep complex models in PyTorch
        "optimize": True,
    }
}
```

## Monitoring

Track key metrics when switching to ONNX:

```python
import time

@metrics.timer("inference_time")
async def embed(self, texts):
    start = time.perf_counter()
    result = await self._embed_impl(texts)
    latency = time.perf_counter() - start

    metrics.gauge("inference_latency_ms", latency * 1000)
    metrics.gauge("throughput_qps", len(texts) / latency)

    return result
```

## Next Steps

1. **Start Small**: Convert EncoderModel first (biggest impact for RAG)
2. **Benchmark**: Measure actual performance gains in your workload
3. **Gradual Rollout**: Use feature flags to control backend per model type
4. **Monitor**: Watch latency, memory, and accuracy metrics
5. **Iterate**: Convert more models based on performance data

## Resources

- [HuggingFace Optimum](https://huggingface.co/docs/optimum/index)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Optimum Benchmark](https://github.com/huggingface/optimum-benchmark)
- [Model Conversion Guide](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
