# Universal Runtime

An OpenAI-compatible API server for running any HuggingFace model locally without restrictions. The Universal Runtime provides a unified interface for text generation, embeddings, image generation, vision, audio, and multimodal models.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Supported Model Types](#supported-model-types)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Hardware Requirements](#hardware-requirements)
- [Testing](#testing)
- [Examples](#examples)
- [Documentation](#documentation)
- [Troubleshooting](#troubleshooting)

---

## Overview

The Universal Runtime is a FastAPI-based inference server that bridges HuggingFace's `transformers` and `diffusers` libraries with OpenAI-compatible APIs. It enables developers to:

- **Run any HuggingFace model locally** with zero configuration
- **Switch between models dynamically** without restarting the server
- **Use OpenAI SDKs** with local models (drop-in replacement)
- **Optimize automatically** for your hardware (Apple Silicon MPS, NVIDIA CUDA, CPU)
- **Support multimodal workflows** (text, images, audio, vision-language tasks)

### Why Universal Runtime?

- **Privacy**: Keep your data and models on-premises
- **Cost**: No API fees for model inference
- **Flexibility**: Use any model from HuggingFace without vendor lock-in
- **Development**: Test with tiny models, deploy with production models
- **Integration**: OpenAI-compatible endpoints work with existing tools

---

## Features

✅ **6 Model Categories**
- Text generation (CausalLM: GPT, Llama, Mistral, Qwen, Phi)
- Text embeddings & classification (Encoder: BERT, sentence-transformers)
- Image generation (Diffusion: Stable Diffusion, SDXL, FLUX)
- Image understanding (Vision: ViT, CLIP, DINOv2)
- Speech-to-text (Audio: Whisper, Wav2Vec2)
- Vision-language (Multimodal: BLIP, LLaVA, Florence)

✅ **Smart Hardware Detection**
- Auto-detects Apple Silicon (MPS), NVIDIA GPUs (CUDA), or CPU
- Platform-specific optimizations (Metal Performance Shaders, cuDNN)
- Configurable precision (FP32, FP16, BF16, INT8)

✅ **Developer Experience**
- Lazy model loading (models load on first request)
- Model caching (keeps frequently-used models in memory)
- Automatic model unloading (frees VRAM/RAM after 5 minutes of inactivity)
- Streaming responses for text generation
- Base64 and file path support for images/audio
- Comprehensive error messages and logging

✅ **Production Ready**
- OpenAI API compatibility (drop-in replacement)
- Async/await for concurrent requests
- Health and status endpoints
- Nx integration for LlamaFarm monorepo
- Comprehensive test suite

✅ **Advanced Features**
- **GGUF model support** with llama-cpp-python (see [GGUF Support](#gguf-model-support))
- ONNX runtime support (planned, see [ONNX_STRATEGY.md](./ONNX_STRATEGY.md))
- Custom schedulers for diffusion models
- Batch processing for embeddings
- Zero-shot classification with CLIP

---

## Supported Model Types

The Universal Runtime supports 6 major model categories. See [MODEL_TYPES.md](./MODEL_TYPES.md) for detailed information on each type.

| Model Type | API Endpoint | Example Models | Use Cases |
|------------|--------------|----------------|-----------|
| **CausalLM** | `/v1/chat/completions` | GPT-2, Llama, Mistral, Qwen, Phi | Text generation, chat, code completion |
| **Encoder** | `/v1/embeddings` | BERT, sentence-transformers, BGE | Semantic search, RAG, classification |
| **Diffusion** | `/v1/images/generations` | Stable Diffusion, SDXL, FLUX | Image generation, editing, inpainting |
| **Vision** | `/v1/vision/classify` | ViT, CLIP, DINOv2, ResNet | Image classification, zero-shot |
| **Audio** | `/v1/audio/transcriptions` | Whisper, Wav2Vec2 | Speech-to-text, translation |
| **Multimodal** | `/v1/multimodal/caption` | BLIP, LLaVA, Florence | Image captioning, visual QA |

**Quick Model Recommendations:**
- **RAG Embeddings**: `BAAI/bge-base-en-v1.5` or `nomic-ai/nomic-embed-text-v1.5`
- **RAG Embeddings (GGUF/Quantized)**: `nomic-ai/nomic-embed-text-v1.5-GGUF` or `mixedbread-ai/mxbai-embed-xsmall-v1`
- **Chat (Quality)**: `meta-llama/Llama-3.1-8B-Instruct`
- **Chat (Speed)**: `microsoft/phi-2` or `Qwen/Qwen2.5-0.5B-Instruct`
- **Chat (GGUF/Quantized)**: `unsloth/Qwen3-4B-GGUF` or `unsloth/Llama-3.2-3B-Instruct-GGUF`
- **Image Generation**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Speech Recognition**: `openai/whisper-large-v3`

---

## GGUF Model Support

The Universal Runtime now supports GGUF quantized models via llama-cpp-python, providing significantly improved performance and reduced memory usage for local inference.

### What are GGUF Models?

GGUF (GPT-Generated Unified Format) is a quantized model format that offers:

- **50-75% smaller file sizes** through 4-bit/8-bit quantization
- **2-3x faster inference** on Apple Silicon (Metal acceleration)
- **Significantly lower memory usage** - run larger models on the same hardware
- **Optimized CPU inference** - better performance than standard PyTorch on CPU
- **Automatic format detection** - no configuration changes needed

### Using GGUF Models

GGUF models are automatically detected and loaded with llama-cpp-python. Simply specify a GGUF model ID:

```python
# In your configuration or API request
model = "unsloth/Qwen3-4B-GGUF"
```

The runtime automatically:
1. Detects the GGUF format from model files
2. Uses llama-cpp-python for optimized inference
3. Configures appropriate hardware acceleration (Metal/CUDA/CPU)

### Selecting GGUF Quantization

Many GGUF model repositories (like `unsloth/Qwen3-1.7B-GGUF`) contain multiple quantization variants (Q4_K_M, Q8_0, F16, etc.). The Universal Runtime intelligently selects and downloads only the quantization you need, saving disk space.

#### Automatic Selection (Default)

By default, the runtime selects **Q4_K_M** quantization, which offers the best balance of size, speed, and quality:

```python
# Automatically selects Q4_K_M if available
model = "unsloth/Qwen3-1.7B-GGUF"
```

**Selection priority order:**
1. Q4_K_M (default - best balance)
2. Q4_K, Q5_K_M, Q5_K
3. Q8_0 (higher quality, larger)
4. Q6_K (between Q5 and Q8)
5. Q4_K_S (smaller Q4 variant)
6. Q5_K_S (smaller Q5 variant)
7. Q3_K_M, Q2_K (lower quality, smaller)
8. F16 (full precision, very large)

#### Manual Quantization Selection

Specify your preferred quantization in `llamafarm.yaml` or via API:

**In llamafarm.yaml:**
```yaml
runtime:
  models:
    - name: default
      provider: universal
      model: unsloth/Qwen3-1.7B-GGUF:Q8_0  # Use higher quality 8-bit quantization
```

**Via OpenAI-compatible API:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1")
response = client.chat.completions.create(
    model="unsloth/Qwen3-1.7B-GGUF:Q8_0",  # Specify quantization in model name
    messages=[{"role": "user", "content": "Hello!"}]
)
```

**Common quantization types:**
- **Q4_K_M**: 4-bit, medium variant (default, ~50-60% size reduction)
- **Q5_K_M**: 5-bit, medium variant (~40-50% size reduction)
- **Q8_0**: 8-bit (minimal quality loss, ~50% size vs F16)
- **F16**: Full 16-bit precision (largest, highest quality)

**Benefits:**
- ✅ Only downloads the specific quantization you need
- ✅ Saves disk space (no unused variants)
- ✅ Explicit control over quality/size trade-off
- ✅ Works with any GGUF model repository

### Recommended GGUF Models

| Model | Size | Quantization | Best For |
|-------|------|--------------|----------|
| `unsloth/Qwen3-0.6B-GGUF` | ~400MB | Q4_K_M | Fast responses, testing |
| `unsloth/Qwen3-1.7B-GGUF` | ~1GB | Q4_K_M | Balanced performance |
| `unsloth/Qwen3-4B-GGUF` | ~2.5GB | Q4_K_M | High quality (recommended) |
| `unsloth/Qwen3-8B-GGUF` | ~5GB | Q4_K_M | Best quality |
| `unsloth/Llama-3.2-3B-Instruct-GGUF` | ~2GB | Q4_K_M | Instruction-tuned |

### Platform-Specific Acceleration

The Universal Runtime automatically configures llama-cpp-python for your hardware:

- **macOS (Apple Silicon)**: Uses Metal acceleration for 2-3x faster inference
- **Linux/Windows with NVIDIA GPU**: Uses CUDA acceleration
- **Other platforms**: Falls back to highly optimized CPU inference

No configuration needed - acceleration is detected and enabled automatically!

### Context Size Configuration

The Universal Runtime intelligently determines the optimal context window size for GGUF models using a three-tier priority system:

#### 1. Priority System

Context size is determined in this order (highest to lowest):

1. **User Configuration** (via `llamafarm.yaml` → `extra_body.n_ctx` or API `extra_body`)
   - Explicit value from project configuration or API request
   - Highest priority - respects user's explicit choice

2. **Computed Maximum** (based on available memory)
   - Automatically calculated using available VRAM (CUDA) or RAM (MPS/CPU)
   - Accounts for model size and memory overhead
   - Prevents out-of-memory errors

3. **Pattern Match Defaults** (from `config/model_context_defaults.yaml`)
   - Known defaults for model families (e.g., Qwen2.5 → 32k, Llama-3 → 8k)
   - Uses Unix shell-style wildcard patterns

4. **Fallback Default** (2048 tokens)
   - Conservative safe value for unknown models

#### 2. Memory-Based Computation & Model Training Context

The runtime automatically:
- **Reads the model's training context** (`n_ctx_train`) from GGUF metadata using the `gguf` library
- **Computes maximum safe context size** based on available memory

**Priority for determining context size:**
1. **User Configuration** (explicit value in config/API)
2. **Model's n_ctx_train** (what the model was trained for) - NEW! 🎯
3. **Pattern Match Defaults** (from config file)
4. **Computed Max from Memory** (hardware limitation)
5. **Fallback Default** (2048 tokens)

All choices are automatically capped by available memory to prevent OOM errors.

**Memory calculation formula:**
```
usable_memory = (available_memory * 0.8) - model_file_size
max_context = usable_memory / (bytes_per_token * overhead_factor)
```

- **Memory factor**: 80% of available memory by default (configurable in `model_context_defaults.yaml`)
- **Automatic capping**: If any value exceeds computed maximum, it falls back to the maximum with a warning
- **Smart rounding**: Results are rounded to powers of 2 (512, 1024, 2048, 4096, etc.)

#### 3. Configuration File

Default context sizes are defined in `runtimes/universal/config/model_context_defaults.yaml`:

```yaml
memory_usage_factor: 0.8  # Use 80% of available memory

model_defaults:
  # Exact match (highest priority)
  - pattern: "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF"
    n_ctx: 32768

  # Pattern matches for families
  - pattern: "*/Qwen2.5-*-GGUF"
    n_ctx: 32768

  - pattern: "*/Llama-3*-GGUF"
    n_ctx: 8192

  # Fallback
  - pattern: "*"
    n_ctx: 2048
```

You can edit this file to add custom defaults for your models.

#### 4. Specifying Context Size

**Via LlamaFarm configuration** (`llamafarm.yaml`):

```yaml
runtime:
  models:
    - name: my-model
      provider: universal
      model: unsloth/Qwen3-4B-GGUF
      extra_body:
        n_ctx: 16384  # Explicit context size
```

**Via API request** (OpenAI SDK):

```python
response = client.chat.completions.create(
    model="unsloth/Qwen3-4B-GGUF",
    messages=[...],
    extra_body={"n_ctx": 16384}  # Runtime-specific parameter
)
```

**Via direct API call**:

```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3-4B-GGUF",
    "messages": [...],
    "n_ctx": 16384
  }'
```

#### 5. Warnings and Logs

Context size warnings are logged to stderr when:
- Requested size exceeds computed maximum (falls back to maximum)
- No pattern match found (using fallback default)
- Low memory detected (using minimal context)

Example log output:
```
INFO: Using context size: 8192
WARNING: Requested context size 32768 exceeds computed maximum 16384 based on available memory (12.50 GB). Using 16384 instead.
```

#### 6. Best Practices

- **Let it auto-detect**: For most use cases, omit `n_ctx` and let the runtime compute the optimal size
- **Check logs**: Monitor stderr for warnings about memory constraints
- **Start conservative**: If unsure, start with smaller context sizes and increase if needed
- **Monitor memory**: Use system monitoring tools to verify memory usage during inference
- **Test before deploying**: Validate context sizes work reliably with your hardware before production use

### Example Usage

```bash
# Chat completions with GGUF model
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3-4B-GGUF",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "stream": true
  }'

# With custom context window size (for longer conversations)
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3-4B-GGUF",
    "messages": [
      {"role": "user", "content": "Long conversation..."}
    ],
    "n_ctx": 8192
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11540/v1", api_key="universal")

# Basic usage
response = client.chat.completions.create(
    model="unsloth/Qwen3-4B-GGUF",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end="")

# With custom context window
response = client.chat.completions.create(
    model="unsloth/Qwen3-4B-GGUF",
    messages=[{"role": "user", "content": "Long conversation..."}],
    extra_body={"n_ctx": 8192}  # Larger context window
)
```

### Performance Comparison

Typical performance gains with GGUF vs. standard transformers:

| Hardware | Standard FP16 | GGUF Q4_K_M | Speedup |
|----------|---------------|-------------|---------|
| Apple M2 Max | ~15 tokens/s | ~40 tokens/s | **2.7x** |
| NVIDIA RTX 4090 | ~60 tokens/s | ~85 tokens/s | **1.4x** |
| Intel Core i9 (CPU) | ~3 tokens/s | ~12 tokens/s | **4x** |

*Results with Qwen3-4B on typical chat prompts*

### Technical Details

- **Format Detection**: Automatic - checks for `.gguf` files in model repository
- **Cache Sharing**: GGUF and transformers models share the same HuggingFace cache
- **API Compatibility**: Same endpoints work for both GGUF and transformers models
- **Streaming**: Full streaming support with token-by-token generation
- **Context Window**: Configurable via `n_ctx` parameter (default: 2048, max depends on model)
  - Larger context windows allow for longer conversations
  - Each context size is cached separately for optimal performance
  - Model must support the requested context size

### GGUF Embedding Models

The Universal Runtime now supports **GGUF quantized embedding models** for text embeddings, providing the same performance benefits as GGUF language models.

#### Why Use GGUF Embedding Models?

GGUF embedding models offer significant advantages over standard transformers-based embedding models:

- **50-75% smaller file sizes** - Reduced storage requirements
- **2-3x faster inference** on Apple Silicon with Metal acceleration
- **Lower memory usage** - Run larger embedding models on the same hardware
- **Optimized CPU inference** - Better performance on CPU-only systems
- **Automatic format detection** - No code changes needed

#### Recommended GGUF Embedding Models

| Model | Size | Dimensions | Languages | Best For |
|-------|------|------------|-----------|----------|
| `nomic-ai/nomic-embed-text-v1.5-GGUF` | ~250MB | 768 | 100+ | Multilingual RAG, semantic search |
| `mixedbread-ai/mxbai-embed-xsmall-v1` | ~30MB | 384 | English | Fast embeddings, resource-constrained |
| `CompendiumLabs/bge-base-en-v1.5-gguf` | ~130MB | 768 | English | High-quality English embeddings |

#### Example Usage

**cURL:**

```bash
# Generate embeddings with GGUF model
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nomic-ai/nomic-embed-text-v1.5-GGUF",
    "input": ["Hello world", "How are you?"]
  }'
```

**Python (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:11540/v1", api_key="universal")

# Generate embeddings with GGUF model
response = client.embeddings.create(
    model="nomic-ai/nomic-embed-text-v1.5-GGUF",
    input=["Hello world", "How are you?"]
)

# Access embeddings
embedding1 = response.data[0].embedding  # List of floats
embedding2 = response.data[1].embedding

print(f"Embedding dimension: {len(embedding1)}")
print(f"First 5 values: {embedding1[:5]}")
```

**LlamaFarm Configuration:**

```yaml
runtime:
  provider: universal
  model: nomic-ai/nomic-embed-text-v1.5-GGUF
  base_url: http://localhost:11540/v1

rag:
  databases:
    - name: main_db
      embedder:
        type: universal
        model: nomic-ai/nomic-embed-text-v1.5-GGUF
        dimensions: 768
```

#### Performance Comparison

Typical performance with GGUF vs. transformers embedding models:

| Hardware | Transformers FP16 | GGUF Q8_0 | Speedup |
|----------|-------------------|-----------|---------|
| Apple M3 Pro | ~1200 docs/s | ~3000 docs/s | **2.5x** |
| NVIDIA RTX 4090 | ~3500 docs/s | ~4800 docs/s | **1.4x** |
| Intel Core i9 (CPU) | ~150 docs/s | ~600 docs/s | **4x** |

*Results with nomic-embed-text-v1.5, batch size 32, 128-token documents*

#### Technical Details

- **Automatic Detection**: GGUF format is automatically detected from model files
- **L2 Normalization**: Embeddings are normalized by default (compatible with cosine similarity)
- **GPU Acceleration**: Automatically uses Metal (macOS) or CUDA (Linux/Windows) when available
- **Batch Processing**: Processes multiple texts efficiently in sequence
- **No Context Window Limits**: Embedding models have fixed input lengths (no `n_ctx` parameter needed)

---

## Installation

### Prerequisites

- **Python 3.11-3.13** (3.12 recommended, **NOT 3.14+**)
  - PyTorch 2.8.0 doesn't support Python 3.14 yet
  - Use `python3 --version` to check your version
- **[uv](https://github.com/astral-sh/uv)** package manager
- **8GB+ RAM** minimum (16GB+ recommended for larger models)
- **Optional**: Apple Silicon Mac (M1/M2/M3/M4) or NVIDIA GPU for acceleration

### Install Python 3.12 (if needed)

If you have Python 3.14 or another incompatible version:

```bash
# macOS
brew install python@3.12

# Linux (using pyenv)
pyenv install 3.12
pyenv local 3.12

# Or let uv manage Python for you
uv python install 3.12
```

### Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup the Runtime

**Method 1: Using Nx (from project root) - Recommended**

```bash
# Setup dependencies only:
nx sync universal-runtime

# Or just start (automatically runs setup first):
nx start universal-runtime
```

This automatically installs Python 3.12 and all dependencies!

**Method 2: Using uv directly**

```bash
# Navigate to the universal runtime directory
cd runtimes/universal

# If Python 3.12 isn't installed yet, install it (one-time):
uv python install 3.12

# Install dependencies
uv sync
```

The `pyproject.toml` constraint (`requires-python = ">=3.11,<3.14"`) ensures only compatible Python versions are used.

### Optional: Install xformers (for Diffusion optimization)

**GPU-only** - xformers requires CUDA and is not available for CPU-only installations.

```bash
# Only install on CUDA-enabled systems
uv pip install xformers
```

This significantly speeds up Stable Diffusion models on CUDA GPUs. xformers is installed manually (not via extras) to avoid lockfile conflicts with CPU-only PyTorch builds used in CI/testing.

---

## Quick Start

### 1. Start the Server

**From project root (recommended):**

```bash
nx start universal-runtime
```

**From runtimes/universal directory:**

```bash
# Using start script:
bash start.sh

# Or directly with uv:
uv run python server.py
```

**Server runs at:** `http://127.0.0.1:11540`

The server will:
- Auto-detect your hardware (MPS/CUDA/CPU)
- Load models on-demand when first requested
- Cache models in memory for subsequent requests
- Save generated images to `~/.llamafarm/outputs/images/`

### 2. Test the Server

```bash
# Check health
curl http://localhost:11540/health

# Generate text
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'

# Generate embeddings
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Hello world"
  }'

# Generate an image
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain lake at sunset",
    "model": "stabilityai/stable-diffusion-2-1",
    "size": "512x512"
  }'
```

**First request downloads the model** (~500MB-10GB depending on model), which takes 5-30 minutes. Subsequent requests use the cached model.

### 3. Use with OpenAI SDK

The Universal Runtime is compatible with OpenAI's Python SDK:

```python
import openai

# Point to local server
openai.api_base = "http://localhost:11540/v1"
openai.api_key = "not-used"  # Not required for local runtime

# Chat completion
response = openai.ChatCompletion.create(
    model="microsoft/phi-2",
    messages=[
        {"role": "user", "content": "Explain quantum computing in one sentence"}
    ]
)
print(response.choices[0].message.content)

# Embeddings
response = openai.Embedding.create(
    model="sentence-transformers/all-MiniLM-L6-v2",
    input="Hello world"
)
print(response.data[0].embedding)

# Image generation
response = openai.Image.create(
    prompt="a beautiful sunset over mountains",
    model="stabilityai/stable-diffusion-2-1",
    size="512x512"
)
print(response.data[0].url)
```

---

## API Endpoints

### Health & Status

#### `GET /health`

Check server health and hardware information.

**Response:**
```json
{
  "status": "healthy",
  "device": "mps",
  "device_info": {
    "type": "mps",
    "name": "Apple M3 Pro",
    "memory_allocated": "2.3 GB",
    "memory_reserved": "4.1 GB"
  },
  "loaded_models": ["microsoft/phi-2", "sentence-transformers/all-MiniLM-L6-v2"]
}
```

---

### Text Generation (CausalLM)

#### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint.

**Request:**
```json
{
  "model": "microsoft/phi-2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Explain quantum computing"}
  ],
  "temperature": 0.7,
  "max_tokens": 512,
  "stream": false
}
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1760640000,
  "model": "microsoft/phi-2",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Quantum computing is a revolutionary..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 50,
    "total_tokens": 65
  }
}
```

**Streaming:** Set `"stream": true` to receive Server-Sent Events.

---

### Embeddings (Encoder)

#### `POST /v1/embeddings`

OpenAI-compatible embeddings endpoint.

**Request:**
```json
{
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "input": ["Hello world", "How are you?"]
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    },
    {
      "object": "embedding",
      "embedding": [0.789, -0.012, ...],
      "index": 1
    }
  ],
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  }
}
```

---

### Image Generation (Diffusion)

#### `POST /v1/images/generations`

Generate images from text prompts.

**Request:**
```json
{
  "prompt": "a serene mountain lake at sunset, photorealistic, 8k",
  "model": "stabilityai/stable-diffusion-2-1",
  "size": "512x512",
  "n": 1,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "seed": 42,
  "negative_prompt": "blurry, low quality, distorted"
}
```

**Response:**
```json
{
  "created": 1760640000,
  "data": [
    {
      "url": "/Users/you/.llamafarm/outputs/images/stabilityai_stable-diffusion-2-1_20250127_143025_42_0.png"
    }
  ]
}
```

#### `POST /v1/images/edits`

Edit images using inpainting.

**Request:**
```json
{
  "prompt": "Add glowing holographic displays",
  "image": "<base64_encoded_image>",
  "mask": "<base64_encoded_mask>",
  "model": "stabilityai/stable-diffusion-2-inpainting",
  "size": "512x512",
  "num_inference_steps": 50,
  "guidance_scale": 7.5
}
```

#### `POST /v1/images/variations`

Create variations of an image (img2img).

**Request:**
```json
{
  "prompt": "Transform into a cyberpunk version",
  "image": "<base64_encoded_image>",
  "model": "stabilityai/stable-diffusion-2-1",
  "size": "512x512",
  "strength": 0.75,
  "num_inference_steps": 40,
  "guidance_scale": 7.5
}
```

---

### Vision (Classification & CLIP)

#### `POST /v1/vision/classify`

Classify images using vision models.

**Request:**
```json
{
  "model": "google/vit-base-patch16-224",
  "images": ["<base64_encoded_image>"],
  "top_k": 5
}
```

#### `POST /v1/vision/clip`

Zero-shot classification with CLIP.

**Request:**
```json
{
  "model": "openai/clip-vit-base-patch32",
  "images": ["<base64_encoded_image>"],
  "candidate_labels": ["dog", "cat", "bird", "car"]
}
```

---

### Audio (Speech-to-Text)

#### `POST /v1/audio/transcriptions`

Transcribe audio to text (OpenAI-compatible).

**Request:**
```json
{
  "file": "<base64_encoded_audio>",
  "model": "openai/whisper-large-v3",
  "language": "en",
  "response_format": "json"
}
```

**Response:**
```json
{
  "text": "This is the transcribed audio content..."
}
```

#### `POST /v1/audio/translations`

Translate audio to English.

**Request:**
```json
{
  "file": "<base64_encoded_audio>",
  "model": "openai/whisper-large-v3"
}
```

---

### Multimodal (Image Captioning & VQA)

#### `POST /v1/multimodal/caption`

Generate captions for images.

**Request:**
```json
{
  "model": "Salesforce/blip-image-captioning-base",
  "image": "<base64_encoded_image>",
  "max_length": 50
}
```

#### `POST /v1/multimodal/vqa`

Visual question answering.

**Request:**
```json
{
  "model": "Salesforce/blip-vqa-base",
  "image": "<base64_encoded_image>",
  "question": "What color is the car?"
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `UNIVERSAL_RUNTIME_HOST` | `127.0.0.1` | Server host |
| `UNIVERSAL_RUNTIME_PORT` | `11540` | Server port |
| `TRANSFORMERS_CACHE` | `~/.cache/huggingface` | Model cache directory |
| `HF_TOKEN` | None | HuggingFace token (for gated models) |
| `RUNTIME_BACKEND` | `pytorch` | Backend (future: `onnx`) |
| `MODEL_UNLOAD_TIMEOUT` | `300` | Seconds of inactivity before unloading models (5 minutes) |
| `CLEANUP_CHECK_INTERVAL` | `30` | Seconds between cleanup checks for idle models |

### LlamaFarm Integration

Add to your `llamafarm.yaml`:

```yaml
version: v1
name: my-project
namespace: default

runtime:
  default_model: phi-2

  models:
    - name: phi-2
      description: Fast small language model
      provider: universal
      model: microsoft/phi-2
      base_url: http://127.0.0.1:11540
      prompts: [default]
      transformers:
        device: auto
        dtype: auto
        trust_remote_code: true
        model_type: text

    - name: embedder
      description: Text embeddings for RAG
      provider: universal
      model: sentence-transformers/all-MiniLM-L6-v2
      base_url: http://127.0.0.1:11540
      prompts: [default]
      transformers:
        device: auto
        dtype: auto
        trust_remote_code: true
        model_type: embedding

    - name: sd-2-1
      description: Image generation
      provider: universal
      model: stabilityai/stable-diffusion-2-1
      base_url: http://127.0.0.1:11540
      prompts: [default]
      transformers:
        device: auto
        dtype: auto
        trust_remote_code: true
        model_type: image
      diffusion:
        default_steps: 30
        default_guidance: 7.5
        default_size: "512x512"
        scheduler: euler
        enable_optimizations: true

prompts:
  - name: default
    messages:
      - role: system
        content: You are a helpful AI assistant.
```

See [examples/universal-embedder-rag/](../../examples/universal-embedder-rag/) for a complete RAG example.

---

## Hardware Requirements

| Model Type | Min RAM | Recommended | GPU VRAM | Notes |
|------------|---------|-------------|----------|-------|
| **CausalLM (small)** | 4GB | 8GB | 4GB+ | Phi-2, Qwen-0.5B |
| **CausalLM (medium)** | 8GB | 16GB | 8GB+ | Llama-3B, Mistral-7B |
| **CausalLM (large)** | 16GB | 32GB | 16GB+ | Llama-8B, requires quantization on smaller GPUs |
| **Encoder** | 2GB | 4GB | 2GB+ | Fast inference |
| **Diffusion (SD 2.1)** | 8GB | 16GB | 6GB+ | SDXL needs 8GB+ VRAM |
| **Diffusion (SDXL)** | 16GB | 32GB | 8GB+ | High memory usage |
| **Vision** | 2GB | 4GB | 2GB+ | Fast |
| **Audio (Whisper)** | 4GB | 8GB | 4GB+ | Whisper-large |
| **Multimodal** | 8GB | 16GB | 8GB+ | LLaVA needs 16GB+ |

### Device Support

- **Apple Silicon (MPS)**: M1/M2/M3/M4 chips auto-detected, excellent performance
- **NVIDIA CUDA**: Auto-detected, supports all models
- **CPU**: Fallback for all models (slower but functional)

### Performance Tips

- **Enable xformers** for faster diffusion: `uv pip install xformers`
- **Use FP16** on compatible GPUs: automatically enabled on CUDA
- **Reduce image size** for faster generation: `512x512` instead of `1024x1024`
- **Lower inference steps** for speed: 20-30 steps instead of 50
- **Use turbo models** for fastest generation: `stabilityai/sdxl-turbo`

---

## Testing

### Run All Tests

```bash
cd runtimes/universal
uv run pytest tests/ -v
```

### Run Specific Test Suites

```bash
# Test language models
uv run pytest tests/test_language_model.py -v

# Test encoder models
uv run pytest tests/test_encoder_model.py -v

# Test diffusion models (requires GPU/MPS)
uv run pytest tests/test_diffusion_model.py -v

# Test vision models
uv run pytest tests/test_vision_model.py -v

# Test audio models
uv run pytest tests/test_audio_model.py -v

# Test multimodal models
uv run pytest tests/test_multimodal_model.py -v

# Test streaming server
uv run pytest tests/test_streaming_server.py -v
```

### Quick Smoke Test

```bash
bash quick_test.sh
```

### Manual Testing

See [MANUAL_TEST_EXAMPLES.md](./MANUAL_TEST_EXAMPLES.md) for comprehensive manual testing scenarios.

---

## Examples

### Example 1: RAG with Universal Embedder

Complete example at [examples/universal-embedder-rag/](../../examples/universal-embedder-rag/)

```bash
cd examples/universal-embedder-rag
lf init universal-rag
cp llamafarm-example.yaml llamafarm.yaml
lf start
lf datasets create -s pdf_ingest -b main_db research
lf datasets upload research ./sample.pdf
lf datasets process research
lf rag query --database main_db "What are the key findings?"
```

### Example 2: Generate Images with Stable Diffusion

```python
import requests

url = "http://localhost:11540/v1/images/generations"
payload = {
    "prompt": "NVIDIA Jetson Orin Nano on a desk, tech photography, 8k",
    "model": "stabilityai/stable-diffusion-2-1",
    "size": "512x512",
    "num_inference_steps": 30,
    "guidance_scale": 7.5,
    "seed": 42
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Generated: {result['data'][0]['url']}")
```

### Example 3: Transcribe Audio with Whisper

```python
import requests
import base64

with open("audio.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

url = "http://localhost:11540/v1/audio/transcriptions"
payload = {
    "file": audio_b64,
    "model": "openai/whisper-large-v3",
    "language": "en"
}

response = requests.post(url, json=payload)
print(response.json()["text"])
```

### Example 4: Zero-Shot Image Classification with CLIP

```python
import requests
import base64
from PIL import Image

# Load image
with open("photo.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

url = "http://localhost:11540/v1/vision/clip"
payload = {
    "model": "openai/clip-vit-base-patch32",
    "images": [image_b64],
    "candidate_labels": ["indoor", "outdoor", "portrait", "landscape"]
}

response = requests.post(url, json=payload)
print(response.json())
```

---

## Documentation

- **[GETTING_STARTED.md](./GETTING_STARTED.md)** - Beginner-friendly walkthrough
- **[MODEL_TYPES.md](./MODEL_TYPES.md)** - All supported model types and examples
- **[CURL_TEST_COMMANDS.md](./CURL_TEST_COMMANDS.md)** - Quick curl commands for testing
- **[IMAGE_UPLOAD_EXAMPLES.md](./IMAGE_UPLOAD_EXAMPLES.md)** - Image handling guide
- **[MANUAL_TEST_EXAMPLES.md](./MANUAL_TEST_EXAMPLES.md)** - Comprehensive testing guide
- **[ONNX_STRATEGY.md](./ONNX_STRATEGY.md)** - ONNX optimization roadmap
- **[ONNX_IMPLEMENTATION_GUIDE.md](./ONNX_IMPLEMENTATION_GUIDE.md)** - ONNX integration guide

### Related Documentation

- [LlamaFarm Models Documentation](../../docs/website/docs/models/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

---

## Troubleshooting

### Model Downloads Slowly

**Solution:** Set a HuggingFace token for faster downloads:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

### Out of Memory Errors

**Solutions:**
- Use smaller models: `microsoft/phi-2` or `Qwen/Qwen2.5-0.5B-Instruct`
- Reduce image size: `"size": "512x512"`
- Reduce inference steps: `"num_inference_steps": 20`
- Force CPU mode: `python server.py --device cpu` (slower but uses less VRAM)
- Enable model offloading (future feature)

### Generation is Very Slow

**Causes & Solutions:**
- **First-time downloads**: Models are 2-10GB. Wait for cache, subsequent loads are fast.
- **CPU mode**: Switch to GPU if available.
- **Large models**: Use smaller variants or turbo models.
- **Too many steps**: Reduce `num_inference_steps`.

**Expected generation times (512x512 image):**
- Tiny SD (M1 Mac): 2-5 seconds
- SD 2.1 (M1 Mac): 20-30 seconds
- SD 2.1 (RTX 3090): 5-10 seconds
- SD 2.1 (CPU): 2-5 minutes

### Images Look Weird or Low Quality

**Solutions:**
- Improve prompt quality (be descriptive)
- Increase `guidance_scale` (7.0-9.0 range)
- Increase `num_inference_steps` (40-50 range)
- Add `negative_prompt` to avoid unwanted elements
- Use higher-quality models: SDXL instead of SD 2.1

### Model Not Found Error

**Solution:** Ensure the model ID is correct. Browse [HuggingFace Models](https://huggingface.co/models) to find valid model IDs.

### CUDA Out of Memory

**Solutions:**
- Close other GPU-using applications
- Reduce batch size (`n=1`)
- Use gradient checkpointing (future feature)
- Use smaller models or quantized variants

### Python Version Error (torch can't be installed)

**Error:** `Distribution torch==2.8.0 can't be installed because it doesn't have a source distribution or wheel for the current platform`

**Cause:** You're using Python 3.14 or newer, which PyTorch doesn't support yet.

**Solution:**
```bash
# From project root (recommended):
nx sync universal-runtime

# Or manually:
uv python install 3.12  # Install Python 3.12 (one-time)
uv sync                 # Install dependencies with Python 3.12
```

The `pyproject.toml` constraint ensures only compatible Python versions are used.

### Server Won't Start

**Solutions:**
- Check port 11540 is not in use: `lsof -i :11540`
- Ensure Python 3.11-3.13 is installed: `python --version`
- Reinstall dependencies: `nx sync universal-runtime` or `uv sync`
- Check logs for detailed error messages

---

## Contributing

Contributions are welcome! Please follow the [LlamaFarm contribution guidelines](../../CONTRIBUTING.md).

### Development Workflow

1. **Research**: Document findings in `thoughts/shared/research/`
2. **Plan**: Outline steps in `thoughts/shared/plans/`
3. **Implement**: Follow Python style guide (4 spaces, run `uv run ruff check --fix .`)
4. **Test**: Run `uv run pytest tests/ -v`
5. **Document**: Update README and relevant docs

**Note:** Pre-commit hooks are configured to automatically format code with ruff when committing changes in any Python component (`server/`, `rag/`, `config/`, `runtimes/universal/`). The hooks are installed at the repository root and will run `ruff check --fix` and `ruff format` on all staged Python files in these directories.

### Adding New Model Types

1. Create model class in `models/` inheriting from `BaseModel`
2. Implement `load()` and inference methods
3. Add API endpoints in `server.py`
4. Write tests in `tests/`
5. Update documentation (README, MODEL_TYPES.md)

---

## License

See [LICENSE](../../LICENSE) file in the repository root.

---

## Need Help?

- **GitHub Issues**: [Report bugs or request features](https://github.com/llama-farm/llamafarm/issues)
- **LlamaFarm Docs**: [Full platform documentation](../../docs/website/docs/)
- **HuggingFace Forums**: [Model-specific questions](https://discuss.huggingface.co/)

---

**Happy modeling! 🚀🦙**
