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
- **Chat (Quality)**: `meta-llama/Llama-3.1-8B-Instruct`
- **Chat (Speed)**: `microsoft/phi-2` or `Qwen/Qwen2.5-0.5B-Instruct`
- **Image Generation**: `stabilityai/stable-diffusion-xl-base-1.0`
- **Speech Recognition**: `openai/whisper-large-v3`

---

## Installation

### Prerequisites

- **Python 3.11+** (3.11 or 3.12 recommended)
- **[uv](https://github.com/astral-sh/uv)** package manager
- **8GB+ RAM** minimum (16GB+ recommended for larger models)
- **Optional**: Apple Silicon Mac (M1/M2/M3/M4) or NVIDIA GPU for acceleration

### Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup the Runtime

```bash
# Navigate to the universal runtime directory
cd runtimes/universal

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

### Optional: Install xformers (for Diffusion optimization)

```bash
uv pip install xformers
```

This significantly speeds up Stable Diffusion models on CUDA GPUs.

---

## Quick Start

### 1. Start the Server

```bash
cd runtimes/universal
bash start.sh
```

Or directly with Python:

```bash
uv run python server.py
```

Or via Nx (if in LlamaFarm monorepo):

```bash
nx start universal
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

### Server Won't Start

**Solutions:**
- Check port 11540 is not in use: `lsof -i :11540`
- Ensure Python 3.11+ is installed: `python --version`
- Reinstall dependencies: `uv sync`
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
