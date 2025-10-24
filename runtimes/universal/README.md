# Universal Runtime

OpenAI-compatible API server for running any HuggingFace model locally with hardware acceleration.

## Overview

The Universal Runtime is a flexible, production-ready inference server that provides:

- **Universal Model Support**: Run ANY HuggingFace model (transformers, diffusers) without restrictions
- **OpenAI API Compatibility**: Drop-in replacement for OpenAI endpoints
- **Hardware Acceleration**: Auto-detects and uses MPS/CUDA/CPU
- **Multiple Model Types**: Text generation, embeddings, image generation, audio, vision, multimodal
- **SSE Streaming**: Real-time token streaming for text generation
- **Content Negotiation**: Multiple image formats (JPEG, PNG, WebP) via Accept headers
- **Production Ready**: Stateless, cloud-native, horizontally scalable

---

## Table of Contents

- [Quick Start](#quick-start)
- [Model Types](#model-types)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)
- [Testing](#testing)
- [Production Deployment](#production-deployment)
- [Performance Optimization](#performance-optimization)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### Installation

```bash
cd runtimes/universal
uv sync
```

### Start Server

```bash
# Option 1: Direct start
uv run uvicorn server:app --host 0.0.0.0 --port 11540 --reload

# Option 2: Using start script
bash start.sh

# Option 3: Via nx (if in LlamaFarm monorepo)
nx start universal
```

Server will be available at: `http://localhost:11540`

### Quick Test

```bash
# Generate text
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
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

# Generate image (JPEG - smallest size!)
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Accept: image/jpeg" \
  -d '{
    "model": "stabilityai/stable-diffusion-2-1-base",
    "prompt": "A serene mountain landscape"
  }' > image.jpg
```

---

## Model Types

The Universal Runtime supports 6 major model categories:

### 1. Causal Language Models (Text Generation)

**Purpose:** Generate text continuations, chat responses, code completion

**Example Models:**
- `Qwen/Qwen2.5-0.5B-Instruct` - Fast, small
- `microsoft/phi-2` - 2.7B params
- `mistralai/Mistral-7B-Instruct-v0.3` - High quality
- `meta-llama/Llama-3.2-3B-Instruct` - Latest Llama

**Features:**
- ✅ SSE streaming support
- ✅ Chat template formatting
- ✅ Stop sequence support
- ✅ Temperature control

**API Endpoint:** `POST /v1/chat/completions`

```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "stream": true,
    "max_tokens": 200
  }'
```

---

### 2. Encoder Models (Embeddings & Classification)

**Purpose:** Convert text to vectors, classify text

**Example Models:**
- `sentence-transformers/all-MiniLM-L6-v2` - 384-dim, fast
- `BAAI/bge-base-en-v1.5` - 768-dim, high quality
- `nomic-ai/nomic-embed-text-v1.5` - Long context

**Features:**
- ✅ Single and batch embeddings
- ✅ Normalization support
- ✅ Critical for RAG systems
- ✅ ONNX optimization available (3x faster)

**API Endpoint:** `POST /v1/embeddings`

```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": ["Text 1", "Text 2", "Text 3"]
  }'
```

---

### 3. Diffusion Models (Image Generation)

**Purpose:** Generate images from text prompts

**Example Models:**
- `stabilityai/stable-diffusion-xl-base-1.0` - SDXL, 1024x1024
- `stabilityai/stable-diffusion-2-1` - SD 2.1, 768x768
- `stabilityai/sdxl-turbo` - Fast (1-4 steps)
- `black-forest-labs/FLUX.1-dev` - Cutting-edge quality

**Features:**
- ✅ Text-to-image generation
- ✅ Image editing (inpainting)
- ✅ Image variations (img2img)
- ✅ Multiple formats (JPEG, PNG, WebP)
- ✅ Seed-based reproducibility

**API Endpoints:**
- `POST /v1/images/generations` - Generate from text
- `POST /v1/images/edits` - Edit/inpaint
- `POST /v1/images/variations` - Create variations

**Format Selection:**
```bash
# JPEG (default - 50-70% smaller!)
curl -H "Accept: image/jpeg" ... > image.jpg

# PNG (lossless)
curl -H "Accept: image/png" ... > image.png

# WebP (modern)
curl -H "Accept: image/webp" ... > image.webp

# JSON (multiple images or metadata)
curl -H "Accept: application/json" ... | jq -r '.data[0].b64_json' | base64 -d > image.png
```

---

### 4. Vision Models (Image Understanding)

**Purpose:** Image classification, zero-shot classification, feature extraction

**Example Models:**
- `google/vit-base-patch16-224` - ViT classification
- `openai/clip-vit-base-patch32` - CLIP zero-shot
- `microsoft/resnet-50` - ResNet-50
- `facebook/dinov2-base` - Self-supervised features

**Features:**
- ✅ Image classification
- ✅ Zero-shot CLIP classification
- ✅ Image embeddings
- ✅ Direct file upload support

**API Endpoints:**
- `POST /v1/vision/classify` - Classification
- `POST /v1/vision/clip` - CLIP zero-shot

```bash
# Direct file upload
curl -X POST http://localhost:11540/v1/vision/classify/upload \
  -F "file=@image.jpg" \
  -F "model=google/vit-base-patch16-224"
```

---

### 5. Audio Models (Speech-to-Text)

**Purpose:** Transcribe speech, translate audio

**Example Models:**
- `openai/whisper-tiny` - 39M params, fast
- `openai/whisper-base` - 74M params
- `openai/whisper-small` - 244M params
- `openai/whisper-large-v3` - 1.5B params, best
- `distil-whisper/distil-large-v3` - 50% faster

**Features:**
- ✅ Multi-language transcription
- ✅ Translation to English
- ✅ Timestamp generation
- ✅ Direct file upload

**API Endpoints:**
- `POST /v1/audio/transcriptions` - Transcribe
- `POST /v1/audio/translations` - Translate to English

```bash
curl -X POST http://localhost:11540/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "model=openai/whisper-tiny" \
  -F "language=en"
```

---

### 6. Multimodal Models (Vision-Language)

**Purpose:** Image captioning, visual question answering, visual chat

**Example Models:**
- `Salesforce/blip-image-captioning-base` - Image captioning
- `Salesforce/blip-vqa-base` - Visual QA
- `llava-hf/llava-1.5-7b-hf` - Visual chat
- `microsoft/Florence-2-base` - Unified vision-language

**Features:**
- ✅ Image captioning
- ✅ Visual question answering
- ✅ Direct file upload support

**API Endpoints:**
- `POST /v1/multimodal/caption` - Generate captions
- `POST /v1/multimodal/vqa` - Answer questions

```bash
# Image captioning
curl -X POST http://localhost:11540/v1/multimodal/caption/upload \
  -F "file=@photo.jpg" \
  -F "model=Salesforce/blip-image-captioning-base"

# Visual question answering
curl -X POST http://localhost:11540/v1/multimodal/vqa/upload \
  -F "file=@room.jpg" \
  -F "model=Salesforce/blip-image-captioning-base" \
  -F "question=How many people are in this image?"
```

---

## API Endpoints

### Health & Info

```bash
# Health check
GET /health

# List loaded models
GET /v1/models

# Server info
GET /
```

### Text Generation

```bash
POST /v1/chat/completions

Parameters:
- model (string, required): HuggingFace model ID
- messages (array, required): Chat messages
- temperature (number): Sampling temperature (default: 1.0)
- max_tokens (number): Maximum tokens to generate
- top_p (number): Nucleus sampling (default: 1.0)
- stream (boolean): Enable SSE streaming (default: false)
- stop (string|array): Stop sequences
```

### Embeddings

```bash
POST /v1/embeddings

Parameters:
- model (string, required): HuggingFace model ID
- input (string|array, required): Text(s) to embed
- encoding_format (string): "float" (default)
```

### Image Generation

```bash
POST /v1/images/generations

Parameters:
- prompt (string, required): Text description
- model (string, optional): Model ID (default configured)
- n (integer): Number of images (1-10, default: 1)
- size (string): Image dimensions (e.g., "512x512")
- negative_prompt (string): What to avoid
- num_inference_steps (integer): Denoising steps (1-150)
- guidance_scale (number): Prompt adherence (1.0-20.0)
- seed (integer): Random seed for reproducibility
- scheduler (string): Diffusion scheduler (ddim, euler, etc.)
- response_format (string): "b64_json" or "url" (default: "b64_json")

Accept Header Options:
- image/jpeg - JPEG format (smallest, default for binary)
- image/png - PNG format (lossless)
- image/webp - WebP format (modern)
- application/json - JSON with base64
```

### Image Editing

```bash
POST /v1/images/edits

Parameters:
- image (string, required): Base64 encoded image OR multipart file
- prompt (string, required): Edit description
- mask (string, optional): Base64 encoded mask
- model (string, optional): Model ID
- (other parameters same as generations)
```

### Audio Transcription

```bash
POST /v1/audio/transcriptions

Parameters (multipart/form-data):
- file (file, required): Audio file
- model (string, required): Model ID
- language (string): ISO-639-1 code (default: auto-detect)
- prompt (string): Guide transcription
- response_format (string): "json", "text", or "verbose_json"
- temperature (float): Sampling temperature (0.0-1.0)
```

### Vision Classification

```bash
POST /v1/vision/classify
POST /v1/vision/classify/upload (file upload)

Parameters:
- model (string, required): Model ID
- image (string) OR file: Base64 or file upload
- top_k (integer): Number of predictions (default: 5)
```

### Multimodal

```bash
POST /v1/multimodal/caption
POST /v1/multimodal/caption/upload (file upload)

Parameters:
- model (string, required): Model ID
- image (string) OR file: Base64 or file upload
- max_length (integer): Caption length

POST /v1/multimodal/vqa
POST /v1/multimodal/vqa/upload (file upload)

Parameters:
- model (string, required): Model ID
- image (string) OR file: Base64 or file upload
- question (string, required): Question about the image
```

---

## Configuration

### Environment Variables

```bash
# Server Configuration
UNIVERSAL_PORT=11540          # Server port (default: 11540)
UNIVERSAL_HOST=127.0.0.1      # Server host (default: 127.0.0.1)
UNIVERSAL_API_KEY=universal   # API key (default: "universal")

# Runtime Backend (for optimization)
RUNTIME_BACKEND=pytorch       # "pytorch" (default) or "onnx"
ONNX_PROVIDER=CUDAExecutionProvider  # For ONNX backend

# Model Configuration
DEFAULT_IMAGE_MODEL=stabilityai/stable-diffusion-xl-base-1.0
DEFAULT_INPAINT_MODEL=runwayml/stable-diffusion-inpainting
```

### LlamaFarm Integration

Add to your `llamafarm.yaml`:

```yaml
version: v1
name: my-project
namespace: default

models:
  # Fast local streaming
  - id: local-chat
    provider: universal
    model: Qwen/Qwen2.5-0.5B-Instruct
    base_url: http://127.0.0.1:11540/v1

  # Embeddings for RAG
  - id: embeddings
    provider: universal
    model: sentence-transformers/all-MiniLM-L6-v2

  # Image generation
  - id: images
    provider: universal
    model: stabilityai/stable-diffusion-xl-base-1.0

prompts:
  - role: system
    content: "You are a helpful AI assistant."

rag:
  databases: []
```

---

## Testing

### Quick Test

```bash
# Run automated quick tests (~90 seconds)
./quick_test.sh
```

### Full Test Suite

```bash
# Fast tests only (recommended for development)
./run_tests.sh

# All tests including slow model downloads (~5 minutes)
./run_tests.sh --slow

# With coverage report
./run_tests.sh --coverage

# Specific test file
uv run python -m pytest tests/test_encoder_model.py -v
```

### Test Categories

**Fast Tests** (~60 seconds):
- CausalLM: Text generation, streaming
- Encoder: Embeddings, normalization
- Diffusion: Image generation (tiny model)

**Slow Tests** (marked `@pytest.mark.slow`):
- Vision: CLIP tests (~60s)
- Audio: Whisper tests (~120s)
- Multimodal: BLIP tests (~180s)

### Manual Testing

Comprehensive curl examples available in:
- `CURL_TEST_COMMANDS.md` - Complete API reference
- `MANUAL_TEST_EXAMPLES.md` - Copy-paste examples
- `SERVER_TESTING_GUIDE.md` - Testing procedures

---

## Production Deployment

### Quick Production Checklist

**Priority 1: Essential (30 minutes)**
1. ✅ Convert EncoderModel to ONNX (3x faster embeddings)
2. ✅ Set `RUNTIME_BACKEND=onnx` for embeddings
3. ✅ Add monitoring

**Priority 2: Optional**
4. ⚠️ CausalLM → ONNX (1.4x faster, if high volume)
5. ⚠️ VisionModel → ONNX (2x faster, if high volume)

**Don't Bother:**
6. ❌ DiffusionModel → ONNX (complex, use `torch.compile()` instead)
7. ❌ Multimodal → ONNX (limited support)

### Docker Deployment

```dockerfile
FROM python:3.12

# Install ONNX Runtime (optional, for optimization)
RUN pip install onnxruntime-gpu optimum[onnxruntime]

# Install dependencies
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync

# Copy application
COPY . .

# Set production backend (optional)
ENV RUNTIME_BACKEND=onnx
ENV ONNX_PROVIDER=CUDAExecutionProvider

# Start server
CMD ["uv", "run", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "11540"]
```

### Docker Compose

```yaml
version: '3.8'
services:
  universal-runtime:
    build: .
    environment:
      - RUNTIME_BACKEND=onnx
      - ONNX_PROVIDER=CUDAExecutionProvider
    ports:
      - "11540:11540"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Hardware Recommendations

**For RAG (Embeddings + Text):**
- CPU-Only: 8-core, 16GB RAM (~$50-100/month)
- GPU: NVIDIA T4 or better, 16GB VRAM (~$300-500/month)
- High Scale: NVIDIA A100, 40GB+ VRAM (~$1500+/month)

**For Image Generation:**
- Minimum: NVIDIA RTX 3060 (12GB) or T4 (16GB)
- Recommended: NVIDIA A10G (24GB) or RTX 4090
- High Performance: NVIDIA A100 (40-80GB)

### Monitoring

Track key metrics:

```python
# Recommended metrics
- inference_latency_ms: Model inference time
- throughput_qps: Requests per second
- model_load_time_ms: Model loading time
- gpu_utilization: GPU usage percentage
- memory_usage_bytes: RAM/VRAM usage
```

---

## Performance Optimization

### ONNX Acceleration

For embeddings (critical for RAG systems):

**Impact:** 3x faster embeddings, 65% cost reduction

```python
# Enable ONNX backend
export RUNTIME_BACKEND=onnx
export ONNX_PROVIDER=CUDAExecutionProvider

# Models automatically convert on first load
# Subsequent loads use cached ONNX models
```

**Performance Comparison:**

| Model Type | PyTorch | ONNX | Speedup |
|------------|---------|------|---------|
| Embeddings | 15ms | 5ms | **3x** |
| Text Gen | 50ms | 35ms | **1.4x** |
| Vision | 12ms | 6ms | **2x** |

**ONNX Recommendation by Model:**
- ✅ **EncoderModel**: HIGHLY RECOMMENDED (easy, 3x speedup)
- ⚠️ **CausalLM**: Optional (1.4x speedup, moderate effort)
- ⚠️ **VisionModel**: Optional (2x speedup, moderate effort)
- ❌ **DiffusionModel**: NOT RECOMMENDED (complex, use torch.compile)

See `ONNX_STRATEGY.md` and `ONNX_IMPLEMENTATION_GUIDE.md` for details.

### Content Negotiation

Use JPEG for smallest image responses:

```bash
# JPEG (default) - 50-70% smaller than PNG!
curl -H "Accept: image/jpeg" ... > image.jpg

# PNG (when lossless needed)
curl -H "Accept: image/png" ... > image.png
```

**Size Comparison (512x512 image):**
- JPEG: 350 KB (95% quality)
- PNG: 1.2 MB (lossless)
- WebP: 420 KB (90% quality)

See `CONTENT_NEGOTIATION_GUIDE.md` for details.

### Streaming

Enable streaming for better UX:

```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "stream": true
  }'
```

See `SSE_STREAMING.md` for implementation details.

---

## Development

### Project Structure

```
universal/
├── server.py              # FastAPI application
├── start.sh              # Startup script
├── models/
│   ├── base.py          # Base model class
│   ├── language_model.py    # Text generation
│   ├── encoder_model.py      # Embeddings
│   ├── diffusion_model.py    # Image generation
│   ├── vision_model.py       # Image understanding
│   ├── audio_model.py        # Speech-to-text
│   └── multimodal_model.py   # Vision-language
├── utils/
│   ├── device.py        # Device detection
│   └── file_utils.py    # File I/O helpers
├── tests/              # Test suite
│   ├── conftest.py
│   ├── test_*.py
│   └── README.md
└── docs/              # Documentation
```

### Adding New Model Types

1. Create new model class in `models/`:
   ```python
   from models.base import BaseModel

   class NewModel(BaseModel):
       def __init__(self, model_id: str, device: str):
           super().__init__(model_id, device)
           self.supports_streaming = False

       async def load(self):
           # Load model
           pass

       async def generate(self, **kwargs):
           # Implement generation
           pass
   ```

2. Register endpoint in `server.py`

3. Add tests in `tests/test_new_model.py`

4. Update documentation

### Code Style

**Python:**
- 4-space indentation
- Line length: 88 (Black style)
- Type hints required
- Docstrings for public methods

**Formatting:**
```bash
# Format code
cd runtimes/universal
uv run ruff check --fix .
```

### Contributing

1. Create feature branch
2. Write tests first (TDD)
3. Implement feature
4. Run tests: `./run_tests.sh`
5. Format code: `uv run ruff check --fix .`
6. Update documentation
7. Submit PR

---

## Troubleshooting

### Server Issues

**Server won't start:**
```bash
# Check port availability
lsof -i :11540

# Kill existing process
pkill -f uvicorn

# Restart
uv run uvicorn server:app --reload
```

**Model not loading:**
```bash
# Check HuggingFace access
export HF_TOKEN=hf_xxxxx

# Try smaller model first
curl ... -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", ...}'
```

### Performance Issues

**Slow inference:**
- First-time: Model download (2-10GB, one-time)
- Subsequent: Check GPU utilization
- Consider ONNX backend for embeddings

**Out of memory:**
- Use smaller models
- Reduce batch size
- Reduce image size/inference steps
- Force CPU: `--device cpu`

### Image Issues

**Images look weird:**
- Increase `guidance_scale` (7.0-9.0)
- Increase `num_inference_steps` (40-50)
- Add negative prompts
- Try different scheduler

**Wrong format:**
```bash
# Specify format explicitly
curl -H "Accept: image/png" ...  # For PNG
curl -H "Accept: image/jpeg" ... # For JPEG
```

### Streaming Not Working

**Tokens arrive all at once:**
- Verify `"stream": true` in request
- Check client supports SSE
- Check for buffering proxies
- Use curl with `--no-buffer`

### Common Errors

**"Model does not support tools":**
- Disable instructor mode
- Choose different model
- Use `--no-rag` flag

**"CUDA out of memory":**
- Use smaller model
- Reduce batch size
- Enable memory optimizations
- Use CPU fallback

**Base64 encoding issues:**
```bash
# macOS
base64 -i file.jpg

# Linux
base64 -w 0 file.jpg
```

---

## Resources

### Documentation

- `GETTING_STARTED.md` - Detailed setup guide
- `MODEL_TYPES.md` - Complete model type reference
- `CURL_TEST_COMMANDS.md` - API testing commands
- `MANUAL_TEST_EXAMPLES.md` - Copy-paste examples
- `SERVER_TESTING_GUIDE.md` - Testing procedures
- `CONTENT_NEGOTIATION_GUIDE.md` - Image format guide
- `SSE_STREAMING.md` - Streaming implementation
- `ONNX_STRATEGY.md` - Performance optimization
- `ONNX_IMPLEMENTATION_GUIDE.md` - ONNX setup
- `PRODUCTION_READY_CHECKLIST.md` - Deployment guide
- `IMAGE_UPLOAD_EXAMPLES.md` - File upload guide
- `AUDIO_UPLOAD_FIX.md` - Audio file handling

### External Resources

- [HuggingFace Models](https://huggingface.co/models)
- [Diffusers Docs](https://huggingface.co/docs/diffusers)
- [Transformers Docs](https://huggingface.co/docs/transformers)
- [ONNX Runtime](https://onnxruntime.ai/)
- [HuggingFace Optimum](https://huggingface.co/docs/optimum)

### Support

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check detailed guides in this directory
- **Examples**: See `examples/` in main repository

---

## License

Part of the LlamaFarm project. See main repository for license details.

---

## Summary

The Universal Runtime provides a production-ready, OpenAI-compatible API for running any HuggingFace model locally with:

- ✅ **6 Model Types**: Text, embeddings, images, audio, vision, multimodal
- ✅ **Hardware Acceleration**: MPS/CUDA/CPU auto-detection
- ✅ **SSE Streaming**: Real-time token generation
- ✅ **Multiple Formats**: JPEG/PNG/WebP image output
- ✅ **ONNX Optimization**: 3x faster embeddings
- ✅ **Direct File Upload**: Efficient image/audio handling
- ✅ **OpenAI Compatible**: Drop-in replacement
- ✅ **Production Ready**: Stateless, scalable, cloud-native

**Quick Start:**
```bash
cd runtimes/universal
uv sync
uv run uvicorn server:app --port 11540 --reload
```

**Health Check:**
```bash
curl http://localhost:11540/health
```

For detailed guides, see the documentation files in this directory.
