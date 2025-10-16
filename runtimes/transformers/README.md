# Transformers Runtime

OpenAI-compatible API server for any HuggingFace model (text and image generation).

## Features

- **Universal Model Support**: Run ANY HuggingFace model without restrictions
- **OpenAI API Compatible**: Drop-in replacement for OpenAI endpoints
- **Hardware Acceleration**: Auto-detects and uses MPS/CUDA/CPU
- **Diffusion Models**: Full support for Stable Diffusion, SDXL, FLUX, etc.
- **No Restrictions**: Uses `trust_remote_code=True` for maximum compatibility
- **UV-based**: Fast dependency management with `uv`

## Quick Start

### 1. Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) installed:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

### 2. Start the server

```bash
# Using nx (recommended)
nx start transformers

# Or directly
cd runtimes/transformers
bash start.sh
```

The server will:
- Auto-detect your hardware (MPS/CUDA/CPU)
- Install dependencies via `uv sync`
- Start on `http://localhost:11540`

### 3. Generate an image

```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain landscape at sunset",
    "size": "1024x1024",
    "n": 1,
    "model": "stabilityai/stable-diffusion-xl-base-1.0"
  }'
```

### 4. Text generation

```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-2",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ]
  }'
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `TRANSFORMERS_PORT` | `11540` | Server port |
| `TRANSFORMERS_HOST` | `127.0.0.1` | Server host |
| `TRANSFORMERS_OUTPUT_DIR` | `~/.llamafarm/outputs/images/` | Image output directory |
| `TRANSFORMERS_CACHE_DIR` | `~/.cache/huggingface` | Model cache directory |
| `DEFAULT_IMAGE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | Default diffusion model |
| `DEFAULT_INPAINT_MODEL` | `runwayml/stable-diffusion-inpainting` | Default inpainting model |

## Supported Models

### Text Generation
- Any HuggingFace `AutoModelForCausalLM` model
- Phi, Llama, Mistral, Qwen, Gemma, etc.
- Automatic chat template formatting
- Streaming support (coming soon)

### Image Generation
- Stable Diffusion (all versions)
- Stable Diffusion XL (SDXL)
- SDXL Turbo (1-step generation)
- FLUX.1-dev, FLUX.1-schnell
- Any `DiffusionPipeline` model

### Image Editing
- Stable Diffusion Inpainting
- ControlNet (coming soon)
- IP-Adapter (coming soon)

## API Endpoints

All endpoints follow the OpenAI API specification.

### Health Check

```bash
GET /health
```

Returns device information and loaded models.

### List Models

```bash
GET /v1/models
```

Returns currently loaded models.

### Chat Completions

```bash
POST /v1/chat/completions
```

Generate text responses using any HuggingFace text model.

**Parameters:**
- `model` (string, required): HuggingFace model ID
- `messages` (array, required): Chat messages
- `temperature` (number, optional): Sampling temperature (default: 1.0)
- `max_tokens` (number, optional): Maximum tokens to generate
- `top_p` (number, optional): Nucleus sampling parameter (default: 1.0)

### Image Generation

```bash
POST /v1/images/generations
```

Generate images from text prompts using diffusion models.

**Parameters:**
- `prompt` (string, required): Text description of the image
- `model` (string, optional): HuggingFace model ID (uses `DEFAULT_IMAGE_MODEL` if not specified)
- `n` (integer, optional): Number of images (1-10, default: 1)
- `size` (string, optional): Image dimensions (default: "1024x1024")
- `negative_prompt` (string, optional): What to avoid in the image
- `num_inference_steps` (integer, optional): Denoising steps (1-150)
- `guidance_scale` (number, optional): Prompt adherence (1.0-20.0)
- `seed` (integer, optional): Random seed for reproducibility
- `scheduler` (string, optional): Diffusion scheduler (ddim, euler, dpm++, etc.)
- `response_format` (string, optional): "url" or "b64_json" (default: "url")

### Image Editing

```bash
POST /v1/images/edits
```

Edit images using inpainting models.

**Parameters:**
- `image` (string, required): Base64 encoded image
- `prompt` (string, required): Edit description
- `mask` (string, optional): Base64 encoded mask
- `model` (string, optional): HuggingFace model ID
- Other parameters same as `/v1/images/generations`

## Hardware Requirements

- **Minimum**: 8GB RAM, CPU only
- **Recommended**: 16GB RAM, Apple Silicon or NVIDIA GPU
- **Optimal**: 24GB+ VRAM for large models (SDXL, etc.)

## Platform Support

### macOS (Apple Silicon)
- Auto-detected MPS acceleration
- Attention slicing for memory efficiency
- Optimized for M1/M2/M3 chips

### Linux (NVIDIA)
- CUDA acceleration
- xformers memory efficient attention
- Model CPU offload for large models

### Linux/Windows (CPU)
- Float32 precision
- Smaller batch sizes
- Compatible with all models

## Development

### Install dependencies

```bash
cd runtimes/transformers
uv sync
```

### Run tests

```bash
uv run pytest tests/
# or via nx
nx test transformers
```

### Start development server

```bash
uv run python server.py
```

## Troubleshooting

### Out of Memory

Diffusion models can use 4-8GB+ of VRAM. The runtime automatically enables optimizations:

- **MPS**: `enable_attention_slicing()`
- **CUDA**: `enable_xformers_memory_efficient_attention()`, `enable_model_cpu_offload()`
- **CPU**: Lower precision, smaller batch sizes

Reduce image size or inference steps if you still encounter OOM errors.

### Model Not Loading

Most models require:
- Sufficient memory (8GB+ RAM recommended)
- Accepting model license on HuggingFace (for gated models)
- Setting `HF_TOKEN` environment variable for gated models:
  ```bash
  export HF_TOKEN=hf_xxxxx
  ```

### Slow Generation

First-time model loading requires downloading 2-10GB+ from HuggingFace. Subsequent loads are fast (cached to `~/.cache/huggingface`).

Image generation typically takes:
- **Mac M1**: 20-30s for 1024x1024 SDXL
- **NVIDIA RTX 3090**: 5-10s for 1024x1024 SDXL
- **CPU**: 2-5 minutes for 512x512 SD

## Architecture

```
transformers/
├── server.py              # Main FastAPI application
├── start.sh              # Startup script with auto-install
├── pyproject.toml        # UV/Python project config
├── project.json          # NX configuration
├── models/
│   ├── base.py          # Base model class
│   ├── text_model.py    # Text generation wrapper
│   └── image_model.py   # Image generation wrapper
├── utils/
│   ├── device.py        # Device detection (MPS/CUDA/CPU)
│   └── file_utils.py    # File I/O helpers
└── tests/
    ├── test_server.py
    ├── test_text_models.py
    └── test_image_models.py
```

## Resources

- **HuggingFace Models**: https://huggingface.co/models
- **Diffusers Documentation**: https://huggingface.co/docs/diffusers
- **Transformers Documentation**: https://huggingface.co/docs/transformers
- **LlamaFarm Docs**: See `docs/website/docs/models/` for integration details

## License

Part of the LlamaFarm project. See main repository for license details.
