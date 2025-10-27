# Getting Started with Transformers Runtime

This guide walks you through setting up the Transformers runtime, downloading recommended models, and running your first image generation, editing, and variation tasks.

## Table of Contents

1. [Installation](#installation)
2. [Download Recommended Models](#download-recommended-models)
3. [Start the Server](#start-the-server)
4. [API Usage Examples](#api-usage-examples)
5. [Sample Configuration](#sample-configuration)
6. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager
- 16GB+ RAM recommended (8GB minimum)
- Apple Silicon (M1/M2/M3) or NVIDIA GPU recommended for performance

### Install uv (if not installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup Virtual Environment

```bash
cd runtimes/transformers
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

---

## Download Recommended Models

Models download automatically on first use, but pre-downloading saves time. We recommend starting with these 4 models:

### 1. Tiny SD (Testing/Development)
- **Size**: ~500MB
- **Use**: Fast testing and development
- **Quality**: Low (but very fast)

```bash
uv run python download_models.py --model hf-internal-testing/tiny-stable-diffusion-torch
```

### 2. Stable Diffusion 2.1 (Production Generation)
- **Size**: ~5GB
- **Use**: High-quality image generation
- **Quality**: High

```bash
uv run python download_models.py --model stabilityai/stable-diffusion-2-1
```

### 3. Stable Diffusion 2 Inpainting (Editing)
- **Size**: ~3.5GB
- **Use**: Image editing and inpainting
- **Quality**: High

```bash
uv run python download_models.py --model stabilityai/stable-diffusion-2-inpainting
```

### 4. Runway Inpainting (Smaller Alternative)
- **Size**: ~2GB
- **Use**: Lighter inpainting model
- **Quality**: Medium

```bash
uv run python download_models.py --model runwayml/stable-diffusion-inpainting
```

### Download All Recommended Models

```bash
# Download all 4 models at once
uv run python download_models.py \
  --model hf-internal-testing/tiny-stable-diffusion-torch \
  --model stabilityai/stable-diffusion-2-1 \
  --model stabilityai/stable-diffusion-2-inpainting \
  --model runwayml/stable-diffusion-inpainting
```

**First download takes 15-30 minutes depending on internet speed.** Models are cached at `~/.cache/huggingface/`.

---

## Start the Server

### Option 1: Direct Start (Recommended for development)

```bash
cd runtimes/transformers
uv run python server.py
```

### Option 2: Via NX (if in LlamaFarm monorepo)

```bash
nx start transformers
```

### Option 3: Via start script

```bash
cd runtimes/transformers
bash start.sh
```

**Server URL**: `http://127.0.0.1:11540`

The server will:
- Auto-detect hardware (MPS/CUDA/CPU)
- Load models on-demand
- Save generated images to `~/.llamafarm/outputs/images/`

---

## API Usage Examples

### Example 1: Generate an Image

**Basic generation with default settings:**

```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain lake at sunset, photorealistic, 8k",
    "model": "stabilityai/stable-diffusion-2-1",
    "size": "512x512"
  }'
```

**Advanced generation with all parameters:**

```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "NVIDIA Jetson Orin Nano developer kit on a modern desk, professional tech photography, studio lighting, 8k",
    "model": "stabilityai/stable-diffusion-2-1",
    "size": "512x512",
    "num_inference_steps": 50,
    "guidance_scale": 8.0,
    "seed": 42,
    "negative_prompt": "blurry, low quality, distorted, amateur",
    "n": 1
  }'
```

**Response:**

```json
{
  "created": 1760640000,
  "data": [
    {
      "url": "/Users/you/.llamafarm/outputs/images/stabilityai_stable-diffusion-2-1_20250116_143025_42_0.png"
    }
  ]
}
```

---

### Example 2: Image Variations (img2img)

Transform an existing image based on a text prompt.

**Python example with file path:**

```python
import requests
import base64

# Encode image to base64
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

url = "http://localhost:11540/v1/images/variations"
payload = {
    "prompt": "Transform into a cyberpunk version with neon lights and futuristic elements",
    "image": image_b64,
    "model": "stabilityai/stable-diffusion-2-1",
    "size": "512x512",
    "strength": 0.75,  # 0.0 = no change, 1.0 = complete remake
    "num_inference_steps": 40,
    "guidance_scale": 7.5,
    "seed": 42
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Generated: {result['data'][0]['url']}")
```

**Via LlamaFarm API (automatic file handling):**

```bash
# LlamaFarm API handles file paths automatically
curl -X POST http://localhost:8000/v1/projects/default/my-project/images/variations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Make it look like a watercolor painting",
    "image": "/Users/me/photos/input.jpg",
    "model": "sd-2-1",
    "strength": 0.8,
    "seed": 123
  }'
```

---

### Example 3: Image Editing (Inpainting)

Edit specific regions of an image using a mask.

```python
import requests
import base64
from PIL import Image, ImageDraw
import io

# Load original image
with open("input.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

# Create mask (white = edit area, black = preserve)
mask = Image.new('RGB', (512, 512), 'black')
draw = ImageDraw.Draw(mask)
draw.rectangle([100, 100, 400, 400], fill='white')  # Edit center square

# Encode mask
mask_bytes = io.BytesIO()
mask.save(mask_bytes, format='PNG')
mask_b64 = base64.b64encode(mask_bytes.getvalue()).decode()

# Make request
url = "http://localhost:11540/v1/images/edits"
payload = {
    "prompt": "Add glowing holographic displays and screens",
    "image": image_b64,
    "mask": mask_b64,
    "model": "stabilityai/stable-diffusion-2-inpainting",
    "size": "512x512",
    "num_inference_steps": 50,
    "guidance_scale": 7.5
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Edited: {result['data'][0]['url']}")
```

---

### Example 4: Batch Generation

Generate multiple variations at once:

```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic city at night, cyberpunk style",
    "model": "stabilityai/stable-diffusion-2-1",
    "size": "512x512",
    "n": 4,
    "seed": 42
  }'
```

---

## Sample Configuration

Add to your `llamafarm.yaml`:

```yaml
version: v1
name: my-image-project
namespace: default

prompts:
  - name: default
    messages:
      - role: system
        content: |
          You are a helpful AI assistant for image generation tasks.

runtime:
  default_model: sd-2-1

  models:
    # Fast test model
    - name: tiny-sd
      description: Tiny Stable Diffusion for testing
      provider: transformers
      model: hf-internal-testing/tiny-stable-diffusion-torch
      base_url: http://127.0.0.1:11540
      prompts: [default]
      transformers:
        device: auto
        dtype: auto
        trust_remote_code: true
        model_type: image
      diffusion:
        default_steps: 25
        default_guidance: 7.5
        default_size: "512x512"
        scheduler: euler_a
        enable_optimizations: true

    # Production generation model
    - name: sd-2-1
      description: Stable Diffusion 2.1 for high-quality images
      provider: transformers
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

    # Inpainting/editing model
    - name: sd-inpaint
      description: Stable Diffusion 2 Inpainting for editing
      provider: transformers
      model: stabilityai/stable-diffusion-2-inpainting
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
        scheduler: ddim
        enable_optimizations: true

    # Lighter inpainting alternative
    - name: sd-inpaint-small
      description: Smaller inpainting model
      provider: transformers
      model: runwayml/stable-diffusion-inpainting
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
        scheduler: ddim
        enable_optimizations: true
```

---

## Next Steps

### 1. Explore More Models

Browse HuggingFace for additional models:
- **SDXL**: `stabilityai/stable-diffusion-xl-base-1.0` (10GB, higher quality)
- **SDXL Turbo**: `stabilityai/sdxl-turbo` (1-4 steps, very fast)
- **FLUX.1**: `black-forest-labs/FLUX.1-dev` (12GB, cutting-edge quality)

### 2. Optimize Performance

**Reduce generation time:**
```json
{
  "num_inference_steps": 20,  // Lower steps = faster (trade quality)
  "scheduler": "euler_a"       // Fast scheduler
}
```

**Improve quality:**
```json
{
  "num_inference_steps": 50,   // More steps = better quality
  "guidance_scale": 8.0,       // Stricter prompt following
  "scheduler": "dpm++"         // High-quality scheduler
}
```

### 3. Integration Examples

**OpenAI SDK compatibility:**

```python
import openai

openai.api_base = "http://localhost:11540/v1"
openai.api_key = "not-used"  # Transformers runtime doesn't require auth

response = openai.Image.create(
    prompt="A beautiful sunset over mountains",
    size="512x512",
    n=1,
    model="stabilityai/stable-diffusion-2-1"
)

print(response['data'][0]['url'])
```

**Via LlamaFarm CLI:**

```bash
# Start LlamaFarm with your config
cd /path/to/your/project
lf start --no-docker

# Images are generated via the project API
# Access via http://localhost:8000/v1/projects/{namespace}/{project}/images/...
```

### 4. Read More Documentation

- [Transformers Runtime README](./README.md) - Full API reference
- [LlamaFarm Models Docs](../../docs/website/docs/models/) - Integration guide
- [HuggingFace Diffusers](https://huggingface.co/docs/diffusers) - Model documentation

---

## Troubleshooting

### Model downloads slowly
Set a HuggingFace token for faster downloads:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

### Out of memory errors
- Use smaller models (`tiny-sd` or `runwayml/stable-diffusion-inpainting`)
- Reduce image size: `"size": "512x512"` instead of `1024x1024`
- Reduce inference steps: `"num_inference_steps": 20`
- Force CPU: `python server.py --device cpu`

### Generation is slow
First-time loads download 2-10GB. Subsequent loads use cache and are fast.

**Expected generation times:**
- **Tiny SD (M1 Mac)**: 2-5 seconds
- **SD 2.1 (M1 Mac)**: 20-30 seconds
- **SD 2.1 (RTX 3090)**: 5-10 seconds
- **SD 2.1 (CPU)**: 2-5 minutes

### Images look weird
- Check your prompt quality
- Increase `guidance_scale` (7.0-9.0)
- Increase `num_inference_steps` (40-50)
- Add negative prompts to avoid unwanted elements

---

## Need Help?

- **GitHub Issues**: Report bugs or request features
- **LlamaFarm Docs**: Full platform documentation
- **HuggingFace Forums**: Model-specific questions

Happy generating! 🎨✨
