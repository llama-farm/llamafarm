# Lemonade Runtime - Quick Start

Get up and running with Lemonade in under 5 minutes.

## Prerequisites

1. **Install Lemonade SDK**:
   ```bash
   pip install lemonade-sdk
   ```

   Or using uv (recommended):
   ```bash
   uv pip install lemonade-sdk
   ```

## Step 1: Download a Model

Before starting the server, download a model using the `lemonade-server-dev pull` command:

```bash
# For a small, fast model (0.6B parameters)
uv run lemonade-server-dev pull user.Qwen3-0.6B \
  --checkpoint unsloth/Qwen3-0.6B-GGUF \
  --recipe llamacpp

# For a balanced model (4B parameters, recommended)
uv run lemonade-server-dev pull user.Qwen3-4B \
  --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M \
  --recipe llamacpp

# For a powerful model (8B parameters)
uv run lemonade-server-dev pull user.Qwen3-8B \
  --checkpoint unsloth/Qwen3-8B-GGUF:Q4_K_M \
  --recipe llamacpp
```

### Recommended Models to Try

**Small & Fast (< 1GB):**
```bash
# Qwen3-0.6B - Great for quick responses
uv run lemonade-server-dev pull user.Qwen3-0.6B \
  --checkpoint unsloth/Qwen3-0.6B-GGUF \
  --recipe llamacpp

# Qwen3-1.7B - Small reasoning model
uv run lemonade-server-dev pull user.Qwen3-1.7B \
  --checkpoint unsloth/Qwen3-1.7B-GGUF:Q4_K_M \
  --recipe llamacpp

# Llama-3.2-1B - Meta's small model
uv run lemonade-server-dev pull user.Llama-3.2-1B-Instruct \
  --checkpoint unsloth/Llama-3.2-1B-Instruct-GGUF:Q4_K_M \
  --recipe llamacpp
```

**Balanced (2-5GB):**
```bash
# Qwen3-4B - Best balance of speed and quality
uv run lemonade-server-dev pull user.Qwen3-4B \
  --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M \
  --recipe llamacpp

# Gemma-3-4b-it - Google's instruction-tuned model with vision
uv run lemonade-server-dev pull user.Gemma-3-4b-it \
  --checkpoint bartowski/Gemma-3-4b-it-GGUF:Q4_K_M \
  --recipe llamacpp
```

**Powerful (5GB+):**
```bash
# Qwen3-8B - High-quality reasoning
uv run lemonade-server-dev pull user.Qwen3-8B \
  --checkpoint unsloth/Qwen3-8B-GGUF:Q4_K_M \
  --recipe llamacpp

# DeepSeek-Qwen3-8B - Advanced reasoning capabilities
uv run lemonade-server-dev pull user.DeepSeek-Qwen3-8B \
  --checkpoint unsloth/DeepSeek-Qwen3-8B-GGUF:Q4_K_M \
  --recipe llamacpp

# Qwen2.5-Coder-32B - Large coding model
uv run lemonade-server-dev pull user.Qwen2.5-Coder-32B-Instruct \
  --checkpoint unsloth/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M \
  --recipe llamacpp
```

**Specialized:**
```bash
# Devstral-Small-2507 - Coding and tool-calling
uv run lemonade-server-dev pull user.Devstral-Small-2507 \
  --checkpoint unsloth/Devstral-Small-2507-GGUF:Q4_K_M \
  --recipe llamacpp

# Qwen2.5-VL-7B - Vision model
uv run lemonade-server-dev pull user.Qwen2.5-VL-7B-Instruct \
  --checkpoint unsloth/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M \
  --recipe llamacpp
```

**Check Downloaded Models:**
```bash
uv run lemonade-server-dev list
```

## Step 2: Start Lemonade Server

```bash
# From the llamafarm project root
LEMONADE_MODEL=user.Qwen3-4B nx start lemonade
```

This starts Lemonade on port 11534 with llama.cpp backend (recommended for GGUF models).

> **Note:** The `nx start lemonade` command automatically picks up configuration from your `llamafarm.yaml`. Currently, Lemonade must be manually started from the project root. In the future, Lemonade will run as a container and be auto-started by the LlamaFarm server.

## Step 3: Configure Your Project

Create or update `llamafarm.yaml`:

```yaml
version: v1
name: my-lemonade-project
namespace: default

runtime:
  models:
    - name: lemon
      description: "Lemonade local model"
      provider: lemonade
      model: user.Qwen3-4B  # Use the model you downloaded
      base_url: "http://127.0.0.1:11534/v1"
      default: true
      lemonade:
        backend: llamacpp
        port: 11534
        context_size: 32768

prompts:
  - role: system
    content: "You are a helpful assistant."
```

## Step 4: Chat!

```bash
lf chat "What is the capital of France?"
```

## Multi-Model Setup

You can configure multiple Lemonade models by running separate instances on different ports:

```yaml
runtime:
  models:
    - name: fast
      description: "Fast Lemonade model"
      provider: lemonade
      model: user.Qwen3-0.6B
      base_url: "http://127.0.0.1:11534/v1"
      default: true
      lemonade:
        backend: llamacpp
        port: 11534

    - name: powerful
      description: "Powerful Lemonade model"
      provider: lemonade
      model: user.Qwen3-8B
      base_url: "http://127.0.0.1:11535/v1"
      lemonade:
        backend: llamacpp
        port: 11535
```

Start each instance (from llamafarm project root):
```bash
# Terminal 1
LEMONADE_MODEL=user.Qwen3-0.6B LEMONADE_PORT=11534 nx start lemonade

# Terminal 2
LEMONADE_MODEL=user.Qwen3-8B LEMONADE_PORT=11535 nx start lemonade
```

> **Note:** In the future, multiple Lemonade instances will run as containers and be auto-started by the LlamaFarm server.

## Custom Backends

### Using ONNX (cross-platform)

```bash
LEMONADE_BACKEND=onnx nx start lemonade
```

Or in `llamafarm.yaml`:

```yaml
runtime:
  models:
    - name: onnx-model
      description: "ONNX model"
      provider: lemonade
      model: Phi-3-mini-4k-instruct-onnx
      base_url: "http://127.0.0.1:11534/v1"
      default: true
      lemonade:
        backend: onnx
        port: 11534
```

### Using Transformers (PyTorch)

```bash
LEMONADE_BACKEND=transformers nx start lemonade
```

## Troubleshooting

### Port Already in Use
```bash
LEMONADE_PORT=11535 nx start lemonade
```

### Lemonade Not Installed
```bash
pip install lemonade-sdk
# or
uv pip install lemonade-sdk
```

### Model Not Found
Make sure you've downloaded the model first:
```bash
uv run lemonade-server-dev list  # Check what's downloaded
uv run lemonade-server-dev pull user.ModelName --checkpoint ... --recipe llamacpp
```

### Check Health
Visit: http://localhost:8000/health

Look for the "lemonade" component status.

## Next Steps

- Read the full [README.md](./README.md) for advanced configuration
- Check the [example config](./example.llamafarm.yaml)
- See available models: `uv run lemonade-server-dev list`
- Learn about model recipes: https://lemonade-server.ai/docs/server/server_models/
