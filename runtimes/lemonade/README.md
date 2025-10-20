# Lemonade Runtime

Lemonade is an SDK for running local LLMs with optimized performance across different hardware configurations.

## Overview

Lemonade provides:
- **Multiple inference backends**: ONNX Runtime GenAI, llama.cpp, Hugging Face Transformers
- **Hardware-aware optimization**: Automatically optimizes for NPUs, GPUs, and CPUs
- **OpenAI-compatible API**: Standard `/v1/chat/completions` and `/v1/completions` endpoints
- **Multiple model formats**: GGUF and ONNX support
- **Cross-platform**: Works on Linux, macOS (including M-chips), and Windows

## Installation

Install the Lemonade SDK using uv (recommended):

```bash
uv pip install lemonade-sdk
```

Alternatively, using pip:

```bash
pip install lemonade-sdk
```

For more information, see: https://lemonade-server.ai/docs/

## Quick Start

See [QUICKSTART.md](./QUICKSTART.md) for a step-by-step guide to get running in 5 minutes.

## Model Management

### Downloading Models

Before starting the server, download models using the `lemonade-server-dev pull` command:

```bash
# Basic syntax
uv run lemonade-server-dev pull user.ModelName \
  --checkpoint HuggingFace/Repo:variant \
  --recipe llamacpp
```

**Example Downloads:**

```bash
# Small, fast model (0.6B)
uv run lemonade-server-dev pull user.Qwen3-0.6B \
  --checkpoint unsloth/Qwen3-0.6B-GGUF \
  --recipe llamacpp

# Balanced model (4B, recommended)
uv run lemonade-server-dev pull user.Qwen3-4B \
  --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M \
  --recipe llamacpp

# Powerful model (8B)
uv run lemonade-server-dev pull user.Qwen3-8B \
  --checkpoint unsloth/Qwen3-8B-GGUF:Q4_K_M \
  --recipe llamacpp

# Coding model
uv run lemonade-server-dev pull user.Qwen2.5-Coder-32B-Instruct \
  --checkpoint unsloth/Qwen2.5-Coder-32B-Instruct-GGUF:Q4_K_M \
  --recipe llamacpp

# Vision model
uv run lemonade-server-dev pull user.Qwen2.5-VL-7B-Instruct \
  --checkpoint unsloth/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M \
  --recipe llamacpp
```

### Listing Models

```bash
uv run lemonade-server-dev list
```

### Model Storage

By default, Lemonade stores models in:
- **Linux/macOS**: `~/.cache/lemonade/models`
- **Windows**: `%LOCALAPPDATA%\lemonade\models`

### Current Limitation

Lemonade currently supports **one model at a time**. To switch models, restart the Lemonade server with a different `LEMONADE_MODEL`.

This is a known limitation tracked in: https://github.com/lemonade-sdk/lemonade/issues/163

## Starting Lemonade

### Using nx (recommended)

The `nx` command is part of LlamaFarm's monorepo tooling (via Nx). It provides a convenient way to start Lemonade:

```bash
# Start with default settings (port 11534, GGUF/llamacpp backend)
nx start lemonade

# Start with a specific model
LEMONADE_MODEL=user.Qwen3-4B nx start lemonade

# Start on a different port
LEMONADE_PORT=11535 nx start lemonade

# Combine environment variables
LEMONADE_MODEL=user.Qwen3-8B LEMONADE_PORT=11535 nx start lemonade
```

**What `nx start lemonade` does:**
1. Checks if Lemonade SDK is installed (auto-installs if missing)
2. Detects your hardware (GPU, NPU, or CPU)
3. Configures optimal acceleration settings
4. Starts the Lemonade server with OpenAI-compatible API

### Using the script directly

If you prefer not to use `nx`, you can run the startup script directly:

```bash
bash runtimes/lemonade/start.sh
```

Environment variables work the same way:
```bash
LEMONADE_MODEL=user.Qwen3-4B bash runtimes/lemonade/start.sh
```

## Configuration

Configure Lemonade using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LEMONADE_PORT` | `11534` | Port for the Lemonade API server |
| `LEMONADE_HOST` | `127.0.0.1` | Host address to bind to |
| `LEMONADE_BACKEND` | `onnx` | Inference backend: `onnx`, `llamacpp`, or `transformers` |
| `LEMONADE_MODEL` | (none) | Optional: Pre-load a specific model on startup |

### Example with custom configuration

```bash
LEMONADE_PORT=11535 LEMONADE_BACKEND=llamacpp nx start lemonade
```

## Using Lemonade in llamafarm.yaml

Configure your project to use Lemonade as the runtime provider:

```yaml
version: v1
name: my-project
namespace: default

runtime:
  models:
    - name: lemon
      description: "Lemonade local model"
      provider: lemonade
      model: user.Qwen3-4B
      base_url: "http://127.0.0.1:11534/v1"
      default: true
      lemonade:
        backend: llamacpp  # llamacpp, onnx, or transformers
        port: 11534
        context_size: 32768

prompts:
  - name: default
    messages:
      - role: system
        content: "You are a helpful assistant."
```

## API Endpoints

Once started, Lemonade exposes OpenAI-compatible endpoints at:

- **Base URL**: `http://127.0.0.1:11534/v1`
- **Chat Completions**: `POST /v1/chat/completions`
- **Completions**: `POST /v1/completions`
- **Models**: `GET /v1/models`

## Backend Selection

Lemonade supports three inference backends, each optimized for different scenarios. The startup script **automatically detects your hardware** and configures the optimal acceleration.

### 1. ONNX Runtime GenAI (default: `onnx`)
**Best for**: Cross-platform compatibility, production deployments

```yaml
runtime:
  lemonade:
    backend: onnx  # default
```

- **Model format**: ONNX
- **Auto-detection**: Automatically uses available execution providers
  - CUDA (NVIDIA GPUs)
  - DirectML (Windows GPUs)
  - CoreML (macOS)
  - CPU fallback
- **Pros**: Best compatibility, works everywhere, minimal configuration
- **Cons**: Requires ONNX-format models

### 2. llama.cpp (`llamacpp`)
**Best for**: GGUF models, maximum performance, Apple Silicon

```yaml
runtime:
  lemonade:
    backend: llamacpp
    context_size: 32768  # optional, default 32768
```

- **Model format**: GGUF (quantized models)
- **Auto-detection**:
  - **macOS**: Uses Metal (Apple Silicon/Intel GPUs)
  - **Linux with NVIDIA**: Uses CUDA (best for NVIDIA)
  - **Linux with AMD/Intel GPU**: Uses Vulkan
  - **CPU-only**: Automatically falls back to CPU mode
- **Pros**: Excellent performance, supports quantized models, flexible
- **Cons**: Requires GGUF-format models

**Hardware Detection Output Examples:**
```bash
# On macOS
Using Metal acceleration for Apple Silicon...

# On Linux with NVIDIA
Detected NVIDIA GPU, attempting to use CUDA acceleration...

# On Linux with AMD/Intel GPU
Detected GPU device, using Vulkan acceleration...

# On CPU-only system
No GPU detected, using CPU-only mode...
```

### 3. Hugging Face Transformers (`transformers`)
**Best for**: Development, model experimentation, PyTorch ecosystem

```yaml
runtime:
  lemonade:
    backend: transformers
```

- **Model format**: Hugging Face Hub models
- **Auto-detection**: Uses PyTorch's built-in acceleration
  - CUDA (NVIDIA GPUs on Linux/Windows)
  - MPS (Metal Performance Shaders on macOS)
  - CPU fallback
- **Pros**: Direct access to HuggingFace models, PyTorch optimizations
- **Cons**: Larger memory footprint

## Backend Configuration Examples

### Using llamacpp on macOS (Metal acceleration)
```yaml
runtime:
  provider: lemonade
  model: user.Qwen3-0.6B  # Must match downloaded model name

  lemonade:
    backend: llamacpp
    port: 11534
    context_size: 32768
```

**Command**:
```bash
LEMONADE_MODEL=user.Qwen3-0.6B nx start lemonade
```
**Result**: Automatically uses Metal acceleration

### Using llamacpp on Linux with NVIDIA GPU
```yaml
runtime:
  provider: lemonade
  model: user.Llama-3.2-1B-Instruct  # Must match downloaded model name

  lemonade:
    backend: llamacpp
    context_size: 65536
```

**Command**:
```bash
LEMONADE_MODEL=user.Llama-3.2-1B-Instruct nx start lemonade
```
**Result**: Automatically detects NVIDIA GPU and uses CUDA

### Using ONNX (default)
```yaml
runtime:
  provider: lemonade
  model: Phi-3-mini-4k-instruct-onnx  # Pre-installed ONNX model

  lemonade:
    backend: onnx  # or omit, it's the default
    port: 11534
```

**Command**:
```bash
LEMONADE_MODEL=Phi-3-mini-4k-instruct-onnx nx start lemonade
```
**Result**: ONNX runtime auto-detects available execution providers

### Using Transformers
```yaml
runtime:
  provider: lemonade
  model: microsoft/Phi-3-mini-4k-instruct  # HuggingFace model ID

  lemonade:
    backend: transformers
```

**Command**:
```bash
LEMONADE_BACKEND=transformers LEMONADE_MODEL=microsoft/Phi-3-mini-4k-instruct nx start lemonade
```
**Result**: PyTorch auto-detects CUDA/MPS/CPU

## Override Environment Variables

You can override any configuration with environment variables:

```bash
# Force a specific backend
LEMONADE_BACKEND=transformers nx start lemonade

# Use a different model (must be downloaded first)
LEMONADE_MODEL=user.Qwen3-0.6B nx start lemonade

# Change port and context size
LEMONADE_PORT=11535 LEMONADE_CONTEXT_SIZE=65536 nx start lemonade
```

## Health Checks

The LlamaFarm server will automatically health-check Lemonade if configured in `llamafarm.yaml`. Health checks verify:
- Lemonade server is running and accessible
- API endpoints respond correctly
- Model is loaded (if configured)

## Troubleshooting

### Port already in use
If port 11534 is occupied, either:
1. Stop the process using the port
2. Change the port: `LEMONADE_PORT=11535 nx start lemonade`

### Lemonade command not found
Ensure Lemonade SDK is installed:
```bash
pip install lemonade-sdk
# or
uv pip install lemonade-sdk
```

### Backend-specific issues
- **ONNX**: Requires ONNX Runtime dependencies
- **llama.cpp**: Ensure you have GGUF-format models
- **transformers**: Requires PyTorch and Hugging Face transformers

Check Lemonade documentation for backend-specific requirements: https://lemonade-server.ai/docs/

## Port Allocation Strategy

LlamaFarm uses the following port allocation:
- **8000**: LlamaFarm main server
- **11434**: Ollama (default)
- **11534**: Lemonade (this runtime)
- **Future runtimes**: Will use similar high-numbered ports

## Extensibility

This runtime is designed as a template for adding additional local runtimes (e.g., vLLM, TGI). Key design principles:

1. **Isolated**: Runs as a separate nx service
2. **Optional**: Does not start with `nx dev`
3. **Health-checked**: Main server monitors runtime availability
4. **OpenAI-compatible**: Standard API interface
5. **Configurable**: Environment variables + schema configuration

## Resources

- **Lemonade Website**: https://lemonade-server.ai/
- **GitHub**: https://github.com/lemonade-sdk/lemonade
- **Documentation**: https://lemonade-server.ai/docs/
- **LlamaFarm Docs**: See `docs/website/docs/models/` for integration details
