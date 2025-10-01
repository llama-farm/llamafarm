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

Install the Lemonade SDK:

```bash
pip install lemonade-sdk
```

Or using uv (recommended for LlamaFarm development):

```bash
uv pip install lemonade-sdk
```

For more information, see: https://lemonade-server.ai/docs/

## Starting Lemonade

### Using nx (recommended)

```bash
nx start lemonade
```

This will start the Lemonade server on port **11534** (default).

### Using the script directly

```bash
bash runtimes/lemonade/start.sh
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
  provider: lemonade
  model: "Phi-3-mini-4k-instruct-onnx"  # Example ONNX model
  base_url: "http://127.0.0.1:11534/v1"

  # Lemonade-specific configuration (optional)
  lemonade:
    backend: onnx  # onnx, llamacpp, or transformers
    port: 11534
    model_path: ~/.cache/lemonade/models  # Custom model storage location

prompts:
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

Lemonade supports three inference backends, each optimized for different scenarios:

### ONNX Runtime GenAI (default: `onnx`)
- Best for: NPU/GPU acceleration, Windows with NPUs, production deployments
- Model format: ONNX
- Hardware: Optimized for NPUs, DirectML (Windows), CUDA (NVIDIA)

### llama.cpp (`llamacpp`)
- Best for: CPU inference, broad model compatibility, Apple Silicon
- Model format: GGUF
- Hardware: Optimized for CPUs, Metal (macOS), CUDA (NVIDIA)

### Hugging Face Transformers (`transformers`)
- Best for: Development, model experimentation, PyTorch ecosystem
- Model format: Hugging Face model hub
- Hardware: Flexible GPU/CPU support

## Model Management

### Current Limitation
Lemonade currently supports **one model at a time**. To switch models, restart the Lemonade server with a different `LEMONADE_MODEL`.

This is a known limitation tracked in: https://github.com/lemonade-sdk/lemonade/issues/163

### Model Storage
By default, Lemonade stores models in:
- **Linux/macOS**: `~/.cache/lemonade/models`
- **Windows**: `%LOCALAPPDATA%\lemonade\models`

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
