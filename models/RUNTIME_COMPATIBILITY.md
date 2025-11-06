# Runtime Compatibility Guide

This document explains how different model formats work with different runtimes in LlamaFarm.

## Runtimes Overview

### 1. Universal Runtime
- **Format**: `transformers` (HuggingFace models)
- **Provider**: `universal`
- **Models**: Any HuggingFace transformer model
- **Download**: Automatic on first use
- **Use Case**: Most versatile, works with any HuggingFace model
- **Hardware**: Auto-detects (MPS, CUDA, CPU)

### 2. Ollama
- **Format**: `gguf` (quantized models)
- **Provider**: `ollama`
- **Models**: GGUF models from Ollama library
- **Download**: `ollama pull model:tag`
- **Use Case**: Optimized local inference with quantization
- **Hardware**: CPU-friendly, GPU-accelerated when available

### 3. Lemonade
- **Formats**: `gguf`, `transformers`, `onnx`
- **Provider**: `lemonade`
- **Models**: Depends on backend
  - `llamacpp` backend: GGUF models
  - `transformers` backend: HuggingFace models
  - `onnx` backend: ONNX models
- **Download**: `uv run lemonade-server-dev pull user.ModelName --checkpoint HF/Repo --recipe BACKEND`
- **Use Case**: Hardware-optimized (NPU/GPU), flexible backends
- **Hardware**: Auto-optimizes for NPU, GPU, or CPU

### 4. OpenAI
- **Format**: `api` (cloud API)
- **Provider**: `openai`
- **Models**: Custom API endpoints
- **Download**: N/A (hosted service)
- **Use Case**: Custom endpoints or OpenAI-compatible APIs
- **Hardware**: Remote (no local requirements)

## Format Compatibility Matrix

| Format       | Universal | Ollama | Lemonade (llamacpp) | Lemonade (transformers) | Lemonade (onnx) |
|--------------|-----------|--------|---------------------|-------------------------|-----------------|
| transformers | ✅         | ❌      | ❌                   | ✅                       | ❌               |
| gguf         | ❌         | ✅      | ✅                   | ❌                       | ❌               |
| onnx         | ❌         | ❌      | ❌                   | ❌                       | ✅               |

## Download Commands by Runtime

### Universal
```bash
# No manual download needed - auto-downloads on first use
# Model is pulled from HuggingFace automatically
```

### Ollama
```bash
# Pull a model
ollama pull qwen3:4b

# List available models
ollama list

# Run a model
ollama run qwen3:4b
```

### Lemonade (llamacpp - GGUF)
```bash
# Download GGUF model
uv run lemonade-server-dev pull user.Qwen3-4B \
  --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M \
  --recipe llamacpp

# List models
uv run lemonade-server-dev list

# Start with model
LEMONADE_MODEL=user.Qwen3-4B nx start lemonade
```

### Lemonade (transformers - HuggingFace)
```bash
# Download transformers model
uv run lemonade-server-dev pull user.Qwen3-4B-Transformers \
  --checkpoint Qwen/Qwen3-4B \
  --recipe transformers

# Start with model
LEMONADE_MODEL=user.Qwen3-4B-Transformers LEMONADE_BACKEND=transformers nx start lemonade
```

## Choosing the Right Runtime

### Use Universal when:
- ✅ You want the easiest setup
- ✅ You want any HuggingFace model
- ✅ You don't mind larger model sizes
- ✅ You have sufficient RAM/VRAM

### Use Ollama when:
- ✅ You want pre-optimized quantized models
- ✅ You want fast CPU inference
- ✅ You want simple `ollama pull` command
- ✅ You want lower memory usage

### Use Lemonade when:
- ✅ You have NPU hardware (Apple Silicon, Intel NPU, AMD NPU)
- ✅ You want maximum performance optimization
- ✅ You need flexibility (GGUF or Transformers)
- ✅ You're comfortable with advanced configuration

### Use OpenAI when:
- ✅ You have a custom API endpoint
- ✅ You want to use hosted services
- ✅ You don't want local inference

## Example Model Configurations

### Example 1: Small Fast Model (Qwen3 0.6B)
```yaml
variants:
  - id: qwen3:0.6b
    providers:
      universal:
        runtime: universal
        format: transformers
        model_id: "Qwen/Qwen3-0.6B"
        download_command: "Auto-downloads on first use"

      ollama:
        runtime: ollama
        format: gguf
        model_id: "qwen3:0.6b"
        download_command: "ollama pull qwen3:0.6b"

      lemonade:
        runtime: lemonade
        format: gguf
        backend: llamacpp
        model_id: "user.Qwen3-0.6B"
        checkpoint: "unsloth/Qwen3-0.6B-GGUF"
        recipe: "llamacpp"
        download_command: "uv run lemonade-server-dev pull user.Qwen3-0.6B --checkpoint unsloth/Qwen3-0.6B-GGUF --recipe llamacpp"
```

### Example 2: Vision Model (Llama 3.2 11B)
```yaml
variants:
  - id: llama3.2:11b
    providers:
      universal:
        runtime: universal
        format: transformers
        model_id: "meta-llama/Llama-3.2-11B-Vision-Instruct"
        requires_token: true

      ollama:
        runtime: ollama
        format: gguf
        model_id: "llama3.2-vision:11b"
        download_command: "ollama pull llama3.2-vision:11b"
```

## Provider Field Mapping

- `runtime`: Which runtime executes the model (ollama, lemonade, universal, openai)
- `format`: File/API format (gguf, transformers, onnx, api)
- `provider`: Same as runtime (kept for backward compatibility)
- `backend`: Lemonade-specific backend (llamacpp, transformers, onnx)
- `recipe`: Lemonade-specific recipe (same as backend)

## Notes

- **GGUF**: Quantized format, smaller sizes, faster inference on CPU
- **Transformers**: Full precision, larger sizes, better quality
- **ONNX**: Optimized format for specific hardware (NPU)
- Some models only available in certain formats
- Download commands are provided for convenience in the UI
