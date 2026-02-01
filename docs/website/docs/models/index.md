---
title: Models & Runtime
sidebar_position: 7
---

# Models & Runtime

LlamaFarm focuses on inference rather than fine-tuning. The runtime section of `llamafarm.yaml` describes how chat completions are executed—whether using the Universal Runtime (recommended), local Ollama/Lemonade, or cloud providers.

## Quick Start: Choosing a Provider

| Provider | Best For | Setup |
|----------|----------|-------|
| **[Universal Runtime](#universal-runtime)** ⭐ | Full HuggingFace access, multimodal (images, audio, embeddings), specialized ML | `provider: universal` |
| **[Cloud Providers](#cloud-providers)** | Production APIs (OpenAI, Grok, Together AI) | `provider: openai` + `base_url` + `api_key` |
| **[Ollama](#ollama)** | Easy local setup, quantized GGUF models | `provider: ollama` |
| **[Lemonade](#lemonade)** | NPU/GPU acceleration, Apple Silicon | `provider: lemonade` |

:::tip Recommended Setup
Start with **Universal Runtime** for development—it provides the most flexibility with access to any HuggingFace model, embeddings, image generation, and specialized ML capabilities (OCR, NER, classification, anomaly detection). Use cloud providers for production workloads requiring enterprise reliability.
:::

## Multi-Model Support

LlamaFarm supports configuring multiple models in a single project. You can switch between models via CLI or API:

```yaml
runtime:
  default_model: chat  # Which model to use by default

  models:
    - name: chat
      description: "Universal Runtime chat model"
      provider: universal
      model: microsoft/phi-2
      base_url: http://127.0.0.1:11540

    - name: embedder
      description: "Embeddings for RAG"
      provider: universal
      model: sentence-transformers/all-MiniLM-L6-v2
      base_url: http://127.0.0.1:11540

    - name: cloud
      description: "Cloud fallback for production"
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
```

**Using multi-model:**
- CLI: `lf chat --model cloud "your question"`
- CLI: `lf models list` (shows all available models)
- API: `POST /v1/projects/{ns}/{id}/chat/completions` with `{"model": "cloud", ...}`

**Legacy single-model configs are still supported** and automatically converted internally.

## Runtime Responsibilities

- Route chat requests to the configured provider.
- Respect instructor modes (`tools`, `json`, `md_json`, etc.) when available.
- Surface provider errors directly (incorrect model name, missing API key).
- Cooperate with agent handlers (simple chat, structured output, RAG-aware prompts).

## Universal Runtime

The Universal Runtime is LlamaFarm's **recommended** runtime provider, supporting **any HuggingFace model** through PyTorch Transformers and Diffusers. Unlike Ollama (GGUF-only) or Lemonade (optimized quantized models), Universal Runtime provides access to the entire HuggingFace Hub ecosystem plus specialized ML capabilities.

### Why Universal Runtime?

- **Full HuggingFace Access** – Any PyTorch model: text, embeddings, images, audio, vision
- **Specialized ML** – OCR, document extraction, NER, classification, reranking, anomaly detection
- **No Pre-conversion** – Models auto-download and work immediately
- **Port 11540** – Default port (accessed via LlamaFarm Server on port 14345)

### Quick Setup

**1. Start Universal Runtime server:**
```bash
# From project root (recommended)
nx start universal-runtime

# Or with custom port
LF_RUNTIME_PORT=8080 nx start universal-runtime
```

**2. Configure your project:**
```yaml
runtime:
  models:
    - name: chat
      description: "Fast small language model"
      provider: universal
      model: microsoft/phi-2
      base_url: http://127.0.0.1:11540
      transformers:
        device: auto              # auto, cuda, mps, cpu
        dtype: auto               # auto, fp16, fp32, bf16
        trust_remote_code: true
```

**3. Start chatting:**
```bash
lf chat --model chat "Explain quantum computing"
```

### Supported Model Formats

**Current Support (Production):**
- **HuggingFace Transformers** – All PyTorch text models (GPT-2, Llama, Mistral, Qwen, Phi, BERT, etc.)
- **HuggingFace Diffusers** – All PyTorch diffusion models (Stable Diffusion, SDXL, FLUX)
- **GGUF Models** – Quantized models via llama.cpp (supports offline loading from HuggingFace cache)
- **Model Types**: Text Generation, Embeddings, Image Generation, Vision Classification, Audio Processing, Multimodal

**Coming Soon:**
- **ONNX Runtime** – 2-5x faster inference with automatic model conversion
- **TensorRT** – GPU-optimized inference for NVIDIA hardware

### GGUF Model Configuration

Universal Runtime supports GGUF models via llama.cpp with full parameter control. This is especially useful for **memory-constrained devices** like Jetson Orin Nano (8GB shared memory).

**Key Features:**
- **Offline Loading**: Models cached locally are used without network calls
- **Memory Guard**: Automatic batch size reduction when available memory is low
- **Full Parameter Passthrough**: Configure all llama.cpp parameters via `extra_body`

#### GGUF Parameters Reference

| Parameter | Type | Description |
|-----------|------|-------------|
| `n_ctx` | int | Context window size. Lower = less memory. Auto-detected if not set. |
| `n_batch` | int | Batch size for prompt processing. Lower values (512) reduce memory. |
| `n_gpu_layers` | int | GPU layer count. `-1` = all layers on GPU. |
| `n_threads` | int | CPU thread count. Auto-detected if not set. |
| `flash_attn` | bool | Enable flash attention for faster inference. |
| `use_mmap` | bool | Memory-map model file. Recommended for large models. |
| `use_mlock` | bool | Lock model in RAM. Set `false` on constrained devices. |
| `cache_type_k` | string | KV cache key quantization: `f32`, `f16`, `q8_0`, `q4_0`, etc. |
| `cache_type_v` | string | KV cache value quantization. `q4_0` reduces cache by ~4x. |

#### Example: Jetson Orin Nano Configuration

```yaml
runtime:
  models:
    - name: qwen3-8b
      provider: universal
      model: unsloth/Qwen3-8B-GGUF:Q4_K_M
      base_url: http://127.0.0.1:11540
      extra_body:
        n_ctx: 2048          # Small context to save KV cache memory
        n_batch: 512         # Reduced batch for smaller compute buffer
        n_gpu_layers: -1     # Full GPU offload
        flash_attn: true     # Enable flash attention
        use_mmap: true       # Memory-map for efficient swapping
        use_mlock: false     # Allow OS memory management
        cache_type_k: q4_0   # Quantize KV cache keys
        cache_type_v: q4_0   # Quantize KV cache values
```

### Multi-Model Setup Example

```yaml
runtime:
  default_model: balanced

  models:
    # Fast chat for quick responses
    - name: fast
      provider: universal
      model: Qwen/Qwen2.5-0.5B-Instruct
      base_url: http://127.0.0.1:11540
      transformers:
        device: auto
        dtype: auto

    # Balanced chat for quality
    - name: balanced
      provider: universal
      model: microsoft/phi-2
      base_url: http://127.0.0.1:11540

    # Embeddings for RAG
    - name: embedder
      provider: universal
      model: sentence-transformers/all-MiniLM-L6-v2
      base_url: http://127.0.0.1:11540
      transformers:
        model_type: embedding

    # Image generation
    - name: image-gen
      provider: universal
      model: stabilityai/stable-diffusion-2-1
      base_url: http://127.0.0.1:11540
      transformers:
        model_type: image
      diffusion:
        default_steps: 30
        default_guidance: 7.5
        default_size: "512x512"
```

### Hardware Acceleration

Universal Runtime automatically detects and optimizes for your hardware:

| Device | Configuration | Best For |
|--------|--------------|----------|
| **NVIDIA CUDA** | `device: cuda` | Best performance on NVIDIA GPUs |
| **Apple Metal** | `device: mps` | Optimized for Apple Silicon (M1/M2/M3) |
| **CPU** | `device: cpu` | Fallback for all platforms |
| **Auto** | `device: auto` | Recommended: auto-detect best device |

```yaml
transformers:
  device: auto    # Recommended: auto-detect best device
  dtype: auto     # auto (fp16 on GPU, fp32 on CPU)
```

### Specialized ML Capabilities

Beyond text generation, Universal Runtime provides specialized ML endpoints:

| Capability | Endpoint | Use Case |
|-----------|----------|----------|
| **OCR** | `POST /v1/vision/ocr` | Extract text from images/PDFs |
| **Document Extraction** | `POST /v1/vision/documents/extract` | Extract structured data from forms |
| **Text Classification** | `POST /v1/nlp/classify` | Sentiment analysis, routing |
| **Named Entity Recognition** | `POST /v1/nlp/ner` | Extract people, places, organizations |
| **Reranking** | `POST /v1/nlp/rerank` | Improve RAG retrieval accuracy |
| **Embeddings** | `POST /v1/nlp/embeddings` | Generate vector embeddings |
| **Anomaly Detection** | `POST /v1/ml/anomaly/*` | Detect outliers in data |

See the detailed guides:
- [Specialized ML Models](./specialized-ml.md) - OCR, document extraction, classification, NER, reranking
- [Anomaly Detection Guide](./anomaly-detection.md) - Complete anomaly detection documentation

---

## Cloud Providers

LlamaFarm supports any **OpenAI-compatible API** for production workloads. Use `provider: openai` with custom `base_url` and `api_key` to connect to cloud services.

### OpenAI

```yaml
runtime:
  models:
    - name: gpt4
      description: "OpenAI GPT-4o"
      provider: openai
      model: gpt-4o
      api_key: ${OPENAI_API_KEY}  # Set via environment variable

    - name: gpt4-mini
      description: "OpenAI GPT-4o Mini (cost-effective)"
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
```

:::tip Environment Variables
Store API keys in environment variables: `export OPENAI_API_KEY=sk-proj-xxx`
LlamaFarm automatically substitutes `${VAR_NAME}` in config files.
:::

### xAI Grok

```yaml
runtime:
  models:
    - name: grok
      description: "xAI Grok"
      provider: openai
      model: grok-beta
      base_url: https://api.x.ai/v1
      api_key: ${XAI_API_KEY}
```

### Together AI

```yaml
runtime:
  models:
    - name: together-llama
      description: "Llama 3.1 70B on Together AI"
      provider: openai
      model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
      base_url: https://api.together.xyz/v1
      api_key: ${TOGETHER_API_KEY}

    - name: together-mixtral
      description: "Mixtral 8x7B on Together AI"
      provider: openai
      model: mistralai/Mixtral-8x7B-Instruct-v0.1
      base_url: https://api.together.xyz/v1
      api_key: ${TOGETHER_API_KEY}
```

### Groq (Fast Inference)

```yaml
runtime:
  models:
    - name: groq-llama
      description: "Llama 3.1 on Groq (ultra-fast)"
      provider: openai
      model: llama-3.1-70b-versatile
      base_url: https://api.groq.com/openai/v1
      api_key: ${GROQ_API_KEY}

    - name: groq-mixtral
      description: "Mixtral on Groq"
      provider: openai
      model: mixtral-8x7b-32768
      base_url: https://api.groq.com/openai/v1
      api_key: ${GROQ_API_KEY}
```

### Fireworks AI

```yaml
runtime:
  models:
    - name: fireworks-llama
      description: "Llama 3.1 on Fireworks"
      provider: openai
      model: accounts/fireworks/models/llama-v3p1-70b-instruct
      base_url: https://api.fireworks.ai/inference/v1
      api_key: ${FIREWORKS_API_KEY}
```

### Mistral AI

```yaml
runtime:
  models:
    - name: mistral-large
      description: "Mistral Large"
      provider: openai
      model: mistral-large-latest
      base_url: https://api.mistral.ai/v1
      api_key: ${MISTRAL_API_KEY}

    - name: mistral-small
      description: "Mistral Small (cost-effective)"
      provider: openai
      model: mistral-small-latest
      base_url: https://api.mistral.ai/v1
      api_key: ${MISTRAL_API_KEY}
```

### Self-Hosted vLLM

```yaml
runtime:
  models:
    - name: vllm-local
      description: "Self-hosted vLLM"
      provider: openai
      model: mistral-7b
      base_url: http://localhost:8000/v1
      api_key: not-needed  # vLLM doesn't require auth by default
      instructor_mode: json
```

### LM Studio

```yaml
runtime:
  models:
    - name: lmstudio
      description: "LM Studio local model"
      provider: openai
      model: local-model  # Model name from LM Studio
      base_url: http://localhost:1234/v1
      api_key: not-needed
```

### Cloud Provider Reference

| Provider | Base URL | Model Examples |
|----------|----------|----------------|
| **OpenAI** | (default) | `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo` |
| **xAI Grok** | `https://api.x.ai/v1` | `grok-beta` |
| **Together AI** | `https://api.together.xyz/v1` | `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` |
| **Groq** | `https://api.groq.com/openai/v1` | `llama-3.1-70b-versatile`, `mixtral-8x7b-32768` |
| **Fireworks** | `https://api.fireworks.ai/inference/v1` | `accounts/fireworks/models/llama-v3p1-70b-instruct` |
| **Mistral AI** | `https://api.mistral.ai/v1` | `mistral-large-latest`, `mistral-small-latest` |
| **vLLM** | `http://localhost:8000/v1` | Your deployed model |
| **LM Studio** | `http://localhost:1234/v1` | Your loaded model |

---

## Other Local Runtimes

### Ollama

Ollama provides easy local model setup with quantized GGUF models. Great for getting started quickly.

```yaml
runtime:
  models:
    - name: ollama-fast
      description: "Fast Ollama model"
      provider: ollama
      model: gemma3:1b

    - name: ollama-powerful
      description: "More capable Ollama model"
      provider: ollama
      model: qwen3:8b
```

**Setup:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models
ollama pull gemma3:1b
ollama pull qwen3:8b
```

**Port:** 11434 (default)

### Lemonade

Lemonade is a high-performance local runtime that runs GGUF models with NPU/GPU acceleration. Excellent on Apple Silicon.

**Quick Setup:**

```bash
# 1. Install Lemonade SDK
uv pip install lemonade-sdk

# 2. Download a model
uv run lemonade-server-dev pull user.Qwen3-4B \
  --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M \
  --recipe llamacpp

# 3. Start Lemonade server
LEMONADE_MODEL=user.Qwen3-4B nx start lemonade
```

**Configure:**
```yaml
runtime:
  models:
    - name: lemon
      description: "Lemonade local model"
      provider: lemonade
      model: user.Qwen3-4B
      base_url: "http://127.0.0.1:11534/v1"
      lemonade:
        backend: llamacpp
        port: 11534
        context_size: 32768
```

**Port:** 11534 (default)

**Key Features:**
- Hardware acceleration: Metal (macOS), CUDA (NVIDIA), Vulkan (AMD/Intel)
- Multiple backends: llamacpp (GGUF), ONNX, Transformers
- OpenAI-compatible API

---

## Agent Handlers

LlamaFarm selects an agent handler based on configuration:

- **Simple chat** – direct user/system prompts, suitable for models without tool support.
- **Structured chat** – uses instructor modes (`tools`, `json`) for models that support function/tool calls.
- **RAG chat** – augments prompts with retrieved context, citations, and guardrails.
- **Classifier / Custom** – future handlers for specialized workflows.

Choose handler behaviour in your project configuration (e.g., advanced agents defined by the server). Ensure the model supports the required features—some small models (TinyLlama) don't handle tools, so stick with simple chat.

## Inline Tools with Dynamic Variables

Models can define tools inline in the config, and these tools support dynamic variable substitution:

```yaml
runtime:
  models:
    - name: assistant
      provider: universal
      model: llama3.2:3b
      tool_call_strategy: native_api
      tools:
        - type: function
          name: search_docs
          description: "Search {{company_name | the company}} documentation"
          parameters:
            type: object
            properties:
              query:
                type: string
                description: "Search query for {{department | general}} topics"
            required:
              - query
```

Pass values at request time via the `variables` field:

```bash
curl -X POST .../chat/completions -d '{
  "messages": [{"role": "user", "content": "Find shipping info"}],
  "variables": {"company_name": "Acme Corp", "department": "logistics"}
}'
```

See [Dynamic Variables](../prompts/index.md#dynamic-variables) for full syntax documentation.

---

## Extending Provider Support

To add a new provider enum:

1. Update `config/schema.yaml` (`runtime.provider` enum).
2. Regenerate datamodels via `config/generate_types.py`.
3. Map the provider to an execution path in the server runtime service.
4. Update CLI defaults or additional flags if needed.
5. Document usage in this guide.

## Upcoming Roadmap

- **Advanced agent handler configuration** – choose handlers per command and dataset.
- **Fine-tuning pipeline integration** – track status in the roadmap.

## Next Steps

- [Specialized ML Models](./specialized-ml.md) – OCR, document extraction, and more.
- [Anomaly Detection](./anomaly-detection.md) – detect outliers in your data.
- [Configuration Guide](../configuration/index.md) – runtime schema details.
- [Extending runtimes](../extending/index.md#extend-runtimes) – step-by-step provider integration.
- [Prompts](../prompts/index.md) – control how system prompts interact with runtime capabilities.
