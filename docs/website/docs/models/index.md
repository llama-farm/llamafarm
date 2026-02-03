---
title: Models & Runtime
sidebar_position: 7
---

# Models & Runtime

LlamaFarm focuses on inference rather than fine-tuning. The runtime section of `llamafarm.yaml` describes how chat completions are executed—whether against local Ollama, Lemonade, a vLLM gateway, or a remote hosted provider.

## Multi-Model Support

LlamaFarm supports configuring multiple models in a single project. You can switch between models via CLI or API:

```yaml
runtime:
  default_model: fast  # Which model to use by default

  models:
    fast:
      description: "Fast Ollama model for quick responses"
      provider: ollama
      model: gemma3:1b
      prompt_format: unstructured

    powerful:
      description: "More capable model"
      provider: ollama
      model: qwen3:8b
```

**Using multi-model:**
- CLI: `lf chat --model powerful "your question"`
- CLI: `lf models list` (shows all available models)
- API: `POST /v1/projects/{ns}/{id}/chat/completions` with `{"model": "powerful", ...}`

**Legacy single-model configs are still supported** and automatically converted internally.

## Runtime Responsibilities

- Route chat requests to the configured provider.
- Respect instructor modes (`tools`, `json`, `md_json`, etc.) when available.
- Surface provider errors directly (incorrect model name, missing API key).
- Cooperate with agent handlers (simple chat, structured output, RAG-aware prompts).

## Choosing a Provider

| Use Case | Configuration |
| -------- | ------------- |
| **Local models (Ollama)** | `provider: ollama` (omit API key). Supports models pulled via `ollama pull`. |
| **Local models (Lemonade)** | `provider: lemonade` with GGUF models. Hardware-accelerated (NPU/GPU). See [Lemonade Setup](#lemonade-runtime) below. |
| **Self-hosted vLLM / OpenAI-compatible** | `provider: openai`, set `base_url` to your gateway, `api_key` as required. |
| **Hosted APIs (OpenAI, Anthropic via proxy, Together, LM Studio)** | `provider: openai`, set `base_url` if not using api.openai.com, provide API key. |

Example using vLLM locally:

```yaml
runtime:
  models:
    - name: vllm-model
      provider: openai
      model: mistral-small
      base_url: http://localhost:8000/v1
      api_key: sk-local-placeholder
      instructor_mode: json
      default: true
```

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

## Lemonade Runtime

Lemonade is a high-performance local runtime that runs GGUF models with NPU/GPU acceleration. It's an alternative to Ollama with excellent performance on Apple Silicon and other hardware.

### Quick Setup

**1. Install Lemonade SDK:**
```bash
uv pip install lemonade-sdk
```

**2. Download a model:**
```bash
# Balanced 4B model (recommended)
uv run lemonade-server-dev pull user.Qwen3-4B \
  --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M \
  --recipe llamacpp
```

**3. Start Lemonade server:**
```bash
# From the llamafarm project root
LEMONADE_MODEL=user.Qwen3-4B nx start lemonade
```

> **Note:** The `nx start lemonade` command automatically picks up configuration from your `llamafarm.yaml`. Currently, Lemonade must be manually started. In the future, Lemonade will run as a container and be auto-started by the LlamaFarm server.

**4. Configure your project:**
```yaml
runtime:
  models:
    - name: lemon
      description: "Lemonade local model"
      provider: lemonade
      model: user.Qwen3-4B
      base_url: "http://127.0.0.1:11534/v1"
      default: true
      lemonade:
        backend: llamacpp  # llamacpp (GGUF), onnx, or transformers
        port: 11534
        context_size: 32768
```

### Key Features

- **Hardware acceleration**: Automatically detects and uses Metal (macOS), CUDA (NVIDIA), Vulkan (AMD/Intel), or CPU
- **Multiple backends**: llamacpp (GGUF models), ONNX, or Transformers (PyTorch)
- **OpenAI-compatible API**: Drop-in replacement for OpenAI-compatible endpoints
- **Port 11534**: Default port (different from LlamaFarm's 14345 and Ollama's 11434)

### Multi-Model with Lemonade

Run multiple Lemonade instances on different ports:

```yaml
runtime:
  models:
    - name: lemon-fast
      provider: lemonade
      model: user.Qwen3-0.6B
      base_url: "http://127.0.0.1:11534/v1"
      lemonade:
        port: 11534

    - name: lemon-powerful
      provider: lemonade
      model: user.Qwen3-8B
      base_url: "http://127.0.0.1:11535/v1"
      lemonade:
        port: 11535
```

Start each instance in a separate terminal (from the llamafarm project root):
```bash
# Terminal 1
LEMONADE_MODEL=user.Qwen3-0.6B LEMONADE_PORT=11534 nx start lemonade

# Terminal 2
LEMONADE_MODEL=user.Qwen3-8B LEMONADE_PORT=11535 nx start lemonade
```

> **Note:** In the future, Lemonade instances will run as containers and be auto-started by the LlamaFarm server.

### More Information

For detailed setup instructions, model recommendations, and troubleshooting, see:
- [Lemonade Quickstart](#quick-setup)
- [Lemonade Runtime overview](#lemonade-runtime)

## Universal Runtime

The Universal Runtime is LlamaFarm's most versatile runtime provider, supporting **any HuggingFace model** through PyTorch Transformers and Diffusers. Unlike Ollama (GGUF-only) or Lemonade (optimized quantized models), Universal Runtime provides access to the entire HuggingFace Hub ecosystem.

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

#### Memory Estimation

| Parameter | Memory Impact |
|-----------|--------------|
| `n_ctx: 2048` | ~256MB KV cache |
| `n_ctx: 8192` | ~1GB KV cache |
| `n_batch: 2048` | ~1.2GB compute buffer |
| `n_batch: 512` | ~300MB compute buffer |
| `cache_type: q4_0` | ~4x reduction vs f16 |

#### Memory Guard (Automatic)

When available memory drops below 3GB, Universal Runtime automatically:
1. Reduces `n_batch` from 2048 to 512
2. Logs a warning with the adjustment

This prevents out-of-memory crashes on constrained devices.

#### Offline Operation

GGUF models support fully offline operation:
1. Download model once (requires network)
2. Model is cached in `~/.cache/huggingface/hub/`
3. Subsequent loads use cache without network
4. Works on air-gapped systems

```bash
# First run downloads the model
curl -X POST http://localhost:11540/v1/chat/completions \
  -d '{"model": "unsloth/Qwen3-1.7B-GGUF:Q4_K_M", "messages": [...]}'

# Subsequent runs work offline
# (disconnect network and it still works)
```

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
    - name: phi-2
      description: "Fast small language model"
      provider: universal
      model: microsoft/phi-2
      base_url: http://127.0.0.1:11540
      transformers:
        device: auto              # auto, cuda, mps, cpu
        dtype: auto               # auto, fp16, fp32, bf16
        trust_remote_code: true
        model_type: text          # text, embedding, image
```

**3. Start chatting:**
```bash
lf chat --model phi-2 "Explain quantum computing"
```

### Configuration Examples

**Example 1: Multi-Model Setup (Chat + Embeddings + Images)**

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
      transformers:
        device: auto
        dtype: auto

    # Embeddings for RAG
    - name: embedder
      provider: universal
      model: sentence-transformers/all-MiniLM-L6-v2
      base_url: http://127.0.0.1:11540
      transformers:
        device: auto
        dtype: auto
        model_type: embedding

    # Image generation
    - name: image-gen
      provider: universal
      model: stabilityai/stable-diffusion-2-1
      base_url: http://127.0.0.1:11540
      transformers:
        device: auto
        dtype: auto
        model_type: image
      diffusion:
        default_steps: 30
        default_guidance: 7.5
        default_size: "512x512"
        scheduler: euler
        enable_optimizations: true
```

**Example 2: RAG with Universal Embeddings**

```yaml
runtime:
  models:
    - name: chat
      provider: universal
      model: microsoft/phi-2
      base_url: http://127.0.0.1:11540

    - name: embedder
      provider: universal
      model: nomic-ai/nomic-embed-text-v1.5
      base_url: http://127.0.0.1:11540
      transformers:
        model_type: embedding

rag:
  databases:
    - name: main_database
      type: ChromaStore
      embedding_strategies:
        - name: default_embeddings
          type: UniversalEmbedder
          config:
            model: nomic-ai/nomic-embed-text-v1.5
            dimension: 768
            batch_size: 16
```

### Hardware Acceleration

Universal Runtime automatically detects and optimizes for your hardware:

**Detection Priority:**
1. **NVIDIA CUDA** – Best performance on NVIDIA GPUs
2. **Apple Metal (MPS)** – Optimized for Apple Silicon (M1/M2/M3)
3. **CPU** – Fallback for all platforms

**Device Configuration:**
```yaml
transformers:
  device: auto    # Recommended: auto-detect best device
  device: cuda    # Force CUDA (NVIDIA)
  device: mps     # Force Metal (Apple Silicon)
  device: cpu     # Force CPU (compatible everywhere)
```

**Data Type Configuration:**
```yaml
transformers:
  dtype: auto     # auto (fp16 on GPU, fp32 on CPU)
  dtype: fp16     # Half precision (faster, less memory)
  dtype: fp32     # Full precision (highest quality)
  dtype: bf16     # BFloat16 (NVIDIA Ampere+)
```

### Environment Variables

```bash
# Server configuration
UNIVERSAL_RUNTIME_HOST=127.0.0.1
UNIVERSAL_RUNTIME_PORT=11540

# Model caching
TRANSFORMERS_CACHE=~/.cache/huggingface
HF_TOKEN=hf_xxxxx  # For gated models (Llama, etc.)

# Device control
TRANSFORMERS_FORCE_CPU=1     # Force CPU mode
TRANSFORMERS_SKIP_MPS=1      # Skip MPS, use CPU instead
```

### API Usage

Universal Runtime provides an OpenAI-compatible API:

**Chat Completions:**
```bash
curl -X POST http://localhost:11540/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/phi-2",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Embeddings:**
```bash
curl -X POST http://localhost:11540/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "model": "sentence-transformers/all-MiniLM-L6-v2",
    "input": "Hello world"
  }'
```

**Image Generation:**
```bash
curl -X POST http://localhost:11540/v1/images/generations \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a serene mountain lake at sunset",
    "model": "stabilityai/stable-diffusion-2-1",
    "size": "512x512"
  }'
```

### Supported Model Categories

1. **Text Generation (CausalLM)** – GPT-2, Llama, Mistral, Qwen, Phi
2. **Embeddings (Encoder)** – BERT, sentence-transformers, BGE, nomic-embed
3. **Image Generation (Diffusion)** – Stable Diffusion, SDXL, FLUX
4. **Vision Classification** – ViT, CLIP, DINOv2
5. **Audio Processing** – Whisper, Wav2Vec2
6. **Multimodal (Vision-Language)** – BLIP, LLaVA, Florence

### Key Differences: Universal vs Other Runtimes

| Feature | Universal Runtime | Ollama | Lemonade |
|---------|------------------|--------|----------|
| **Model Format** | PyTorch (Transformers) | GGUF (llama.cpp) | GGUF (llama.cpp) |
| **Model Source** | HuggingFace Hub | Ollama library | HuggingFace GGUF |
| **Model Types** | 6 types (text, image, audio, etc.) | Text only | Text only |
| **Optimization** | PyTorch native | Quantized CPU/GPU | NPU/GPU accelerated |
| **Memory Usage** | Higher (full precision) | Lower (quantized) | Lower (quantized) |
| **Setup** | Auto-download from HF | `ollama pull` | `lemonade-server-dev pull` |
| **Default Port** | 11540 | 11434 | 11534 |
| **Best For** | Flexibility, multimodal | CPU speed, ease of use | NPU/GPU acceleration |

### When to Use Universal Runtime

**Choose Universal Runtime when you need:**
- Access to any HuggingFace model (latest research models, domain-specific fine-tunes)
- Multimodal capabilities (images, audio, vision)
- Embedding generation for RAG
- Custom model configurations (LoRA, adapters)

**Choose Ollama when you need:**
- Fast CPU inference with quantized models
- Simplest setup experience
- Established model library

**Choose Lemonade when you need:**
- Maximum performance on Apple Silicon
- NPU/GPU acceleration
- GGUF model optimization

### Troubleshooting

**Model not found:**
```bash
# Models auto-download from HuggingFace on first use
# Check internet connection and HF_TOKEN for gated models
export HF_TOKEN=hf_your_token_here
```

**Out of memory:**
```bash
# Use smaller models or force CPU mode
TRANSFORMERS_FORCE_CPU=1 nx start universal-runtime

# Or reduce batch size in config
transformers:
  batch_size: 1
```

**Slow inference:**
```bash
# Ensure GPU is detected
# Check device selection in logs
# Consider using GGUF models with Ollama/Lemonade instead
```

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

## Specialized ML Models

Beyond text generation, the Universal Runtime provides specialized ML capabilities:

| Capability | Endpoint | Use Case |
|-----------|----------|----------|
| **OCR** | `POST /v1/ocr` | Extract text from images/PDFs |
| **Document Extraction** | `POST /v1/documents/extract` | Extract structured data from forms |
| **Text Classification** | `POST /v1/classify` | Sentiment analysis, routing |
| **Named Entity Recognition** | `POST /v1/ner` | Extract people, places, organizations |
| **Reranking** | `POST /v1/rerank` | Improve RAG retrieval accuracy |
| **Anomaly Detection** | `POST /v1/anomaly/*` | Detect outliers in data |

See the detailed guides:
- [Specialized ML Models](./specialized-ml.md) - OCR, document extraction, classification, NER, reranking
- [Anomaly Detection Guide](./anomaly-detection.md) - Complete anomaly detection documentation

## Next Steps

- [Specialized ML Models](./specialized-ml.md) – OCR, document extraction, and more.
- [Anomaly Detection](./anomaly-detection.md) – detect outliers in your data.
- [Configuration Guide](../configuration/index.md) – runtime schema details.
- [Extending runtimes](../extending/index.md#extend-runtimes) – step-by-step provider integration.
- [Prompts](../prompts/index.md) – control how system prompts interact with runtime capabilities.
