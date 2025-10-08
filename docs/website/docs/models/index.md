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

Choose handler behaviour in your project configuration (e.g., advanced agents defined by the server). Ensure the model supports the required features—some small models (TinyLlama) don’t handle tools, so stick with simple chat.

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
- **Port 11534**: Default port (different from LlamaFarm's 8000 and Ollama's 11434)

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

## Extending Provider Support

To add a new provider enum:

1. Update `config/schema.yaml` (`runtime.provider` enum).
2. Regenerate datamodels via `config/generate-types.sh`.
3. Map the provider to an execution path in the server runtime service.
4. Update CLI defaults or additional flags if needed.
5. Document usage in this guide.

## Upcoming Roadmap

- **Advanced agent handler configuration** – choose handlers per command and dataset.
- **Fine-tuning pipeline integration** – track status in the roadmap.

## Next Steps

- [Configuration Guide](../configuration/index.md) – runtime schema details.
- [Extending runtimes](../extending/index.md#extend-runtimes) – step-by-step provider integration.
- [Prompts](../prompts/index.md) – control how system prompts interact with runtime capabilities.
