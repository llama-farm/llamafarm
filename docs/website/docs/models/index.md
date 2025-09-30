---
title: Models & Runtime
sidebar_position: 7
---

# Models & Runtime

LlamaFarm currently focuses on inference rather than fine-tuning. The runtime section of `llamafarm.yaml` describes how chat completions are executed—whether against local Ollama, a vLLM gateway, or a remote hosted provider.

## Runtime Responsibilities

- Route chat requests to the configured provider.
- Respect instructor modes (`tools`, `json`, `md_json`, etc.) when available.
- Surface provider errors directly (incorrect model name, missing API key).
- Cooperate with agent handlers (simple chat, structured output, RAG-aware prompts).

## Choosing a Provider

| Use Case | Configuration |
| -------- | ------------- |
| **Local models (Ollama)** | `provider: ollama` (omit API key). Supports models pulled via `ollama pull`. |
| **Self-hosted vLLM / OpenAI-compatible** | `provider: openai`, set `base_url` to your gateway, `api_key` as required. |
| **Hosted APIs (OpenAI, Anthropic via proxy, Together, LM Studio)** | `provider: openai`, set `base_url` if not using api.openai.com, provide API key. |

Example using vLLM locally:

```yaml
runtime:
  provider: openai
  model: mistral-small
  base_url: http://localhost:8000/v1
  api_key: sk-local-placeholder
  instructor_mode: json
```

## Agent Handlers

LlamaFarm selects an agent handler based on configuration:

- **Simple chat** – direct user/system prompts, suitable for models without tool support.
- **Structured chat** – uses instructor modes (`tools`, `json`) for models that support function/tool calls.
- **RAG chat** – augments prompts with retrieved context, citations, and guardrails.
- **Classifier / Custom** – future handlers for specialized workflows.

Choose handler behaviour in your project configuration (e.g., advanced agents defined by the server). Ensure the model supports the required features—some small models (TinyLlama) don’t handle tools, so stick with simple chat.

## Extending Provider Support

To add a new provider enum:

1. Update `config/schema.yaml` (`runtime.provider` enum).
2. Regenerate datamodels via `config/generate-types.sh`.
3. Map the provider to an execution path in the server runtime service.
4. Update CLI defaults or additional flags if needed.
5. Document usage in this guide.

## Upcoming Roadmap

- **Multi-model configurations** – select models (fast, accurate, structured) per request.
- **Advanced agent handler configuration** – choose handlers per command and dataset.
- **Fine-tuning pipeline integration** – track status in the roadmap.

## Next Steps

- [Configuration Guide](../configuration/index.md) – runtime schema details.
- [Extending runtimes](../extending/index.md#extend-runtimes) – step-by-step provider integration.
- [Prompts](../prompts/index.md) – control how system prompts interact with runtime capabilities.
