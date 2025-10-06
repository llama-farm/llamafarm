---
title: Configuration Guide
sidebar_position: 4
---

# Configuration Guide

Every LlamaFarm project is defined by a single file: `llamafarm.yaml`. The server validates it against JSON Schema, so missing fields surface as errors instead of hidden defaults. This guide explains each section and shows how to extend the schema responsibly.

## File Layout

```yaml
version: v1
name: my-project
namespace: default
runtime: { ... }
prompts: [...]
rag: { ... }
datasets: [...]
```

### Metadata

| Field       | Type   | Required  | Notes                                              |
| ----------- | ------ | --------- | -------------------------------------------------- |
| `version`   | string | ✅ (`v1`) | Schema version.                                    |
| `name`      | string | ✅        | Project identifier.                                |
| `namespace` | string | ✅        | Grouping for isolation (matches server namespace). |

### Runtime

Controls how chat completions are executed. LlamaFarm supports both **multi-model** (recommended) and **legacy single-model** configurations.

#### Multi-Model Configuration (Recommended)

Configure multiple models and switch between them via CLI or API:

```yaml
runtime:
  default_model: fast  # Which model to use by default

  models:
    fast:
      description: "Fast Ollama model"
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
- CLI: `lf models list`
- API: `POST /v1/projects/{ns}/{id}/chat/completions` with `{"model": "powerful", ...}`

#### Legacy Single-Model Configuration (Still Supported)

The original flat runtime configuration is automatically converted internally:

```yaml
runtime:
  provider: openai
  model: qwen2.5:7b
  base_url: http://localhost:8000/v1
  api_key: sk-local-placeholder
  instructor_mode: tools
  model_api_parameters:
    temperature: 0.2
```

#### Runtime Fields

**Multi-model format:**

| Field           | Type   | Required | Description                                   |
| --------------- | ------ | -------- | --------------------------------------------- |
| `default_model` | string | ✅       | Name of the default model to use              |
| `models`        | array  | ✅       | List of model configurations (see below)      |

**Per-model fields:**

| Field                  | Type                                  | Required                                                                   | Description                                                                                                            |
| ---------------------- | ------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `name`                 | string                                | ✅                                                                         | Unique identifier for this model                                                                                       |
| `provider`             | enum (`openai`, `ollama`, `lemonade`) | ✅                                                                         | `openai` for OpenAI-compatible APIs, `ollama` for local Ollama, `lemonade` for local GGUF models with NPU/GPU support |
| `model`                | string                                | ✅                                                                         | Model identifier understood by the provider                                                                            |
| `description`          | string                                | Optional                                                                   | Human-readable description of the model                                                                                |
| `default`              | boolean                               | Optional                                                                   | Set to `true` to make this the default model (alternative to `default_model`)                                         |
| `base_url`             | string or null                        | ⚠️ Required for non-default hosts (vLLM, Together, Lemonade)              | API endpoint URL                                                                                                       |
| `api_key`              | string or null                        | ⚠️ Required for most hosted providers. Use `.env` + environment variables | Authentication key                                                                                                     |
| `instructor_mode`      | string or null                        | Optional                                                                   | `json`, `md_json`, `tools` for structured output modes                                                                 |
| `prompt_format`        | string                                | Optional                                                                   | `unstructured` or other format                                                                                         |
| `model_api_parameters` | object                                | Optional                                                                   | Passthrough parameters (temperature, top_p, etc.)                                                                      |
| `lemonade`             | object                                | ⚠️ Required for `provider: lemonade`                                      | Lemonade-specific configuration (see below)                                                                            |

**Lemonade-specific fields:**

| Field          | Type   | Required | Description                                  |
| -------------- | ------ | -------- | -------------------------------------------- |
| `backend`      | string | ✅       | `llamacpp`, `onnx`, or `transformers`        |
| `port`         | number | ✅       | Port number (default: 11534)                 |
| `context_size` | number | Optional | Context window size (default: 32768)         |

> **Extending providers:** To add a new provider enum, update `config/schema.yaml`, regenerate types via `config/generate-types.sh`, and implement routing in the server/CLI. See [Extending runtimes](../extending/index.md#extend-runtimes).

### Prompts

Prompts seed system/content instructions for each session.

```yaml
prompts:
  - role: system
    content: >-
      You are a supportive assistant. Cite documents when relevant.
```

- Roles can be `system`, `user`, or `assistant` (anything supported by the runtime).
- Prompts are appended before user input; combine with RAG context via the RAG guide.

### RAG Configuration

The `rag` section mirrors [`rag/schema.yaml`](/rag/schema.yaml). It defines databases and data-processing strategies.

```yaml
rag:
  databases:
    - name: main_db
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: semantic_search
      embedding_strategies:
        - name: default_embeddings
          type: OllamaEmbedder
          config:
            model: nomic-embed-text:latest
      retrieval_strategies:
        - name: semantic_search
          type: VectorRetriever
          config:
            top_k: 5
  data_processing_strategies:
    - name: pdf_ingest
      parsers:
        - type: PDFParser_LlamaIndex
          config:
            chunk_size: 1500
            chunk_overlap: 200
      extractors:
        - type: HeadingExtractor
        - type: ContentStatisticsExtractor
```

Key points:

- `databases` map to vector stores; choose from `ChromaStore` or `QdrantStore` by default.
- `embedding_strategies` and `retrieval_strategies` let you define hybrid or metadata-aware search.
- `data_processing_strategies` describe parser/extractor pipelines applied during ingestion.
- For a complete field reference, see the [RAG Guide](../rag/index.md).

### Datasets

`datasets` keep metadata about datasets you manage via the CLI.

```yaml
datasets:
  - name: research-notes
    data_processing_strategy: pdf_ingest
    database: main_db
    files:
      - 2d5fd8424e62c56cad39864fac9ecff7af9639cf211deb936a16dc05aca5b3ea
```

- `files` are SHA256 hashes tracked by the server.
- Not required, but useful for syncing dataset metadata across environments.

## Validation & Errors

- The CLI enforces schema validation when loading configs. Missing runtime fields raise `Error: runtime.provider is required`.
- Use `lf chat --curl` to inspect the raw request if responses look wrong (verify prompts and RAG toggles).
- The server logs include full validation errors if API calls fail due to config mismatches.

## Extending the Schema

1. Edit `config/schema.yaml` or `rag/schema.yaml` to add new enums/properties.
2. Run `config/generate-types.sh` to regenerate Pydantic/Go datamodels.
3. Update server/CLI logic to accept the new fields.
4. Document the addition in this guide and the Extending section.

Example: To support a new provider `together`, add it to the `provider` enum, regenerate types, and update runtime selection to issue HTTP requests to Together’s API.

## Best Practices

- Keep secrets out of YAML; use environment variables and reference them at runtime.
- Version control your config; treat `llamafarm.yaml` like application code.
- Use separate namespaces or configs for dev/staging/prod to avoid cross-talk.
- Document uncommon parser/extractor choices for future maintainers.

Need concrete samples? Check the [Example configs](./example-configs.md) and the examples in the repo (`examples/fda_rag/llamafarm-example.yaml`).
