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
voice: { ... }
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
  base_url: http://localhost:14345/v1
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
| `provider`             | enum (`openai`, `ollama`, `lemonade`, `universal`) | ✅                                                                         | `openai` for OpenAI-compatible APIs, `ollama` for local Ollama, `lemonade` for local GGUF models with NPU/GPU support, `universal` for the Universal Runtime |
| `model`                | string                                | ✅                                                                         | Model identifier understood by the provider                                                                            |
| `description`          | string                                | Optional                                                                   | Human-readable description of the model                                                                                |
| `default`              | boolean                               | Optional                                                                   | Set to `true` to make this the default model (alternative to `default_model`)                                         |
| `base_url`             | string or null                        | ⚠️ Required for non-default hosts (vLLM, Together, Lemonade)              | API endpoint URL                                                                                                       |
| `api_key`              | string or null                        | ⚠️ Required for most hosted providers. Use `.env` + environment variables | Authentication key                                                                                                     |
| `instructor_mode`      | string or null                        | Optional                                                                   | `json`, `md_json`, `tools` for structured output modes                                                                 |
| `prompt_format`        | string                                | Optional                                                                   | `unstructured` or other format                                                                                         |
| `model_api_parameters` | object                                | Optional                                                                   | Passthrough parameters (temperature, top_p, etc.)                                                                      |
| `lemonade`             | object                                | ⚠️ Required for `provider: lemonade`                                      | Lemonade-specific configuration (see below)                                                                            |
| `extra_body`           | object                                | Optional                                                                   | Provider-specific parameters (see `n_ctx` below)                                                                       |
| `encoder_config`       | object                                | Optional                                                                   | Configuration for BERT-style encoder models (Universal runtime only)                                                   |
| `tool_call_strategy`   | enum                                  | `native_api`                                                               | `native_api` or `prompt_based` for tool calling strategy                                                               |
| `mcp_servers`          | array                                 | Optional                                                                   | List of MCP server names to use (omit for all, empty for none)                                                         |

**extra_body fields (Universal runtime):**

| Field   | Type    | Default | Description                                              |
| ------- | ------- | ------- | -------------------------------------------------------- |
| `n_ctx` | integer | auto    | Context window size for GGUF models. Auto-detected if not specified. |

**encoder_config fields (Universal runtime):**

| Field                | Type    | Default     | Description                                              |
| -------------------- | ------- | ----------- | -------------------------------------------------------- |
| `max_length`         | integer | auto        | Maximum sequence length (ModernBERT: 8192, classic: 512) |
| `use_flash_attention`| boolean | `true`      | Enable Flash Attention 2 for faster inference            |
| `task`               | enum    | `embedding` | `embedding`, `classification`, `reranking`, `ner`        |

**Lemonade-specific fields:**

| Field          | Type   | Required | Description                                  |
| -------------- | ------ | -------- | -------------------------------------------- |
| `backend`      | string | ✅       | `llamacpp`, `onnx`, or `transformers`        |
| `port`         | number | ✅       | Port number (default: 11534)                 |
| `context_size` | number | Optional | Context window size (default: 32768)         |

> **Extending providers:** To add a new provider enum, update `config/schema.yaml`, regenerate types via `config/generate_types.py`, and implement routing in the server/CLI. See [Extending runtimes](../extending/index.md#extend-runtimes).

### Prompts

Prompts are named sets of messages that seed instructions for each session. Prompts support **dynamic variable substitution** using Jinja2-style `{{variable}}` syntax.

```yaml
prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are a supportive assistant. Cite documents when relevant.
```

- Each prompt has a `name` and a list of `messages` with `role` and `content`.
- Roles can be `system`, `user`, or `assistant` (anything supported by the runtime).
- Models can select which prompt sets to use via `prompts: [list of names]`; if omitted, all prompts stack in definition order.
- Prompts are appended before user input; combine with RAG context via the RAG guide.

#### Dynamic Variable Substitution

Use Jinja2-style `{{variable}}` syntax to inject values at request time:

```yaml
prompts:
  - name: personalized
    messages:
      - role: system
        content: |
          You are a customer service assistant for {{company_name | Acme Corp}}.
          The customer's name is {{user_name | Valued Customer}}.
          Account tier: {{account_tier | standard}}
```

**Syntax:**

| Pattern | Description | Example |
|---------|-------------|---------|
| `{{variable}}` | Required variable | `Hello {{name}}` → Error if `name` not provided |
| `{{variable \| default}}` | Variable with default | `Hello {{name \| Guest}}` → `Hello Guest` if missing |
| `{{ variable }}` | Whitespace allowed | Same as above |

**Passing variables via API:**

```bash
curl -X POST http://localhost:14345/v1/projects/ns/project/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi"}],
    "variables": {
      "company_name": "TechCorp",
      "user_name": "Alice",
      "account_tier": "premium"
    }
  }'
```

Variables can be used in:
- **Prompt message content** - Personalize system instructions
- **Tool descriptions** - Customize tool behavior per request
- **Tool parameter descriptions** - Dynamic parameter help text

See the [Prompts Guide](../prompts/index.md) for detailed examples.

### RAG Configuration

The `rag` section mirrors [`rag/schema.yaml`](/rag/schema.yaml). It defines databases and data-processing strategies.

```yaml
components:
  embedding_strategies:
    - name: default_embeddings
      type: UniversalEmbedder
      config:
        model: sentence-transformers/all-MiniLM-L6-v2
  retrieval_strategies:
    - name: semantic_search
      type: BasicSimilarityStrategy
      config:
        top_k: 5
  parsers:
    - name: pdf_parser
      type: PDFParser_LlamaIndex
      config:
        chunk_size: 1500
        chunk_overlap: 200
  defaults:
    embedding_strategy: default_embeddings
    retrieval_strategy: semantic_search
    parser: pdf_parser

rag:
  databases:
    - name: main_db
      type: ChromaStore
      # Reuse components by reference (defaults also apply when fields are omitted)
      embedding_strategy: default_embeddings
      retrieval_strategy: semantic_search
  data_processing_strategies:
    - name: pdf_ingest
      parsers:
        - pdf_parser         # reference to components.parsers
      extractors:
        - type: HeadingExtractor
        - type: ContentStatisticsExtractor
```

Inline (legacy) configs are still supported:

```yaml
rag:
  databases:
    - name: inline_db
      type: ChromaStore
      default_embedding_strategy: default_embeddings
      embedding_strategies:
        - name: default_embeddings
          type: UniversalEmbedder
          config:
            model: sentence-transformers/all-MiniLM-L6-v2
      retrieval_strategies:
        - name: semantic_search
          type: BasicSimilarityStrategy
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
- Inline parsers can omit `name` (only `type`/`config` are required); `name` is required only for reusable parsers declared under `components.parsers` so they can be referenced by string or set as defaults. This keeps older inline configs valid while allowing named, reusable components.
- For a complete field reference, see the [RAG Guide](../rag/index.md).

Defaults and persistence:
- `components.defaults` are used when a database or processing strategy omits a field.
- Server resolves references at load time and persists fully inlined configs; GET responses return the expanded strategies (no reference strings).

### Memory Configuration

The `memory` section configures optional memory stores for working memory, time-series data, spatial data, and graph relationships.

```yaml
memory:
  default_store: main_memory
  stores:
    - name: main_memory
      working_memory:
        enabled: true
        ttl_seconds: 3600
        max_records: 10000
      timeseries:
        enabled: true
        retention_days: 30
      spatial:
        enabled: false
      graph:
        enabled: true
        max_path_depth: 10
        entity_extraction: true
        relationship_extraction: false
      consolidation:
        enabled: true
        interval_seconds: 300
        min_records: 10
        batch_size: 100
```

#### Memory Store Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Unique identifier for this memory store |
| `working_memory` | object | Short-term memory buffer configuration |
| `timeseries` | object | Time-series store configuration |
| `spatial` | object | Geo-spatial store configuration |
| `graph` | object | Graph store configuration |
| `consolidation` | object | Memory consolidation settings |

#### Working Memory

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable working memory |
| `ttl_seconds` | integer | `3600` | Time-to-live for records |
| `max_records` | integer | `10000` | Maximum records before auto-prune |

#### Time-series

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable time-series store |
| `retention_days` | integer | `30` | Days to retain data |

#### Spatial

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable spatial store |
| `retention_days` | integer | `30` | Days to retain data |
| `index_type` | string | `rtree` | `rtree` or `geohash` |

#### Graph

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable graph store |
| `max_path_depth` | integer | `10` | Maximum depth for path finding |
| `entity_extraction` | boolean | `true` | Extract entities using NER |
| `relationship_extraction` | boolean | `false` | Extract relationships via LLM |

#### Consolidation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | boolean | `true` | Enable consolidation |
| `interval_seconds` | integer | `300` | Consolidation interval |
| `min_records` | integer | `10` | Minimum records before consolidation |
| `batch_size` | integer | `100` | Batch size for consolidation |
| `prune_after_consolidate` | boolean | `true` | Prune after consolidation |
| `extract_summaries` | boolean | `false` | Create embeddings from consolidated data |

---

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

### Voice

The `voice` section configures real-time voice chat via WebSocket. This enables a full-duplex voice assistant pipeline: Speech In → STT → LLM → TTS → Speech Out.

```yaml
voice:
  enabled: true
  llm_model: chat-model      # Reference to runtime.models[].name

  tts:
    model: kokoro            # TTS model ID
    voice: af_heart          # Voice ID
    speed: 1.0               # Speed multiplier (0.5-2.0)

  stt:
    model: base              # Whisper model size
    language: en             # Language code
```

#### Voice Fields

| Field       | Type    | Default | Description                                                              |
| ----------- | ------- | ------- | ------------------------------------------------------------------------ |
| `enabled`   | boolean | `true`  | Enable or disable the voice chat endpoint                                |
| `llm_model` | string  | —       | Reference to a model name in `runtime.models[]` for voice conversations  |
| `tts`       | object  | —       | Text-to-speech configuration                                             |
| `stt`       | object  | —       | Speech-to-text configuration                                             |

#### TTS (Text-to-Speech)

| Field   | Type   | Default     | Description                                     |
| ------- | ------ | ----------- | ----------------------------------------------- |
| `model` | string | `kokoro`    | TTS model ID                                    |
| `voice` | string | `af_heart`  | Voice ID (see available voices below)           |
| `speed` | number | `1.0`       | Speech speed multiplier (0.5-2.0)               |

**Available Voices:**

| Voice ID      | Description               |
| ------------- | ------------------------- |
| `af_heart`    | Heart (American Female) - default |
| `af_bella`    | Bella (American Female)   |
| `af_nicole`   | Nicole (American Female)  |
| `af_sarah`    | Sarah (American Female)   |
| `af_sky`      | Sky (American Female)     |
| `am_adam`     | Adam (American Male)      |
| `am_michael`  | Michael (American Male)   |
| `bf_emma`     | Emma (British Female)     |
| `bf_isabella` | Isabella (British Female) |
| `bm_george`   | George (British Male)     |
| `bm_lewis`    | Lewis (British Male)      |

#### STT (Speech-to-Text)

| Field      | Type   | Default | Description                                         |
| ---------- | ------ | ------- | --------------------------------------------------- |
| `model`    | string | `base`  | Whisper model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `language` | string | `en`    | Language code for transcription                     |

**STT Model Comparison:**

| Model      | Size   | Speed    | Accuracy |
| ---------- | ------ | -------- | -------- |
| `tiny`     | 39M    | Fastest  | Lower    |
| `base`     | 74M    | Fast     | Good     |
| `small`    | 244M   | Medium   | Better   |
| `medium`   | 769M   | Slower   | High     |
| `large-v3` | 1.5B   | Slowest  | Highest  |

#### Example: Complete Voice Configuration

```yaml
version: v1
name: voice-assistant
namespace: default

prompts:
  - name: voice_system
    messages:
      - role: system
        content: |
          You are a friendly voice assistant. Keep responses concise
          and conversational. Avoid long lists or complex formatting
          since your output will be spoken aloud.

runtime:
  default_model: voice-model
  models:
    - name: voice-model
      provider: universal
      model: unsloth/Qwen3-4B-GGUF:Q4_K_M
      base_url: http://localhost:11540/v1
      prompts: [voice_system]
      model_api_parameters:
        temperature: 0.7
        max_tokens: 256

voice:
  enabled: true
  llm_model: voice-model
  tts:
    model: kokoro
    voice: am_adam
    speed: 1.1
  stt:
    model: small
    language: en
```

The prompts attached to the referenced LLM model are automatically applied to voice conversations.

---

## Validation & Errors

- The CLI enforces schema validation when loading configs. Missing runtime fields raise `Error: runtime.provider is required`.
- Use `lf chat --curl` to inspect the raw request if responses look wrong (verify prompts and RAG toggles).
- The server logs include full validation errors if API calls fail due to config mismatches.

## Extending the Schema

1. Edit `config/schema.yaml` or `rag/schema.yaml` to add new enums/properties.
2. Run `config/generate_types.py` to regenerate Pydantic/Go datamodels.
3. Update server/CLI logic to accept the new fields.
4. Document the addition in this guide and the Extending section.

Example: To support a new provider `together`, add it to the `provider` enum, regenerate types, and update runtime selection to issue HTTP requests to Together’s API.

## Best Practices

- Keep secrets out of YAML; use environment variables and reference them at runtime.
- Version control your config; treat `llamafarm.yaml` like application code.
- Use separate namespaces or configs for dev/staging/prod to avoid cross-talk.
- Document uncommon parser/extractor choices for future maintainers.

Need concrete samples? Check the [Example configs](./example-configs.md) and the examples in the repo (`examples/fda_rag/llamafarm-example.yaml`).
