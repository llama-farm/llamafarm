---
title: Embedders Reference
sidebar_position: 3
---

# Embedders Reference

Embedders convert text chunks into vector representations for semantic search. LlamaFarm supports multiple embedding providers to match your infrastructure and requirements.

## Quick Start

Embedders are configured within `embedding_strategies` in your database definition:

```yaml
rag:
  databases:
    - name: my_database
      embedding_strategies:
        - name: default_embeddings
          type: OllamaEmbedder
          config:
            model: nomic-embed-text
```

---

## UniversalEmbedder

Flexible embedder that works with Ollama and falls back to HuggingFace models.

**Best for:** Most use cases, automatic model management

```yaml
- name: default_embeddings
  type: UniversalEmbedder
  config:
    model: nomic-ai/nomic-embed-text-v2-moe
    base_url: http://localhost:11434
    dimension: 768
```

### Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `model` | string | `nomic-ai/nomic-embed-text-v2-moe` | No | HuggingFace model ID |
| `base_url` | string | `http://localhost:11434` | No | Ollama API endpoint |
| `dimension` | integer | 768 | No | Embedding dimension (128-4096) |
| `batch_size` | integer | 16 | No | Batch processing size (1-128) |
| `timeout` | integer | 60 | No | Request timeout in seconds |

---

## OllamaEmbedder

Native Ollama embedder with auto-pull support.

**Best for:** Ollama-based deployments, local inference

```yaml
- name: ollama_embeddings
  type: OllamaEmbedder
  config:
    model: nomic-embed-text:latest
    base_url: http://localhost:11434
    auto_pull: true
```

### Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `model` | string | `nomic-embed-text` | No | Ollama model name |
| `base_url` | string | `http://localhost:11434` | No | Ollama API endpoint |
| `dimension` | integer | 768 | No | Embedding dimension (128-4096) |
| `batch_size` | integer | 16 | No | Batch processing size (1-128) |
| `timeout` | integer | 60 | No | Request timeout in seconds |
| `auto_pull` | boolean | `true` | No | Auto-pull missing models |

### Recommended Models

```bash
# Pull embedding models
ollama pull nomic-embed-text          # 768 dim, general purpose
ollama pull mxbai-embed-large         # 1024 dim, higher quality
ollama pull all-minilm                # 384 dim, fast
```

---

## HuggingFaceEmbedder (Coming Soon)

:::note Coming Soon
HuggingFaceEmbedder is planned but not yet implemented. Use **UniversalEmbedder** or **OllamaEmbedder** instead.
:::

Direct HuggingFace model loading with GPU/MPS support.

**Best for:** Custom models, GPU acceleration, offline use

```yaml
- name: hf_embeddings
  type: HuggingFaceEmbedder
  config:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    device: auto
    normalize_embeddings: true
```

### Planned Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `model_name` | string | `sentence-transformers/all-MiniLM-L6-v2` | No | HuggingFace model ID |
| `device` | string | `auto` | No | `cpu`, `cuda`, `mps`, or `auto` |
| `batch_size` | integer | 32 | No | Batch size (1-256) |
| `normalize_embeddings` | boolean | `true` | No | L2 normalize embeddings |
| `show_progress_bar` | boolean | `false` | No | Show progress bar |
| `cache_folder` | string | `null` | No | Model cache directory |

### Recommended Models (for when available)

| Model | Dimensions | Speed | Quality |
|-------|------------|-------|---------|
| `all-MiniLM-L6-v2` | 384 | Fast | Good |
| `all-mpnet-base-v2` | 768 | Medium | High |
| `BAAI/bge-base-en-v1.5` | 768 | Medium | High |
| `BAAI/bge-large-en-v1.5` | 1024 | Slow | Highest |

---

## SentenceTransformerEmbedder (Coming Soon)

:::note Coming Soon
SentenceTransformerEmbedder is planned but not yet implemented. Use **UniversalEmbedder** or **OllamaEmbedder** instead.
:::

Sentence Transformers library integration.

**Best for:** Sentence-level embeddings, specialized models

```yaml
- name: st_embeddings
  type: SentenceTransformerEmbedder
  config:
    model_name: sentence-transformers/all-MiniLM-L6-v2
    device: cpu
```

### Planned Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `model_name` | string | `sentence-transformers/all-MiniLM-L6-v2` | No | Model name |
| `device` | string | `cpu` | No | `cpu`, `cuda`, or `mps` |

---

## OpenAIEmbedder (Coming Soon)

:::note Coming Soon
OpenAIEmbedder is planned but not yet implemented. Use **UniversalEmbedder** or **OllamaEmbedder** instead.
:::

OpenAI embedding API integration.

**Best for:** OpenAI API users, high-quality embeddings

```yaml
- name: openai_embeddings
  type: OpenAIEmbedder
  config:
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
```

### Options

| Option | Type | Default | Required | Description |
|--------|------|---------|----------|-------------|
| `model` | string | `text-embedding-3-small` | No | OpenAI embedding model |
| `api_key` | string | - | Yes | OpenAI API key |
| `base_url` | string | `null` | No | Custom API base URL |
| `organization` | string | `null` | No | OpenAI organization ID |
| `batch_size` | integer | 100 | No | Batch size (1-2048) |
| `max_retries` | integer | 3 | No | Maximum retry attempts |
| `timeout` | integer | 60 | No | Request timeout in seconds |

### Available Models

| Model | Dimensions | Max Tokens | Use Case |
|-------|------------|------------|----------|
| `text-embedding-3-small` | 1536 | 8191 | Cost-effective |
| `text-embedding-3-large` | 3072 | 8191 | Highest quality |
| `text-embedding-ada-002` | 1536 | 8191 | Legacy |

---

## Multiple Embedding Strategies

Configure multiple embedders for different content types:

```yaml
embedding_strategies:
  - name: general
    type: OllamaEmbedder
    config:
      model: nomic-embed-text
    priority: 0

  - name: code
    type: HuggingFaceEmbedder
    config:
      model_name: microsoft/codebert-base
    condition: "doc.type == 'code'"
    priority: 1

default_embedding_strategy: general
```

## Dimension Matching

Ensure embedding dimensions match your vector store configuration:

```yaml
databases:
  - name: my_db
    type: ChromaStore
    config:
      embedding_dimension: 768  # Must match embedder output

    embedding_strategies:
      - name: default
        type: OllamaEmbedder
        config:
          model: nomic-embed-text
          dimension: 768  # Match store dimension
```

## Next Steps

- [Extractors Reference](./extractors.md) - Add metadata extraction
- [Databases Reference](./databases.md) - Configure vector stores
- [RAG Guide](./index.md) - Full RAG overview
