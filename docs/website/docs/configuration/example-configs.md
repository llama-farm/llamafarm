---
title: Example Configs
sidebar_label: Examples
sidebar_position: 1
---

# Example Configurations

Use these snippets as starting points for real projects. Every example validates against the current schema.

## Local RAG with Ollama (Multi-Model)

```yaml
version: v1
name: local-rag
namespace: default

runtime:
  default_model: default

  models:
    - name: default
      description: "Primary Ollama model"
      provider: ollama
      model: llama3:8b
      default: true

prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are a friendly assistant. Reference document titles when possible.

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
            chunk_size: 1200
            chunk_overlap: 150
      extractors:
        - type: HeadingExtractor
        - type: ContentStatisticsExtractor

datasets:
  - name: policies
    data_processing_strategy: pdf_ingest
    database: main_db
```

## Lemonade Local Runtime (Multi-Model)

```yaml
version: v1
name: lemonade-local
namespace: default

runtime:
  default_model: balanced

  models:
    - name: fast
      description: "Fast 0.6B model for quick responses"
      provider: lemonade
      model: user.Qwen3-0.6B
      base_url: "http://127.0.0.1:11534/v1"
      lemonade:
        backend: llamacpp
        port: 11534
        context_size: 32768

    - name: balanced
      description: "Balanced 4B model - recommended"
      provider: lemonade
      model: user.Qwen3-4B
      base_url: "http://127.0.0.1:11535/v1"
      default: true
      lemonade:
        backend: llamacpp
        port: 11535
        context_size: 32768

    - name: powerful
      description: "Powerful 8B reasoning model"
      provider: lemonade
      model: user.Qwen3-8B
      base_url: "http://127.0.0.1:11536/v1"
      lemonade:
        backend: llamacpp
        port: 11536
        context_size: 65536

prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are a helpful assistant with access to local models.
```

**Setup:**
1. Download models: `uv run lemonade-server-dev pull user.Qwen3-4B --checkpoint unsloth/Qwen3-4B-GGUF:Q4_K_M --recipe llamacpp`
2. Start instances (from llamafarm project root):
   - `LEMONADE_MODEL=user.Qwen3-0.6B LEMONADE_PORT=11534 nx start lemonade`
   - `LEMONADE_MODEL=user.Qwen3-4B LEMONADE_PORT=11535 nx start lemonade`
   - `LEMONADE_MODEL=user.Qwen3-8B LEMONADE_PORT=11536 nx start lemonade`

> **Note:** Currently, Lemonade must be manually started. In the future, it will run as a container and be auto-started by the LlamaFarm server.

See [Lemonade Quickstart](../models#quick-setup) for detailed setup.

## vLLM Gateway with Structured Output

```yaml
version: v1
name: llm-gateway
namespace: enterprise

runtime:
  default_model: vllm-model

  models:
    - name: vllm-model
      description: "vLLM gateway model"
      provider: openai
      model: qwen2.5:7b
      base_url: https://llm.company.internal/v1
      api_key: ${VLLM_API_KEY}
      instructor_mode: tools
      default: true
      model_api_parameters:
        temperature: 0.1

prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are a compliance assistant returning JSON with fields: `summary`, `citations`.

rag:
  databases:
    - name: compliance_db
      type: QdrantStore
      default_embedding_strategy: openai_embeddings
      default_retrieval_strategy: hybrid_search
      embedding_strategies:
        - name: openai_embeddings
          type: OpenAIEmbedder
          config:
            model: text-embedding-3-small
      retrieval_strategies:
        - name: hybrid_search
          type: HybridUniversalStrategy
          config:
            dense_weight: 0.7
            sparse_weight: 0.3
  data_processing_strategies:
    - name: docx_ingest
      parsers:
        - type: DocxParser_LlamaIndex
          config:
            chunk_size: 1000
            chunk_overlap: 100
      extractors:
        - type: EntityExtractor
          config:
            include_types: [ORGANIZATION, LAW]
```

## Multi-Strategy Retrieval

```yaml
rag:
  databases:
    - name: research_db
      type: ChromaStore
      default_embedding_strategy: dense_embeddings
      default_retrieval_strategy: reranked_search
      embedding_strategies:
        - name: dense_embeddings
          type: SentenceTransformerEmbedder
          config:
            model: all-MiniLM-L6-v2
      retrieval_strategies:
        - name: keyword_search
          type: BM25Retriever
          config:
            stop_words: ["the", "a", "and"]
        - name: reranked_search
          type: RerankedStrategy
          config:
            candidate_strategy: keyword_search
            reranker: bm25+embedding
```

## Mixed Providers (Ollama + Lemonade)

```yaml
version: v1
name: mixed-providers
namespace: default

runtime:
  default_model: ollama-default

  models:
    - name: ollama-default
      description: "Primary Ollama model"
      provider: ollama
      model: llama3:8b
      default: true

    - name: ollama-small
      description: "Small Ollama model"
      provider: ollama
      model: gemma3:1b

    - name: lemon-fast
      description: "Lemonade fast model with NPU/GPU"
      provider: lemonade
      model: user.Qwen3-0.6B
      base_url: "http://127.0.0.1:11534/v1"
      lemonade:
        backend: llamacpp
        port: 11534
        context_size: 32768

prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are a helpful assistant. Use the appropriate model for the task.
```

**Usage:**
- Fast responses: `lf chat --model lemon-fast "Quick question"`
- Default model: `lf chat "Normal question"`
- Small model: `lf chat --model ollama-small "Simple task"`

---

Mix and match these patterns to suit your project. Remember to regenerate schema types if you add new providers or store options.
