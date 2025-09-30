---
title: Example Configs
sidebar_label: Examples
sidebar_position: 1
---

# Example Configurations

Use these snippets as starting points for real projects. Every example validates against the current schema.

## Local RAG with Ollama

```yaml
version: v1
name: local-rag
namespace: default

runtime:
  provider: ollama
  model: llama3:8b
  instructor_mode: null

prompts:
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

## vLLM Gateway with Structured Output

```yaml
version: v1
name: llm-gateway
namespace: enterprise

runtime:
  provider: openai
  model: qwen2.5:7b
  base_url: https://llm.company.internal/v1
  api_key: ${VLLM_API_KEY}
  instructor_mode: tools
  model_api_parameters:
    temperature: 0.1

prompts:
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

Mix and match these patterns to suit your project. Remember to regenerate schema types if you add new providers or store options.
