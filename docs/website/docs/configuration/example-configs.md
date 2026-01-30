---
title: Example Configs
sidebar_label: Examples
sidebar_position: 1
---

# Example Configurations

Use these snippets as starting points for real projects. Every example validates against the current schema.

## Quick Start (Minimal Config)

The simplest working configuration:

```yaml
version: v1
name: quickstart
namespace: default

runtime:
  default_model: default
  models:
    - name: default
      provider: ollama
      model: gemma3:1b
      default: true
```

## Simple RAG Setup

Basic RAG with PDF processing:

```yaml
version: v1
name: simple-rag
namespace: default

runtime:
  default_model: default
  models:
    - name: default
      provider: ollama
      model: llama3.1:8b
      default: true

rag:
  databases:
    - name: docs_db
      type: ChromaStore
      embedding_strategies:
        - name: default
          type: OllamaEmbedder
          config:
            model: nomic-embed-text
      retrieval_strategies:
        - name: search
          type: BasicSimilarityStrategy
          config:
            top_k: 5
          default: true
      default_embedding_strategy: default
      default_retrieval_strategy: search

  data_processing_strategies:
    - name: basic
      description: "Basic PDF and text processing"
      parsers:
        - type: PDFParser_PyPDF2
          file_include_patterns: ["*.pdf"]
          config:
            chunk_size: 1000
            chunk_overlap: 100
        - type: TextParser_Python
          file_include_patterns: ["*.txt", "*.md"]
          config:
            chunk_size: 800
```

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

## Universal Runtime with Reranking

Universal Runtime enables cross-encoder reranking for high-precision search:

```yaml
version: v1
name: reranking-demo
namespace: default

runtime:
  default_model: default
  models:
    - name: default
      provider: ollama
      model: llama3.1:8b
      default: true

    # Cross-encoder reranker via Universal Runtime
    - name: reranker
      description: "Fast cross-encoder for document reranking"
      provider: universal
      model: cross-encoder/ms-marco-MiniLM-L-6-v2
      base_url: http://127.0.0.1:11540

rag:
  databases:
    - name: main_db
      type: ChromaStore
      config:
        collection_name: documents
        distance_function: cosine

      embedding_strategies:
        - name: default
          type: OllamaEmbedder
          config:
            model: nomic-embed-text
            dimension: 768

      retrieval_strategies:
        - name: fast
          type: BasicSimilarityStrategy
          config:
            top_k: 10
          default: false

        - name: accurate
          type: CrossEncoderRerankedStrategy
          config:
            model_name: reranker
            initial_k: 30
            final_k: 5
            relevance_threshold: 0.3
          default: true

      default_embedding_strategy: default
      default_retrieval_strategy: accurate
```

## Complex Multi-Format RAG

Full-featured RAG with multiple file types, extractors, and retrieval strategies:

```yaml
version: v1
name: enterprise-rag
namespace: production

runtime:
  default_model: primary
  models:
    - name: primary
      description: "Primary production model"
      provider: ollama
      model: llama3.1:8b
      default: true

    - name: fast
      description: "Fast model for quick queries"
      provider: ollama
      model: gemma3:1b

    - name: query_decomposer
      description: "Model for query decomposition"
      provider: openai
      model: gemma3:1b
      base_url: http://localhost:11434/v1

    - name: reranker
      description: "Cross-encoder reranker"
      provider: universal
      model: BAAI/bge-reranker-v2-m3
      base_url: http://127.0.0.1:11540

prompts:
  - name: default
    messages:
      - role: system
        content: |
          You are an expert document analyst. Always cite sources by filename.
          If information is not in the provided context, say so clearly.

rag:
  default_database: main_db

  databases:
    - name: main_db
      type: ChromaStore
      config:
        collection_name: enterprise_docs
        distance_function: cosine
        persist_directory: ./data/main_db

      embedding_strategies:
        - name: default
          type: UniversalEmbedder
          config:
            model: nomic-ai/nomic-embed-text-v2-moe
            dimension: 768
            batch_size: 16
          priority: 0

      retrieval_strategies:
        # Fast basic search
        - name: fast
          type: BasicSimilarityStrategy
          config:
            top_k: 10
            distance_metric: cosine

        # Filtered by metadata
        - name: filtered
          type: MetadataFilteredStrategy
          config:
            top_k: 10
            filter_mode: pre

        # High-accuracy reranking
        - name: accurate
          type: CrossEncoderRerankedStrategy
          config:
            model_name: reranker
            initial_k: 30
            final_k: 10
          default: true

        # Complex query handling
        - name: complex
          type: MultiTurnRAGStrategy
          config:
            model_name: query_decomposer
            max_sub_queries: 3
            complexity_threshold: 50
            enable_reranking: true
            reranker_config:
              model_name: reranker
              initial_k: 20
              final_k: 10

      default_embedding_strategy: default
      default_retrieval_strategy: accurate

  data_processing_strategies:
    - name: universal_processor
      description: "Process PDFs, Word docs, spreadsheets, and text"
      parsers:
        # PDFs with LlamaIndex (semantic chunking)
        - type: PDFParser_LlamaIndex
          file_include_patterns: ["*.pdf", "*.PDF"]
          priority: 100
          config:
            chunk_size: 1200
            chunk_overlap: 150
            chunk_strategy: semantic
            extract_metadata: true
            extract_tables: true

        # PDF fallback
        - type: PDFParser_PyPDF2
          file_include_patterns: ["*.pdf"]
          priority: 50
          config:
            chunk_size: 1000
            chunk_overlap: 100

        # Word documents
        - type: DocxParser_LlamaIndex
          file_include_patterns: ["*.docx"]
          priority: 100
          config:
            chunk_size: 1000
            chunk_overlap: 100
            extract_tables: true

        # Excel files
        - type: ExcelParser_LlamaIndex
          file_include_patterns: ["*.xlsx", "*.xls"]
          priority: 100
          config:
            chunk_size: 500
            chunk_strategy: rows

        # CSV files
        - type: CSVParser_Pandas
          file_include_patterns: ["*.csv"]
          priority: 100
          config:
            chunk_size: 500
            extract_metadata: true

        # Markdown
        - type: MarkdownParser_LlamaIndex
          file_include_patterns: ["*.md", "*.markdown"]
          priority: 100
          config:
            chunk_size: 800
            chunk_strategy: headings
            extract_code_blocks: true

        # Plain text and code
        - type: TextParser_LlamaIndex
          file_include_patterns: ["*.txt", "*.py", "*.js", "*.html"]
          priority: 80
          config:
            chunk_size: 800
            chunk_strategy: semantic
            preserve_code_structure: true

      extractors:
        # Entity extraction
        - type: EntityExtractor
          priority: 100
          config:
            entity_types: [PERSON, ORG, DATE, PRODUCT, EMAIL, PHONE]
            use_fallback: true

        # Keyword extraction
        - type: KeywordExtractor
          priority: 90
          config:
            algorithm: yake
            max_keywords: 15

        # Content statistics
        - type: ContentStatisticsExtractor
          priority: 80
          config:
            include_readability: true
            include_structure: true

        # Pattern matching
        - type: PatternExtractor
          priority: 70
          file_include_patterns: ["*.pdf"]
          config:
            predefined_patterns: [email, phone, date, version]

datasets:
  - name: documents
    database: main_db
    data_processing_strategy: universal_processor
```

## Dynamic Variables in Prompts and Tools

Use Jinja2-style `{{variable | default}}` syntax to customize prompts and tools at request time:

```yaml
version: v1
name: dynamic-demo
namespace: default

runtime:
  default_model: assistant
  models:
    - name: assistant
      provider: universal
      model: llama3.2:3b
      default: true
      tool_call_strategy: native_api
      prompts:
        - system
        - context
      tools:
        - type: function
          name: search_knowledge_base
          description: "Search the {{company_name | Company}} knowledge base"
          parameters:
            type: object
            properties:
              query:
                type: string
                description: "Search query for {{department | General}} topics"
            required:
              - query
        - type: function
          name: create_ticket
          description: "Create a support ticket for {{user_name | a customer}}"
          parameters:
            type: object
            properties:
              title:
                type: string
              priority:
                type: string
                enum: ["low", "medium", "high"]
            required:
              - title

prompts:
  - name: system
    messages:
      - role: system
        content: |
          You are a helpful assistant for {{company_name | Acme Corp}}.
          You work in the {{department | General}} department.
          Current date: {{current_date | today}}

  - name: context
    messages:
      - role: system
        content: |
          ## Customer Information
          - Name: {{user_name | Valued Customer}}
          - Account Tier: {{account_tier | standard}}
          - Language: {{language | English}}

          Adjust responses based on tier:
          - basic: Focus on self-service
          - standard: Provide helpful guidance
          - premium: Offer personalized assistance
```

**Usage with full variables:**

```bash
curl -X POST http://localhost:14345/v1/projects/default/dynamic-demo/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hi, what can you help me with?"}],
    "variables": {
      "company_name": "TechCorp Solutions",
      "department": "Technical Support",
      "user_name": "Alice Johnson",
      "account_tier": "premium",
      "current_date": "2024-01-15"
    }
  }'
```

**Usage with defaults (minimal variables):**

```bash
curl -X POST http://localhost:14345/v1/projects/default/dynamic-demo/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "variables": {
      "user_name": "Bob"
    }
  }'
```

Unprovided variables use their defaults (e.g., `company_name` becomes "Acme Corp").

**Variable syntax:**

| Pattern | Description |
|---------|-------------|
| `{{variable}}` | Required - error if not provided |
| `{{variable \| default}}` | Uses default if not provided |
| `{{ variable }}` | Whitespace is allowed |

## Qdrant Production Setup

Production-ready configuration with Qdrant vector database:

```yaml
version: v1
name: qdrant-production
namespace: production

runtime:
  default_model: default
  models:
    - name: default
      provider: openai
      model: gpt-4o-mini
      api_key: ${OPENAI_API_KEY}
      default: true

rag:
  databases:
    - name: production_db
      type: QdrantStore
      config:
        host: qdrant.internal.company.com
        port: 6333
        api_key: ${QDRANT_API_KEY}
        collection_name: documents
        vector_size: 1536
        distance: Cosine

      embedding_strategies:
        - name: openai
          type: OpenAIEmbedder
          config:
            model: text-embedding-3-small
            api_key: ${OPENAI_API_KEY}
            batch_size: 100

      retrieval_strategies:
        - name: production
          type: HybridUniversalStrategy
          config:
            combination_method: weighted_average
            final_k: 10
            strategies:
              - type: BasicSimilarityStrategy
                weight: 0.7
                config:
                  top_k: 20
              - type: MetadataFilteredStrategy
                weight: 0.3
                config:
                  top_k: 20
          default: true

      default_embedding_strategy: openai
      default_retrieval_strategy: production
```

---

Mix and match these patterns to suit your project. For detailed component configuration, see the [RAG documentation](../rag/index.md):

- [Parsers Reference](../rag/parsers.md) - All parser types and options
- [Embedders Reference](../rag/embedders.md) - Embedding configurations
- [Extractors Reference](../rag/extractors.md) - Metadata extraction
- [Databases Reference](../rag/databases.md) - Vector store options
- [Retrieval Strategies](../rag/retrieval-strategies.md) - Search configurations
