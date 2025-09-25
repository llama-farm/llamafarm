# Strategy System Documentation

The RAG system uses a **new v1 schema** with modular databases and data processing strategies. This architecture provides maximum flexibility while maintaining consistency through DirectoryParser's always-active file routing.

## Table of Contents

1. [Overview](#overview)
2. [New v1 Schema Structure](#new-v1-schema-structure)
3. [Strategy Naming Convention](#strategy-naming-convention)
4. [DirectoryParser Architecture](#directoryparser-architecture)
5. [Available Components](#available-components)
6. [Retrieval Strategies](#retrieval-strategies)
7. [Creating Custom Strategies](#creating-custom-strategies)
8. [Best Practices](#best-practices)
9. [Examples](#examples)

## Overview

The v1 schema separates configuration into two main sections:
- **Databases**: Vector stores with embedding and retrieval strategies
- **Data Processing Strategies**: Document processing pipelines with parsers and extractors

### Key Changes in v1

- **DirectoryParser Always Active**: Automatic file detection and routing at strategy level
- **Unified Vector Store**: All strategies can share the same database
- **Parser Naming Convention**: `{ParserType}_{Implementation}` (e.g., `PDFParser_LlamaIndex`)
- **Strategy Naming**: `{data_processing_strategy}_{database_name}` for clear organization

### Benefits

- **No Code Changes**: Modify behavior through configuration
- **Automatic File Routing**: DirectoryParser handles file detection
- **Shared Storage**: Multiple strategies can use the same vector database
- **Clear Naming**: Consistent naming conventions for all components
- **Version Control**: Track configuration changes

## New v1 Schema Structure

The v1 schema uses a clean, organized structure with two main sections under the `rag` key:

```yaml
version: v1
name: project-name
namespace: default

rag:
  databases:
    - name: "main_database"
      type: "ChromaStore" 
      config:
        distance_function: "cosine"
        collection_name: "documents"
      default_embedding_strategy: "default_embeddings"
      default_retrieval_strategy: "basic_search"
      embedding_strategies:
        - name: "default_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            dimension: 768
      retrieval_strategies:
        - name: "basic_search"
          type: "BasicSimilarityStrategy"
          config:
            top_k: 10
  
  data_processing_strategies:
    - name: "pdf_processing"
      description: "Standard PDF document processing"
      # DirectoryParser configuration (ALWAYS ACTIVE)
      directory_config:
        recursive: true
        supported_files: ["*.pdf", "*.PDF"]
      parsers:
        - type: "PDFParser_LlamaIndex"
          mime_types: ["application/pdf"]
          file_extensions: [".pdf"]
          config:
            chunk_size: 1000
            chunk_overlap: 200
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG", "DATE"]
```

## Strategy Naming Convention

When using strategies with the CLI, combine the processing strategy and database names:

```
{data_processing_strategy}_{database_name}
```

Examples:
- `pdf_processing_main_database`
- `text_processing_main_database`
- `csv_processing_research_database`

## DirectoryParser Architecture

DirectoryParser is **ALWAYS ACTIVE** at the strategy level and handles:
- File/directory scanning based on `directory_config`
- MIME type detection
- File filtering based on patterns and extensions
- Routing files to appropriate parsers

The `directory_config` section controls how files are discovered and filtered:

```yaml
directory_config:
  recursive: true                           # Scan subdirectories
  supported_files: ["*.pdf", "*.txt"]       # Glob patterns for accepted files
  exclude_patterns: ["*.tmp", ".*"]         # Patterns to exclude  
  max_files: 1000                          # Maximum files to process
  follow_symlinks: false                   # Whether to follow symbolic links
```

## Available Components

### Parsers

All parsers follow the naming convention: `{ParserType}_{Implementation}`

#### PDF Parsers

**PDFParser_LlamaIndex**
```yaml
- type: "PDFParser_LlamaIndex"
  mime_types: ["application/pdf"]
  file_extensions: [".pdf", ".PDF"]
  config:
    chunk_size: 1500
    chunk_overlap: 200
    chunk_strategy: "semantic"  # semantic, pages, fixed
    preserve_equations: true
    extract_images: false
    extract_metadata: true
```

**PDFParser_PyPDF2**
```yaml
- type: "PDFParser_PyPDF2"
  mime_types: ["application/pdf"]
  file_extensions: [".pdf"]
  config:
    chunk_size: 1000
    chunk_overlap: 150
    chunk_strategy: "paragraphs"
    extract_metadata: true
```

#### Text Parsers

**TextParser_LlamaIndex**
```yaml
- type: "TextParser_LlamaIndex"
  mime_types: ["text/plain"]
  file_extensions: [".txt", ".text", ".log"]
  config:
    chunk_size: 1000
    chunk_overlap: 150
    extract_metadata: true
```

**TextParser_Python**
```yaml
- type: "TextParser_Python"
  mime_types: ["text/plain"]
  file_extensions: [".txt", ".log"]
  config:
    encoding: "utf-8"
    chunk_size: 1200
    chunk_overlap: 200
    chunk_strategy: "sentences"
```

#### CSV Parsers

**CSVParser_Pandas**
```yaml
- type: "CSVParser_Pandas"
  mime_types: ["text/csv"]
  file_extensions: [".csv", ".CSV"]
  config:
    content_fields: ["description", "content"]
    metadata_fields: ["id", "date", "category"]
    chunk_size: 500
```

#### Document Parsers

**DocxParser_LlamaIndex**
```yaml
- type: "DocxParser_LlamaIndex"
  mime_types: ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
  file_extensions: [".docx", ".DOCX"]
  config:
    chunk_size: 1500
    chunk_overlap: 200
    extract_tables: true
```

#### Markdown Parsers

**MarkdownParser_LlamaIndex**
```yaml
- type: "MarkdownParser_LlamaIndex"
  mime_types: ["text/markdown", "text/x-markdown"]
  file_extensions: [".md", ".markdown"]
  config:
    chunk_size: 1200
    chunk_overlap: 150
    extract_metadata: true
```

### Extractors

Available extractors for metadata enrichment:

#### EntityExtractor
```yaml
- type: "EntityExtractor"
  config:
    entities: ["PERSON", "ORG", "GPE", "DATE", "MONEY"]
    model: "en_core_web_sm"
    confidence_threshold: 0.7
```

#### KeywordExtractor
```yaml
- type: "KeywordExtractor"
  config:
    max_keywords: 10
    algorithm: "yake"  # or "rake", "tfidf"
    language: "en"
```

#### SummaryExtractor
```yaml
- type: "SummaryExtractor"
  config:
    max_sentences: 3
    include_keywords: true
    include_statistics: true
```

#### HeadingExtractor
```yaml
- type: "HeadingExtractor"
  config:
    levels: [1, 2, 3, 4]
    include_content: true
    max_content_length: 200
```

#### PatternExtractor
```yaml
- type: "PatternExtractor"
  config:
    patterns:
      email: "\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"
      phone: "\\d{3}-\\d{3}-\\d{4}"
      custom: "your-regex-here"
    include_context: true
    context_window: 50
```

#### LinkExtractor
```yaml
- type: "LinkExtractor"
  config:
    validate_urls: false
    extract_anchor_text: true
    categorize: true  # internal vs external
```

#### ContentStatisticsExtractor
```yaml
- type: "ContentStatisticsExtractor"
  config:
    calculate_readability: true
    calculate_sentiment: false
    word_frequency: true
```

### Embedders

#### OllamaEmbedder
```yaml
embedder:
  type: "OllamaEmbedder"
  config:
    model: "nomic-embed-text"
    host: "http://localhost:11434"
    dimension: 768
    batch_size: 32
    timeout: 60
```

#### OpenAIEmbedder (future)
```yaml
embedder:
  type: "OpenAIEmbedder"
  config:
    model: "text-embedding-3-small"
    api_key: "${OPENAI_API_KEY}"
    dimension: 1536
    batch_size: 100
```

### Vector Stores

#### ChromaStore
```yaml
vector_store:
  type: "ChromaStore"
  config:
    collection_name: "my_collection"
    distance_metric: "cosine"  # or "l2", "ip"
```

#### QdrantStore (future)
```yaml
vector_store:
  type: "QdrantStore"
  config:
    collection_name: "my_collection"
    host: "localhost"
    port: 6333
    vector_size: 768
    distance: "Cosine"
```

## Retrieval Strategies

### BasicSimilarityStrategy
Simple vector similarity search.

```yaml
retrieval_strategy:
  type: "BasicSimilarityStrategy"
  config:
    top_k: 5
    distance_metric: "cosine"
    score_threshold: 0.7
```

### MetadataFilteredStrategy
Filter results based on metadata before/after retrieval.

```yaml
retrieval_strategy:
  type: "MetadataFilteredStrategy"
  config:
    top_k: 10
    filters:
      priority: ["high", "critical"]
      date_after: "2024-01-01"
      category: "technical"
    filter_mode: "pre"  # or "post"
```

### RerankedStrategy
Re-rank results using multiple factors.

```yaml
retrieval_strategy:
  type: "RerankedStrategy"
  config:
    initial_k: 20
    final_k: 5
    rerank_factors:
      recency_weight: 0.2
      length_weight: 0.1
      metadata_boost:
        priority:
          high: 1.5
          medium: 1.0
          low: 0.8
```

### MultiQueryStrategy
Generate multiple query variations for better recall.

```yaml
retrieval_strategy:
  type: "MultiQueryStrategy"
  config:
    num_queries: 3
    aggregation: "rrf"  # reciprocal rank fusion
    top_k_per_query: 5
    final_k: 5
```

### HybridUniversalStrategy
Combine multiple strategies with weighted scoring.

```yaml
retrieval_strategy:
  type: "HybridUniversalStrategy"
  config:
    strategies:
      - type: "BasicSimilarityStrategy"
        weight: 0.6
        config:
          top_k: 10
      - type: "MetadataFilteredStrategy"
        weight: 0.4
        config:
          top_k: 10
    fusion_method: "weighted"  # or "rrf"
    final_k: 5
```

## Creating Custom Strategies

### Step 1: Define Your Requirements

Consider:
- Document types and formats
- Important metadata to extract
- Search patterns and use cases
- Performance requirements
- Storage constraints

### Step 2: Create Strategy YAML

Create a new file `my_strategies.yaml` following the v1 schema:

```yaml
version: v1
name: my-project
namespace: custom

rag:
  databases:
    - name: "custom_db"
      type: "ChromaStore"
      config:
        collection_name: "custom_documents"
      default_embedding_strategy: "custom_embeddings"
      default_retrieval_strategy: "custom_search"
      embedding_strategies:
        - name: "custom_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
      retrieval_strategies:
        - name: "custom_search"
          type: "BasicSimilarityStrategy"
          config:
            top_k: 10
  
  data_processing_strategies:
    - name: "my_custom_processing"
      description: "Custom processing for my use case"
      directory_config:
        recursive: true
        supported_files: ["*.pdf", "*.txt"]  # Accept PDF and text files
      parsers:
        - type: "PDFParser_LlamaIndex"
          mime_types: ["application/pdf"]
          file_extensions: [".pdf"]
          config:
            chunk_size: 1500
        - type: "TextParser_Python"
          mime_types: ["text/plain"]
          file_extensions: [".txt"]
          config:
            chunk_size: 1200
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG"]
```

### Step 3: Use with CLI

```bash
# Use your custom strategy file with the combined strategy name
python cli.py --strategy-file my_strategies.yaml \
    ingest documents/ \
    --strategy my_custom_processing_custom_db

# Search using your strategy
python cli.py --strategy-file my_strategies.yaml \
    search "query" \
    --strategy my_custom_processing_custom_db
```

### Step 4: Test and Iterate

```bash
# View collection info
python cli.py --strategy-file my_strategies.yaml \
    info --strategy my_custom_processing_custom_db

# List available strategies
python cli.py --strategy-file my_strategies.yaml \
    strategies list
```

## Best Practices

### 1. Start Simple
Begin with a basic configuration and add complexity as needed:

```yaml
version: v1
name: simple-project

rag:
  databases:
    - name: "simple_db"
      type: "ChromaStore"
      config: {}
      embedding_strategies:
        - name: "simple_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
      retrieval_strategies:
        - name: "simple_search"
          type: "BasicSimilarityStrategy"
          config:
            top_k: 5
  
  data_processing_strategies:
    - name: "simple_processing"
      directory_config:
        recursive: true
        include_patterns: ["*.txt"]
      parsers:
        - type: "TextParser_Python"
          mime_types: ["text/plain"]
          file_extensions: [".txt"]
```

### 2. Use Appropriate Chunk Sizes

| Document Type | Recommended Chunk Size | Overlap |
|--------------|------------------------|---------|
| Technical Docs | 512-1024 | 100-200 |
| Legal Text | 256-512 | 50-100 |
| Research Papers | 1024-2048 | 200-400 |
| Chat/Support | 256-512 | 50 |
| Code | Function/Class level | 0 |

### 3. Choose Extractors Wisely

Match extractors to your content:
- **Academic**: CitationExtractor, EntityExtractor
- **Business**: TableExtractor, ContentStatisticsExtractor
- **Technical**: HeadingExtractor, CodeExtractor
- **Support**: PatternExtractor, SentimentExtractor

### 4. Optimize Retrieval

Balance precision and recall:
- **High Precision**: Use filters and reranking
- **High Recall**: Use multi-query and larger top_k
- **Balanced**: Use hybrid strategies

### 5. Document Your Strategies

Add clear descriptions to track changes and purpose:

```yaml
data_processing_strategies:
  - name: "production_v2"
    description: "Version 2.0 - Added entity extraction, increased chunk overlap"
    # ... rest of configuration
```

## Examples

### Example 1: Legal Document Strategy

```yaml
version: v1
name: legal-system

rag:
  databases:
    - name: "legal_db"
      type: "ChromaStore"
      config:
        collection_name: "legal_documents"
      embedding_strategies:
        - name: "legal_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            dimension: 768
      retrieval_strategies:
        - name: "legal_search"
          type: "MetadataFilteredStrategy"
          config:
            top_k: 10
            filters:
              document_type: "contract"
              jurisdiction: "US"
  
  data_processing_strategies:
    - name: "legal_processing"
      description: "Optimized for legal contracts and agreements"
      directory_config:
        recursive: true
        supported_files: ["*.pdf", "*.docx"]  # Legal documents
      parsers:
        - type: "PDFParser_LlamaIndex"
          mime_types: ["application/pdf"]
          file_extensions: [".pdf"]
          config:
            chunk_size: 256
            chunk_overlap: 50
            extract_metadata: true
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["ORG", "PERSON", "DATE", "MONEY"]
        - type: "PatternExtractor"
          config:
            patterns:
              case_number: "[0-9]{2}-[A-Z]{2}-[0-9]{4}"
              statute: "\\d{1,3}\\s+U\\.S\\.C\\.\\s+§\\s+\\d+"
```

### Example 2: Research Papers Strategy

```yaml
version: v1
name: research-system

rag:
  databases:
    - name: "research_db"
      type: "ChromaStore"
      config:
        collection_name: "papers"
      embedding_strategies:
        - name: "research_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            dimension: 768
            batch_size: 32
      retrieval_strategies:
        - name: "research_search"
          type: "RerankedStrategy"
          config:
            initial_k: 20
            final_k: 5
  
  data_processing_strategies:
    - name: "research_processing"
      description: "Processing for academic papers with citations"
      directory_config:
        recursive: true
        supported_files: ["*.pdf"]  # Research papers
      parsers:
        - type: "PDFParser_LlamaIndex"
          mime_types: ["application/pdf"]
          file_extensions: [".pdf"]
          config:
            chunk_size: 1500
            chunk_overlap: 200
            chunk_strategy: "semantic"
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG", "DATE"]
        - type: "KeywordExtractor"
          config:
            algorithm: "yake"
            max_keywords: 15
        - type: "ContentStatisticsExtractor"
          config:
            include_readability: true
```

### Example 3: Mixed Document Repository

```yaml
version: v1
name: document-hub

rag:
  databases:
    - name: "shared_db"
      type: "ChromaStore"
      config: {}
      embedding_strategies:
        - name: "standard_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
      retrieval_strategies:
        - name: "standard_search"
          type: "BasicSimilarityStrategy"
          config:
            top_k: 10
  
  data_processing_strategies:
    - name: "mixed_documents"
      description: "Handles multiple document types"
      directory_config:
        recursive: true
        supported_files: ["*"]  # Accept all file types
        exclude_patterns: ["*.tmp", ".*"]
      parsers:
        - type: "PDFParser_LlamaIndex"
          mime_types: ["application/pdf"]
          file_extensions: [".pdf"]
          config:
            chunk_size: 1000
        - type: "TextParser_Python"
          mime_types: ["text/plain"]
          file_extensions: [".txt", ".log"]
          config:
            chunk_size: 1200
        - type: "MarkdownParser_LlamaIndex"
          mime_types: ["text/markdown"]
          file_extensions: [".md"]
          config:
            chunk_size: 1000
        - type: "CSVParser_Pandas"
          mime_types: ["text/csv"]
          file_extensions: [".csv"]
          config:
            chunk_size: 500
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG", "GPE", "DATE"]
```

## Troubleshooting

### Strategy Not Loading
- Check YAML syntax
- Verify all component types exist
- Ensure configuration keys are correct

### Poor Search Results
- Adjust chunk size and overlap
- Add relevant extractors
- Try different retrieval strategies
- Increase top_k values

### Slow Performance
- Reduce chunk size
- Use smaller embedding models
- Enable caching
- Optimize batch sizes

### High Memory Usage
- Use disk-based vector stores
- Reduce embedding dimensions
- Enable compression/quantization
- Process in smaller batches

## Next Steps

- [Component Guide](COMPONENTS.md) - Detailed component documentation
- [CLI Guide](../cli/README.md) - Using strategies with CLI
- [API Reference](API_REFERENCE.md) - Programmatic strategy usage
- [Examples](../demos/) - See strategies in action