# RAG Schema v1 Documentation

Complete reference for the RAG system schema format. This document describes the structure and configuration options available in the v1 schema.

## 📋 Schema Overview

The RAG schema is defined in `schema.yaml` and structures configurations into two main sections:

1. **`databases`** - Vector database configurations with embedding and retrieval strategies
2. **`data_processing_strategies`** - Document processing pipelines with parsers and extractors

## 🗂️ Top-Level Structure

```yaml
# Required schema version
version: v1
name: project-name
namespace: default

# Optional runtime configuration
runtime:
  provider: ollama
  model: llama3.2
  base_url: http://localhost:11434
  temperature: 0.5

# Optional prompt configurations
prompts: []

# Optional dataset configurations  
datasets: []

# Main RAG configuration
rag:
  databases: [...]
  data_processing_strategies: [...]
```

## 🗄️ Databases Configuration

Each database defines a vector store with its own embedding and retrieval strategies.

### Database Definition

```yaml
databases:
  - name: "main_database"  # Unique identifier (lowercase, underscores)
    type: "ChromaStore"     # ChromaStore or QdrantStore
    config:
      distance_function: "cosine"  # cosine, l2, ip
      collection_name: "documents"
      port: 8000  # Optional, for client-server mode
    
    # Default strategies for this database
    default_embedding_strategy: "default_embeddings"
    default_retrieval_strategy: "basic_search"
    
    # Available embedding strategies
    embedding_strategies:
      - name: "default_embeddings"
        type: "OllamaEmbedder"
        config:
          model: "nomic-embed-text"
          base_url: "http://localhost:11434/"
          dimension: 768
          batch_size: 16
          timeout: 60
          auto_pull: true
    
    # Available retrieval strategies  
    retrieval_strategies:
      - name: "basic_search"
        type: "BasicSimilarityStrategy"
        config:
          distance_metric: "cosine"
          top_k: 10
        default: true
```

### Supported Database Types

| Type | Description | Key Config Options |
|------|-------------|-------------------|
| `ChromaStore` | ChromaDB vector database | `collection_name`, `distance_function` |
| `QdrantStore` | Qdrant vector database | `host`, `port`, `collection_name`, `vector_size` |

### Embedding Strategy Types

| Type | Description | Key Config Options |
|------|-------------|-------------------|
| `OllamaEmbedder` | Local Ollama embeddings | `model`, `base_url`, `dimension`, `batch_size` |
| `OpenAIEmbedder` | OpenAI embeddings | `model`, `api_key`, `dimension` |
| `HuggingFaceEmbedder` | HuggingFace embeddings | `model_name`, `device`, `batch_size` |

### Retrieval Strategy Types

| Type | Description | Key Config Options |
|------|-------------|-------------------|
| `BasicSimilarityStrategy` | Simple similarity search | `distance_metric`, `top_k` |
| `MetadataFilteredStrategy` | Filtered search | `top_k`, `filter_mode`, `filters` |
| `HybridUniversalStrategy` | Combines multiple strategies | `strategies`, `combination_method`, `final_k` |
| `RerankedStrategy` | Reranks initial results | `initial_k`, `final_k`, `rerank_factors` |

## 📁 Data Processing Strategies

Each strategy defines how documents are processed, including file filtering, parsing, and metadata extraction.

### Strategy Definition

```yaml
data_processing_strategies:
  - name: "pdf_processing"  # Unique identifier (lowercase, underscores)
    description: "Standard PDF document processing"
    
    # DirectoryParser configuration (ALWAYS ACTIVE)
    directory_config:
      recursive: true  # Scan subdirectories
      supported_files: ["*.pdf", "*.PDF"]  # Glob patterns for accepted files
      exclude_patterns: ["*.tmp", ".*"]  # Patterns to exclude
      max_files: 1000  # Maximum files to process
      follow_symlinks: false  # Whether to follow symbolic links
    
    # Parser configurations
    parsers:
      - type: "PDFParser_LlamaIndex"
        mime_types: ["application/pdf"]  # MIME types this parser handles
        file_extensions: [".pdf", ".PDF"]  # Extensions this parser handles
        config:
          chunk_size: 1000
          chunk_overlap: 200
          chunk_strategy: "semantic"  # semantic, pages, fixed
          extract_metadata: true
          preserve_equations: true
          extract_images: false
    
    # Extractor configurations
    extractors:
      - type: "EntityExtractor"
        config:
          entity_types: ["PERSON", "ORG", "GPE", "DATE", "PRODUCT"]
          use_fallback: true
          min_entity_length: 2
      
      - type: "KeywordExtractor"
        config:
          algorithm: "yake"
          max_keywords: 10
          min_keyword_length: 3
```

### Directory Configuration Options

| Option | Type | Description | Default |
|--------|------|-------------|---------|
| `recursive` | boolean | Scan subdirectories | `true` |
| `supported_files` | array | Glob patterns for accepted files | `["*"]` |
| `exclude_patterns` | array | Glob patterns to exclude | `[".*", "__pycache__/*"]` |
| `max_files` | integer | Maximum files to process | `1000` |
| `follow_symlinks` | boolean | Follow symbolic links | `false` |

## 🔧 Available Parsers

All parsers follow the naming convention: `{ParserType}_{Implementation}`

### PDF Parsers

| Parser | Description | Key Config Options |
|--------|-------------|-------------------|
| `PDFParser_LlamaIndex` | LlamaIndex-based PDF parsing | `chunk_strategy`, `preserve_equations`, `extract_images` |
| `PDFParser_PyPDF2` | PyPDF2-based PDF parsing | `chunk_strategy`, `extract_metadata`, `combine_pages` |

### Text Parsers

| Parser | Description | Key Config Options |
|--------|-------------|-------------------|
| `TextParser_LlamaIndex` | LlamaIndex text parser | `chunk_size`, `chunk_overlap`, `extract_metadata` |
| `TextParser_Python` | Pure Python text parser | `encoding`, `chunk_strategy`, `clean_text` |

### CSV Parsers

| Parser | Description | Key Config Options |
|--------|-------------|-------------------|
| `CSVParser_LlamaIndex` | LlamaIndex CSV parser | `chunk_size`, `chunk_overlap` |
| `CSVParser_Pandas` | Pandas-based CSV parser | `content_fields`, `metadata_fields` |
| `CSVParser_Python` | Pure Python CSV parser | `delimiter`, `encoding` |

### Document Parsers

| Parser | Description | Key Config Options |
|--------|-------------|-------------------|
| `DocxParser_LlamaIndex` | LlamaIndex Word parser | `chunk_size`, `extract_tables` |
| `DocxParser_PythonDocx` | python-docx based parser | `extract_metadata`, `extract_comments` |

### Spreadsheet Parsers

| Parser | Description | Key Config Options |
|--------|-------------|-------------------|
| `ExcelParser_LlamaIndex` | LlamaIndex Excel parser | `chunk_size`, `extract_metadata` |
| `ExcelParser_Pandas` | Pandas Excel parser | `sheet_name`, `header_row` |
| `ExcelParser_OpenPyXL` | OpenPyXL Excel parser | `data_only`, `read_only` |

### Markdown Parsers

| Parser | Description | Key Config Options |
|--------|-------------|-------------------|
| `MarkdownParser_LlamaIndex` | LlamaIndex Markdown parser | `extract_headings`, `extract_code_blocks` |
| `MarkdownParser_Python` | Pure Python Markdown parser | `chunk_strategy`, `extract_links` |

## 🔍 Available Extractors

Extractors enrich document chunks with metadata.

| Extractor | Description | Key Config Options |
|-----------|-------------|-------------------|
| `EntityExtractor` | Named entity recognition | `entity_types`, `use_fallback`, `min_entity_length` |
| `KeywordExtractor` | Keyword extraction | `algorithm`, `max_keywords`, `min_keyword_length` |
| `ContentStatisticsExtractor` | Document statistics | `include_readability`, `include_vocabulary`, `include_structure` |
| `SummaryExtractor` | Text summarization | `summary_sentences`, `algorithm`, `include_key_phrases` |
| `HeadingExtractor` | Extract headings | `max_level`, `include_hierarchy`, `extract_outline` |
| `PatternExtractor` | Pattern matching | `predefined_patterns`, `custom_patterns`, `include_context` |
| `DateTimeExtractor` | Date/time extraction | `formats`, `extract_relative`, `fuzzy_parsing` |
| `TableExtractor` | Table extraction | `output_format`, `extract_headers`, `min_rows` |

## 🎯 Strategy Naming Convention

When using strategies with the CLI, combine the processing strategy and database names:

```
{data_processing_strategy}_{database_name}
```

Examples:
- `pdf_processing_main_database`
- `csv_processing_research_database`
- `multi_format_main_database`

## 📝 Complete Example

```yaml
version: v1
name: my-rag-project
namespace: production

rag:
  databases:
    - name: "main_database"
      type: "ChromaStore"
      config:
        distance_function: "cosine"
        collection_name: "documents"
      default_embedding_strategy: "default_embeddings"
      default_retrieval_strategy: "hybrid_search"
      embedding_strategies:
        - name: "default_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            base_url: "http://localhost:11434/"
            dimension: 768
            batch_size: 16
      retrieval_strategies:
        - name: "hybrid_search"
          type: "HybridUniversalStrategy"
          config:
            strategies:
              - type: "BasicSimilarityStrategy"
                weight: 0.7
                config:
                  top_k: 20
              - type: "MetadataFilteredStrategy"
                weight: 0.3
                config:
                  top_k: 20
            final_k: 10
  
  data_processing_strategies:
    - name: "multi_format"
      description: "Process multiple document types"
      directory_config:
        recursive: true
        supported_files: ["*"]  # Accept all files
        exclude_patterns: ["*.tmp", ".*", "__pycache__/*"]
        max_files: 5000
      parsers:
        - type: "PDFParser_LlamaIndex"
          mime_types: ["application/pdf"]
          file_extensions: [".pdf"]
          config:
            chunk_size: 1500
            chunk_overlap: 200
        - type: "TextParser_Python"
          mime_types: ["text/plain"]
          file_extensions: [".txt", ".log"]
          config:
            chunk_size: 1200
            chunk_overlap: 150
        - type: "CSVParser_Pandas"
          mime_types: ["text/csv"]
          file_extensions: [".csv"]
          config:
            chunk_size: 500
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG", "DATE", "PRODUCT"]
        - type: "KeywordExtractor"
          config:
            algorithm: "yake"
            max_keywords: 15
```

## 🔄 Migration from Old Format

If migrating from the old format:

1. Move vector store config to `databases` section
2. Move parser/extractor config to `data_processing_strategies`
3. Add `directory_config` for file filtering
4. Update parser names to new convention
5. Combine strategies using naming convention

## 📚 See Also

- [schema.yaml](../schema.yaml) - The actual schema definition
- [config/templates/default.yaml](../config/templates/default.yaml) - Example configuration
- [config/templates/advanced.yaml](../config/templates/advanced.yaml) - Advanced examples
- [Strategy System](STRATEGY_SYSTEM.md) - How strategies work
- [Parser Guide](PARSERS.md) - Detailed parser documentation