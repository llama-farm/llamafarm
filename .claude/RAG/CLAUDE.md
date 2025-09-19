# LlamaFarm RAG System - Complete Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [NEW Schema System](#new-schema-system)
4. [Configuration Examples](#configuration-examples)
5. [Parser System](#parser-system)
6. [Directory Structure](#directory-structure)
7. [Running Demos](#running-demos)
8. [CLI Commands](#cli-commands)
9. [Migration Notes](#migration-notes)

ALWAYS USE UV for python!

---

## Overview

The LlamaFarm RAG (Retrieval-Augmented Generation) system has been completely migrated to a new schema format with NO backward compatibility. The system now uses a clean separation between databases and data processing strategies.

### Key Changes (as of 2025-09-09)
- **NO Legacy Support**: All `_convert_rag_to_legacy` code removed
- **DirectoryParser Always Active**: Handles both single files and directories at strategy level
- **Clean Naming**: `handler.py` with `SchemaHandler` class (was `DirectSchemaHandler`)
- **Deprecated Files**: `config.py.deprecated` and `manager.py.deprecated`
- **LlamaIndex Parsers**: Used as default fallbacks instead of hacky simple parsers
- **Strategy Naming**: `{data_processing_strategy}_{database_name}`

---

## Architecture

### New Component Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    NEW RAG SCHEMA (YAML)                     │
│                         schema.yaml                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Strategy│
                    │ Loader  │ (loader.py - new schema only)
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
   │Database │    │Processing │   │Directory  │
   │ Configs │    │Strategies │   │  Config   │
   └────┬────┘    └─────┬─────┘   └─────┬─────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                   ┌─────▼─────┐
                   │  Schema    │ (handler.py)
                   │  Handler   │
                   └─────┬─────┘
                         │
                ┌────────▼────────┐
                │ DirectoryParser │ (ALWAYS ACTIVE)
                └────────┬────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────▼───┐ ┌───▼───┐ ┌───▼────┐
         │Parsers │ │Embedder│ │Stores  │
         └────────┘ └────────┘ └────────┘
```

### Key Components

1. **SchemaHandler** (`/core/strategies/handler.py`): Main handler for new schema
2. **StrategyLoader** (`/core/strategies/loader.py`): Loads only new RAG schema format
3. **DirectoryParser**: Always active, handles both files and directories
4. **NO Legacy Components**: No `StrategyManager`, no `StrategyConfig`

---

## NEW Schema System

### Schema Structure (NO BACKWARD COMPATIBILITY)

```yaml
# REQUIRED: Top-level 'rag' key
rag:
  # Databases define vector stores and their strategies
  databases:
    - name: "my_database"
      type: "ChromaStore"
      config:
        persist_directory: "./vectordb"
        distance_function: "cosine"
      # REQUIRED: Default strategies
      default_embedding_strategy: "my_embeddings"
      default_retrieval_strategy: "my_retrieval"
      embedding_strategies:
        - name: "my_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            batch_size: 5
          default: true
      retrieval_strategies:
        - name: "my_retrieval"
          type: "BasicSimilarityStrategy"
          config:
            top_k: 5
          default: true
  
  # Processing strategies define how documents are processed
  data_processing_strategies:
    - name: "my_processing"
      description: "Process text and PDF files"
      # DirectoryParser config (ALWAYS ACTIVE)
      directory_config:
        recursive: true
        include_patterns: ["*.txt", "*.pdf", "*.md"]
        exclude_patterns: ["*.tmp", ".*"]
        # File filtering happens ONLY here
        allowed_mime_types: []  # Empty = accept all
        allowed_extensions: []  # Empty = accept all
        max_files: 1000
      # Parsers to use for different file types
      parsers:
        - type: "TextParser_LlamaIndex"  # Or TextParser_Python, etc.
          config:
            chunk_size: 1500
            chunk_overlap: 200
      # Optional extractors
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG", "DATE"]
```

### Strategy Naming Convention

Strategies are named by combining processing strategy and database:
- Format: `{data_processing_strategy}_{database_name}`
- Example: `text_processing_research_papers_db`
- Example: `pdf_processing_main_chroma_db`

---

## Configuration Examples

### Example 1: Simple Text Processing

```yaml
rag:
  databases:
    - name: "simple_db"
      type: "ChromaStore"
      config:
        persist_directory: "./simple_vectordb"
      default_embedding_strategy: "basic"
      default_retrieval_strategy: "basic"
      embedding_strategies:
        - name: "basic"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
          default: true
      retrieval_strategies:
        - name: "basic"
          type: "BasicSimilarityStrategy"
          config:
            top_k: 3
          default: true
  
  data_processing_strategies:
    - name: "simple_text"
      description: "Basic text file processing"
      directory_config:
        recursive: false
        include_patterns: ["*.txt"]
      parsers:
        - type: "TextParser_LlamaIndex"
          config:
            chunk_size: 1000
```

Usage: `cli.py ingest --strategy simple_text_simple_db file.txt`

### Example 2: Multi-Parser Strategy for Research

```yaml
rag:
  databases:
    - name: "research_db"
      type: "ChromaStore"
      config:
        persist_directory: "./research_vectordb"
        collection_name: "papers"
      default_embedding_strategy: "academic"
      default_retrieval_strategy: "filtered"
      embedding_strategies:
        - name: "academic"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            batch_size: 3
          default: true
      retrieval_strategies:
        - name: "filtered"
          type: "MetadataFilteredStrategy"
          config:
            top_k: 5
            filters:
              file_type: ["pdf", "txt"]
          default: true
  
  data_processing_strategies:
    - name: "research"
      description: "Academic paper processing"
      directory_config:
        recursive: true
        include_patterns: ["*.pdf", "*.txt", "*.md"]
        exclude_patterns: ["*.draft.*", "temp_*"]
        allowed_extensions: [".pdf", ".txt", ".md"]
      parsers:
        - type: "PDFParser_LlamaIndex"
          config:
            extract_images: false
            extract_tables: true
        - type: "TextParser_Python"
          config:
            chunk_size: 2000
            chunk_overlap: 300
        - type: "MarkdownParser_LlamaIndex"
          config:
            preserve_formatting: true
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG", "DATE", "GPE"]
        - type: "SummaryExtractor"
          config:
            summary_sentences: 3
```

Usage: `cli.py ingest --strategy research_research_db /path/to/papers/`

### Example 3: FDA Letters Processing (Production Example)

```yaml
rag:
  databases:
    - name: "fda_letters_db"
      type: "ChromaStore"
      config:
        persist_directory: "./demos/vectordb/fda_letters"
        collection_name: "fda_action_letters"
      default_embedding_strategy: "fda_embeddings"
      default_retrieval_strategy: "fda_similarity"
      embedding_strategies:
        - name: "fda_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            batch_size: 5
            timeout: 120
          default: true
      retrieval_strategies:
        - name: "fda_similarity"
          type: "BasicSimilarityStrategy"
          config:
            distance_metric: "cosine"
            top_k: 5
          default: true
  
  data_processing_strategies:
    - name: "fda_letters_processing"
      description: "FDA action letters PDF processing"
      directory_config:
        recursive: true
        include_patterns: ["*.pdf"]
        exclude_patterns: ["*.tmp", "~*"]
        allowed_extensions: [".pdf"]
        max_files: 1000
      parsers:
        - type: "PDFParser_LlamaIndex"
          config:
            chunk_size: 1000
            chunk_overlap: 200
            extract_metadata: true
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["ORG", "DATE", "PRODUCT"]
            use_fallback: true
        - type: "PatternExtractor"
          config:
            predefined_patterns: ["nda_number", "date", "company"]
```

---

## Parser System

### Parser Types Available

All parsers follow the naming convention: `{ParserType}_{Implementation}`

- **Text**: `TextParser_LlamaIndex`, `TextParser_Python`
- **PDF**: `PDFParser_LlamaIndex`, `PDFParser_PyPDF2`
- **CSV**: `CSVParser_LlamaIndex`, `CSVParser_Pandas`
- **Markdown**: `MarkdownParser_LlamaIndex`, `MarkdownParser_Python`
- **HTML**: `HTMLParser_LlamaIndex`, `HTMLParser_BeautifulSoup`
- **DOCX**: `DocxParser_LlamaIndex`, `DocxParser_Python`

### DirectoryParser (ALWAYS ACTIVE)

The DirectoryParser is automatically active at the strategy level and handles:
- **Single Files**: Processes individual files directly
- **Directories**: Recursively processes based on patterns
- **Filtering**: Applies include/exclude patterns

```python
# Automatically handles both:
cli.py ingest --strategy my_strategy /path/to/file.txt     # Single file
cli.py ingest --strategy my_strategy /path/to/directory/    # Directory
```

---

## Directory Structure

```
/rag/
├── core/
│   └── strategies/
│       ├── __init__.py          # Exports SchemaHandler, StrategyLoader
│       ├── handler.py           # SchemaHandler class (main)
│       ├── loader.py            # StrategyLoader (new schema only)
│       ├── config.py.deprecated # OLD - DO NOT USE
│       └── manager.py.deprecated # OLD - DO NOT USE
├── components/
│   └── parsers/
│       ├── __init__.py          # DirectoryParser, factory methods
│       ├── text/                # Text parsers
│       ├── pdf/                 # PDF parsers
│       └── ...                  # Other parser types
├── demos/
│   ├── demo_strategies.yaml    # Demo strategy configurations
│   ├── demo1_research_papers_cli.py
│   ├── demo_fda_letters_interactive.py
│   └── static_samples/          # Sample data for demos
└── schema.yaml                  # Main schema definition
```

---

## Running Demos

All demos use the new schema format exclusively:

```bash
# Demo 1: Research Papers (12 documents)
DEMO_MODE=automated uv run python demos/demo1_research_papers_cli.py

# Demo 2: Customer Support
DEMO_MODE=automated uv run python demos/demo2_customer_support_cli.py

# Demo 3: Code Documentation (43 documents)
DEMO_MODE=automated uv run python demos/demo3_code_documentation_cli.py

# FDA Demo: Process 40+ FDA letters (706 chunks)
DEMO_MODE=automated uv run python demos/demo_fda_letters_interactive.py
```

---

## CLI Commands

### Basic Usage

```bash
# View available strategies (combines processing + database)
uv run python cli.py --strategy-file config.yaml strategies list

# Ingest documents (single file or directory)
uv run python cli.py --strategy-file config.yaml ingest \
  --strategy text_processing_main_db /path/to/documents

# Search
uv run python cli.py --strategy-file config.yaml search \
  --strategy text_processing_main_db "your query"

# Get info about a collection
uv run python cli.py --strategy-file config.yaml info \
  --strategy text_processing_main_db
```

### Strategy Naming

Remember: strategies are named as `{processing}_{database}`:
- `text_processing_research_db`
- `pdf_processing_fda_db`
- `markdown_processing_docs_db`

---

## Migration Notes

### What Changed (2025-09-09)

1. **Removed ALL Legacy Code**:
   - No more `_convert_rag_to_legacy()`
   - No more `StrategyManager` or `StrategyConfig`
   - Files renamed to `.deprecated`

2. **DirectoryParser Always Active**:
   - Configured at strategy level via `directory_config`
   - Handles both single files and directories
   - File filtering ONLY in `directory_config`

3. **Clean File Structure**:
   - `handler.py` - Main SchemaHandler
   - `loader.py` - Loads new schema only
   - No backward compatibility

4. **Parser Fallbacks**:
   - LlamaIndex parsers used as defaults
   - No more `simple_text_parser.py`

### Breaking Changes

- Old config format will NOT work
- Must have `rag:` top-level key
- Strategies must specify both `default_embedding_strategy` and `default_retrieval_strategy`
- Strategy names must follow `{processing}_{database}` format

### How to Migrate Old Configs

Old format:
```yaml
strategies:
  my_strategy:
    components:
      parser: {type: "text"}
      embedder: {type: "ollama"}
      vector_store: {type: "chroma"}
```

New format:
```yaml
rag:
  databases:
    - name: "my_db"
      type: "ChromaStore"
      config: {}
      default_embedding_strategy: "my_embed"
      default_retrieval_strategy: "my_retrieve"
      embedding_strategies:
        - name: "my_embed"
          type: "OllamaEmbedder"
          config: {}
      retrieval_strategies:
        - name: "my_retrieve"
          type: "BasicSimilarityStrategy"
          config: {}
  
  data_processing_strategies:
    - name: "my_processing"
      directory_config: {}
      parsers:
        - type: "TextParser_LlamaIndex"
          config: {}
```

---

## Testing

Run tests to verify the new schema:
```bash
# Test schema validation
uv run pytest tests/test_schema_verifier.py -v

# Test strategy loading
uv run pytest tests/test_strategies.py -v

# Test demos
for i in 1 2 3 4 5 6; do
  echo "Testing demo$i..."
  DEMO_MODE=automated timeout 30 uv run python demos/demo${i}_*.py
done
```

---

## Important Notes

1. **ALWAYS use `uv run`** for Python commands
2. **NO backward compatibility** - must use new schema
3. **DirectoryParser is always active** but handles single files too
4. **Strategy names** follow `{processing}_{database}` format
5. **LlamaIndex parsers** are the default fallbacks
6. **File filtering** happens ONLY in `directory_config`

---

Last Updated: 2025-09-09
Migration Complete: All legacy code removed, new schema only!