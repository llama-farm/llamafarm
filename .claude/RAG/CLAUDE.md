# LlamaFarm RAG System - Complete Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Schema System](#schema-system)
4. [MIME Type Filtering](#mime-type-filtering)
5. [Parser System](#parser-system)
6. [Directory Structure](#directory-structure)
7. [Configuration Guide](#configuration-guide)
8. [Running Demos](#running-demos)
9. [Testing](#testing)
10. [CLI Commands](#cli-commands)
11. [Development Guide](#development-guide)

ALWAYS USE UV for python!

---

## Overview

The LlamaFarm RAG (Retrieval-Augmented Generation) system is a comprehensive, schema-driven document processing pipeline with advanced MIME type filtering, modular parsers, and flexible configuration management.

### Key Features
- **Schema-Driven Configuration**: All configurations validated against a comprehensive JSON schema
- **Two-Tier MIME Type Filtering**: Strategy-level and parser-level file type control
- **Modular Architecture**: Pluggable parsers, extractors, embedders, and stores
- **Database Abstraction**: Support for multiple vector databases with strategy defaults
- **Priority-Based Parser Selection**: Intelligent parser routing based on file types
- **LlamaIndex Integration**: Advanced document parsing with multiple fallback strategies

---

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Configuration (YAML)                      │
│                         schema.yaml                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ Loader  │
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐    ┌─────▼─────┐   ┌─────▼─────┐
   │Database │    │Processing │   │   MIME    │
   │ Config  │    │Strategies │   │ Filtering │
   └────┬────┘    └─────┬─────┘   └─────┬─────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
              ┌──────────┼──────────┐
              │          │          │
         ┌────▼───┐ ┌───▼───┐ ┌───▼────┐
         │Parsers │ │Embedder│ │Stores  │
         └────────┘ └────────┘ └────────┘
```

### Component Flow

1. **Configuration Loading**: YAML configs validated against schema
2. **MIME Type Filtering**: Two-tier filtering system
   - Strategy level: Which file types the strategy accepts
   - Parser level: Which parser handles each file type
3. **Document Processing**: Parse → Extract → Embed → Store
4. **Retrieval**: Query → Embed → Search → Retrieve

---

## Schema System

### Schema Location
- **Main Schema**: `/rag/schema.yaml`
- **Validator**: `/rag/tests/test_schema_verifier.py`

### Schema Structure

```yaml
# Top-level structure
rag:
  databases: []           # Vector database configurations
  data_processing_strategies: []  # Processing pipelines

# Database schema
databases:
  - name: string
    type: string
    config: object
    default_embedding_strategy: string  # NEW
    default_retrieval_strategy: string  # NEW
    embedding_strategies: []
    retrieval_strategies: []

# Processing strategy schema
data_processing_strategies:
  - name: string
    description: string
    allowed_mime_types: []  # Strategy-level filtering
    allowed_extensions: []  # File extension filtering
    parsers: []
    extractors: []
```

### Key Schema Features

1. **Database Defaults**: Each database can specify default embedding and retrieval strategies
2. **MIME Type Filtering**: Strategies can restrict accepted file types
3. **Parser Routing**: Parsers specify which MIME types they handle
4. **Priority System**: Parser selection based on priority when multiple match

---

## MIME Type Filtering

### Two-Tier Filtering System

#### 1. Strategy-Level Filtering
Controls which files a strategy will accept:

```yaml
data_processing_strategies:
  - name: "pdf_only_strategy"
    allowed_mime_types: ["application/pdf"]
    allowed_extensions: [".pdf", ".PDF"]
    # This strategy ONLY processes PDF files
```

#### 2. Parser-Level Routing
Routes accepted files to appropriate parsers:

```yaml
parsers:
  - type: "PDFParser_LlamaIndex"
    mime_types: ["application/pdf"]
    file_extensions: [".pdf", ".PDF"]
    priority: 10  # Higher priority wins
```

### Filtering Rules

1. **Empty Arrays = Accept All**: `allowed_mime_types: []` accepts all files
2. **Priority-Based Selection**: When multiple parsers match, highest priority wins
3. **Fallback Support**: Lower priority parsers act as fallbacks

### Example: Multi-Format Strategy

```yaml
- name: "business_documents"
  # Accept multiple document types
  allowed_mime_types: [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
  ]
  allowed_extensions: [".pdf", ".docx", ".xlsx"]
  parsers:
    - type: "PDFParser_LlamaIndex"
      mime_types: ["application/pdf"]
      priority: 10
    - type: "DocxParser_LlamaIndex"
      mime_types: ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
      priority: 10
    - type: "ExcelParser_LlamaIndex"
      mime_types: ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
      priority: 10
```

---

## Parser System

### Available Parsers

#### Python-Based Parsers
- `PDFParser_Python`: Basic PDF text extraction
- `TextParser_Python`: Plain text processing
- `MarkdownParser_Python`: Markdown with structure preservation
- `CSVParser_Pandas`: Structured data processing

#### LlamaIndex Parsers
- `PDFParser_LlamaIndex`: Advanced PDF with fallback strategies
- `DocxParser_LlamaIndex`: Word documents
- `ExcelParser_LlamaIndex`: Spreadsheets
- `CSVParser_LlamaIndex`: CSV with schema inference
- `MarkdownParser_LlamaIndex`: Advanced markdown parsing
- `TextParser_LlamaIndex`: Fallback text parser
- `WebParser_LlamaIndex`: HTML content
- `CodeParser_LlamaIndex`: Source code with AST

### Parser Configuration

```yaml
parsers:
  - type: "PDFParser_LlamaIndex"
    mime_types: ["application/pdf"]
    file_extensions: [".pdf"]
    priority: 10
    config:
      chunk_strategy: "semantic"  # sentences, paragraphs, pages, semantic
      chunk_size: 1000
      chunk_overlap: 200
      extract_metadata: true
      extract_images: true
      extract_tables: true
      fallback_strategies: [
        "llama_pdf_reader",
        "llama_pymupdf_reader",
        "pypdf2_fallback"
      ]
```

---

## Directory Structure

```
rag/
├── .claude/                    # Claude-specific documentation
│   └── CLAUDE.md              # This file
├── core/                      # Core system components
│   ├── mime_type_filter.py   # MIME type filtering system
│   └── strategies/
│       └── loader.py          # Strategy loader with MIME support
├── components/                # Modular components
│   ├── parsers/              # Document parsers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── pdf_parser.py
│   │   ├── text_parser.py
│   │   ├── markdown_parser.py
│   │   ├── csv_parser.py
│   │   └── llamaindex/       # LlamaIndex parsers
│   ├── extractors/           # Metadata extractors
│   ├── embedders/            # Embedding models
│   └── stores/               # Vector stores
├── demos/                    # Demo configurations
│   ├── demo_strategies.yaml # Main demo config
│   └── static_samples/      # Sample documents
├── samples/                  # Example configurations
│   └── llamaindex_parser_strategies.yaml
├── tests/                    # Test suite
│   ├── test_schema_verifier.py  # Schema validation
│   ├── test_mime_filtering.py   # MIME type tests
│   └── test_data/
│       └── test_strategies.yaml
├── schema.yaml              # Main schema definition
├── test_mime_filtering_demo.py  # MIME demo script
└── TODO.md                  # Future tasks

../config/templates/
└── default.yaml            # Default configuration template
```

---

## Configuration Guide

### Basic Configuration Structure

```yaml
rag:
  databases:
    - name: "main_db"
      type: "ChromaStore"
      config:
        persist_directory: "./vectordb"
        distance_function: "cosine"
      # NEW: Default strategies
      default_embedding_strategy: "primary_embeddings"
      default_retrieval_strategy: "semantic_search"
      embedding_strategies:
        - name: "primary_embeddings"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
          default: true
      retrieval_strategies:
        - name: "semantic_search"
          type: "BasicSimilarityStrategy"
          config:
            top_k: 10
          default: true

  data_processing_strategies:
    - name: "document_processing"
      # MIME type filtering
      allowed_mime_types: ["application/pdf", "text/plain"]
      allowed_extensions: [".pdf", ".txt"]
      parsers:
        - type: "PDFParser_LlamaIndex"
          mime_types: ["application/pdf"]
          priority: 10
        - type: "TextParser_Python"
          mime_types: ["text/plain"]
          priority: 5
      extractors:
        - type: "EntityExtractor"
          config:
            entity_types: ["PERSON", "ORG", "DATE"]
```

### Configuration Examples

1. **PDF-Only Strategy**: See `samples/llamaindex_parser_strategies.yaml`
2. **Multi-Format Business**: See `demos/demo_strategies.yaml`
3. **Generic Processing**: See `config/templates/default.yaml`

---

## Running Demos

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure Ollama is running
ollama serve

# Pull required model
ollama pull nomic-embed-text
```

### Demo Commands

```bash
# 1. Run MIME type filtering demo
python test_mime_filtering_demo.py

# 2. Test configuration validation
python tests/test_schema_verifier.py demos/demo_strategies.yaml

# 3. Run CLI with demo strategy
python cli.py ingest ./demos/static_samples --strategy text_processing

# 4. Search ingested documents
python cli.py search "your query" --strategy text_processing
```

### Demo Strategies Available

1. **text_processing**: Generic text document processing
2. **csv_processing**: CSV/structured data only
3. **markdown_processing**: Markdown with structure preservation
4. **business_processing**: Multi-format business documents
5. **llamaindex_pdf_processing**: Advanced PDF processing
6. **multi_format_llamaindex**: LlamaIndex multi-format

---

## Testing

### Running Tests

```bash
# Run all tests with UV
uv run pytest tests/ -v

# Run specific test categories
uv run pytest tests/test_mime_filtering.py -v
uv run pytest tests/test_parsers.py -v
uv run pytest tests/test_strategies.py -v

# Run schema validation
python tests/test_schema_verifier.py --summary \
  demos/demo_strategies.yaml \
  samples/*.yaml \
  ../config/templates/default.yaml
```

### Test Coverage

- **MIME Type Filtering**: `tests/test_mime_filtering.py`
- **Parser Tests**: `tests/test_parsers.py`
- **Strategy Loading**: `tests/test_strategies.py`
- **Schema Validation**: `tests/test_schema_verifier.py`

### Adding Tests

```python
# Example test for new parser
def test_new_parser():
    parser = NewParser(config={...})
    result = parser.parse("test_file.ext")
    assert result.success
    assert len(result.chunks) > 0
```

---

## CLI Commands

### Basic Commands

```bash
# Initialize configuration
python cli.py init

# Ingest documents
python cli.py ingest <path> --strategy <strategy_name>

# Search documents
python cli.py search "query" --strategy <strategy_name>

# List available strategies
python cli.py info

# Test configuration
python cli.py test
```

### Advanced Options

```bash
# Override embedding strategy
python cli.py ingest ./docs \
  --strategy business_processing \
  --embedding-strategy fast_embeddings

# Override retrieval strategy
python cli.py search "query" \
  --strategy business_processing \
  --retrieval-strategy metadata_filtered

# Use custom configuration
python cli.py --config my_config.yaml ingest ./docs
```

---

## Development Guide

### Adding a New Parser

1. Create parser file in `components/parsers/`:

```python
# components/parsers/my_parser.py
from .base import BaseParser

class MyParser(BaseParser):
    def parse(self, file_path: str) -> ParseResult:
        # Implementation
        pass
```

2. Register in `components/parsers/__init__.py`:

```python
from .my_parser import MyParser

__all__ = [..., "MyParser"]
```

3. Add to configuration:

```yaml
parsers:
  - type: "MyParser"
    mime_types: ["application/x-myformat"]
    file_extensions: [".myext"]
    priority: 10
    config:
      # Parser-specific config
```

### Adding MIME Type Support

1. Add MIME type mapping in `core/mime_type_filter.py`:

```python
MIME_TYPE_EXTENSIONS = {
    "application/x-myformat": [".myext", ".MYEXT"],
    # ...
}
```

2. Configure strategy to accept it:

```yaml
allowed_mime_types: ["application/x-myformat"]
allowed_extensions: [".myext"]
```

### Testing Your Changes

```bash
# Validate configuration
python tests/test_schema_verifier.py your_config.yaml

# Test MIME filtering
python test_mime_filtering_demo.py

# Run parser tests
uv run pytest tests/test_parsers.py::test_your_parser -v
```

---

## Best Practices

### 1. Configuration Design

- **Specialized Strategies**: Create focused strategies for specific file types
- **Generic Fallbacks**: Include generic strategies for flexibility
- **Priority Management**: Use priority to control parser selection

### 2. MIME Type Filtering

- **Be Explicit**: Specify exact MIME types when possible
- **Use Extensions**: Add file extensions as secondary filters
- **Test Thoroughly**: Validate filtering with test files

### 3. Parser Selection

- **High Priority**: Specialized parsers (priority 10+)
- **Medium Priority**: General parsers (priority 5-9)
- **Low Priority**: Fallback parsers (priority 1-4)

### 4. Performance Optimization

- **Batch Processing**: Use appropriate batch sizes
- **Caching**: Enable embedding caches
- **Parallel Processing**: Use async parsers when available

---

## Troubleshooting

### Common Issues

1. **"No parser found for file type"**
   - Check MIME type detection
   - Verify parser configuration
   - Ensure file extensions match

2. **"Strategy rejected file"**
   - Check `allowed_mime_types`
   - Verify `allowed_extensions`
   - Use empty arrays to accept all

3. **"Schema validation failed"**
   - Run schema verifier
   - Check required fields
   - Validate YAML syntax

### Debug Commands

```bash
# Test MIME type detection
python -c "from core.mime_type_filter import MimeTypeFilter; 
          f = MimeTypeFilter(); 
          print(f.get_mime_type('test.pdf'))"

# Validate all configs
find . -name "*.yaml" -exec python tests/test_schema_verifier.py {} \;

# Test parser directly
python -c "from components.parsers import PDFParser_LlamaIndex;
          p = PDFParser_LlamaIndex();
          print(p.parse('test.pdf'))"
```

---

## Migration Guide

### From Legacy Format to New RAG Schema

#### Old Format (DEPRECATED):
```yaml
strategies:
  - name: "my_strategy"
    components:
      parser: {...}
      embedder: {...}
      vector_store: {...}
```

#### New Format (REQUIRED):
```yaml
rag:
  databases:
    - name: "my_db"
      type: "ChromaStore"
      embedding_strategies: [...]
      retrieval_strategies: [...]
  
  data_processing_strategies:
    - name: "my_strategy"
      parsers: [...]
      extractors: [...]
```

### Key Changes

1. **Separation of Concerns**: Databases and processing strategies are separate
2. **Default Strategies**: Databases specify default embedding/retrieval
3. **MIME Type Filtering**: Strategies can restrict file types
4. **Parser Routing**: Parsers specify which files they handle
5. **No Legacy Support**: Old format will cause validation errors

---

## Future Roadmap

See `TODO.md` for detailed future tasks including:
- Converting extractors to new format
- Adding more LlamaIndex parsers
- Implementing async processing
- Adding more vector database support
- Performance optimizations

---

## Support

For questions or issues:
1. Check this documentation
2. Review example configurations
3. Run tests to verify setup
4. Check TODO.md for known limitations

---

*Last Updated: 2024*
*Schema Version: 2.0*
*Documentation Version: 1.0*