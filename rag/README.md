# RAG System - Next Generation Document Processing

A powerful, extensible RAG (Retrieval-Augmented Generation) system featuring **new schema-based configuration**, **automatic file routing**, and **unified vector storage**. Built for production-ready document processing with the new v1 schema format.

## 🎉 What's New in v1

- **📋 New RAG Schema Format**: Clean, structured YAML with `databases` and `data_processing_strategies`
- **🌍 Global Config Integration**: Uses `llamafarm.yaml` for unified project configuration
- **🚀 DirectoryParser Always Active**: Automatic file detection and routing at strategy level
- **🔄 Unified Vector Store**: All databases can share processing strategies
- **🎯 Clear CLI Arguments**: `--database` and `--data-processing-strategy` for explicit control
- **📦 Enhanced Parser System**: Naming convention `{ParserType}_{Implementation}` (e.g., `PDFParser_LlamaIndex`)

## 🌟 Key Features

- **🎯 Schema-First Design**: Configure entire pipelines through the new RAG schema
- **🔍 Advanced Retrieval**: Multiple retrieval strategies (similarity, reranked, filtered, hybrid)
- **🚫 Deduplication System**: Hash-based document and chunk deduplication
- **📊 Document Management**: Full CRUD operations with version tracking
- **🔧 Modular Components**: Pluggable parsers, extractors, embedders, and stores
- **💻 CLI-First**: Comprehensive command-line interface for all operations
- **🧹 Automatic Cleanup**: Built-in collection management and cleanup

## 📚 Documentation

- **[Schema Documentation](docs/SCHEMA.md)** - Complete v1 schema reference
- **[Strategy System](docs/STRATEGY_SYSTEM.md)** - How strategies work
- **[Parser Guide](docs/PARSERS.md)** - Available parsers and configuration
- **[CLI Guide](docs/CLI.md)** - Command-line interface documentation
- **[Demos Guide](demos/README.md)** - Interactive demonstrations
- **[Component Guide](docs/COMPONENTS.md)** - Extractors, embedders, stores

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **UV** (recommended) - [UV Installation](https://docs.astral.sh/uv/)
- **Ollama** (for local embeddings) - [Download](https://ollama.com/download)

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag/

# Using UV (recommended)
uv sync

# Setup Ollama
ollama serve                     # Start in separate terminal
ollama pull nomic-embed-text     # Download embedding model
```

### 2. Test All Strategies

```bash
# Test all built-in strategies
python demos/demo_test_strategies_quick.py

# Expected output:
# ✅ pdf_processing          PASS
# ✅ text_processing         PASS
# ✅ markdown_processing     PASS
# ✅ csv_processing          PASS
# ✅ multi_format_llamaindex PASS
# ✅ auto_processing         PASS
```

### 3. Basic Usage with Global Config

```bash
# Create a llamafarm.yaml config (see config/templates/default.yaml for example)

# Ingest documents with database and processing strategy
uv run python cli.py --config llamafarm.yaml \
    ingest path/to/document.pdf \
    --database main_database \
    --data-processing-strategy pdf_processing

# Search across all documents
uv run python cli.py --config llamafarm.yaml \
    search "your query" \
    --database main_database

# Search with specific retrieval strategy
uv run python cli.py --config llamafarm.yaml \
    search "your query" \
    --database main_database \
    --retrieval-strategy filtered_search

# View collection info
uv run python cli.py --config llamafarm.yaml \
    info --database main_database
```

## 📖 New Schema Structure

The v1 schema uses a clean, organized structure with two main sections:

### Databases Configuration
```yaml
databases:
  - name: "main_database"
    type: "ChromaStore"
    config:
      collection_name: "documents"
    embedding_strategies:
      - name: "default_embeddings"
        type: "OllamaEmbedder"
        config:
          model: "nomic-embed-text"
    retrieval_strategies:
      - name: "basic_search"
        type: "BasicSimilarityStrategy"
        config:
          top_k: 10
```

### Data Processing Strategies
```yaml
data_processing_strategies:
  - name: "pdf_processing"
    description: "Standard PDF document processing"
    # DirectoryParser configuration (ALWAYS ACTIVE)
    directory_config:
      recursive: true
      include_patterns: ["*.pdf"]
      allowed_mime_types: ["application/pdf"]
      allowed_extensions: [".pdf"]
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

## 🔧 Available Parsers

All parsers follow the naming convention `{Type}_{Implementation}`:

| Parser Type | Implementations | File Types |
|------------|----------------|------------|
| **PDF** | `PDFParser_LlamaIndex`, `PDFParser_PyPDF2` | `.pdf` |
| **Text** | `TextParser_LlamaIndex`, `TextParser_Python` | `.txt`, `.log` |
| **CSV** | `CSVParser_LlamaIndex`, `CSVParser_Pandas`, `CSVParser_Python` | `.csv`, `.tsv` |
| **Excel** | `ExcelParser_LlamaIndex`, `ExcelParser_Pandas`, `ExcelParser_OpenPyXL` | `.xlsx`, `.xls` |
| **Word** | `DocxParser_LlamaIndex`, `DocxParser_PythonDocx` | `.docx` |
| **Markdown** | `MarkdownParser_LlamaIndex`, `MarkdownParser_Python` | `.md`, `.markdown` |

## 🎯 Strategy Naming Convention

Strategies are named using the pattern: `{data_processing_strategy}_{database_name}`

Examples:
- `pdf_processing_main_database`
- `text_processing_research_database`
- `multi_format_llamaindex_main_database`

## 🚦 How It Works

1. **DirectoryParser** (always active) scans files/directories based on `directory_config`
2. Files are filtered by MIME type and extension at strategy level
3. Each file is routed to the appropriate parser based on its type
4. Parsers process documents into chunks
5. Extractors enrich chunks with metadata
6. Embedders generate vector representations
7. Vectors are stored in the configured database
8. Retrieval strategies enable searching

## 📋 Built-in Strategies

From `config/templates/default.yaml`:

1. **pdf_processing** - Optimized for PDF documents
2. **text_processing** - Plain text file processing
3. **markdown_processing** - Markdown with structure preservation
4. **csv_processing** - Structured data from CSV files
5. **multi_format_llamaindex** - Handles multiple formats with LlamaIndex
6. **auto_processing** - Generic fallback for any text-like file

## 🧪 Testing

```bash
# Run all tests
uv run pytest tests/

# Test specific functionality
uv run pytest tests/test_strategies.py -v

# Quick integration test
python demos/demo_test_all_strategies.py
```

## 📊 Example: Complete Workflow

```bash
# 1. Start with a clean slate
python cli.py --strategy-file config/templates/default.yaml \
    manage delete --all --strategy pdf_processing_main_database

# 2. Ingest various document types
python cli.py --strategy-file config/templates/default.yaml \
    ingest docs/papers/ --strategy pdf_processing_main_database

python cli.py --strategy-file config/templates/default.yaml \
    ingest docs/data.csv --strategy csv_processing_main_database

# 3. Search across all documents (same vector store)
python cli.py --strategy-file config/templates/default.yaml \
    search "machine learning results" --top-k 5 \
    --strategy text_processing_main_database

# 4. Get collection statistics
python cli.py --strategy-file config/templates/default.yaml \
    info --strategy pdf_processing_main_database
```

## 🛠️ Advanced Configuration

See [config/templates/advanced.yaml](config/templates/advanced.yaml) for examples of:
- Multiple databases with different settings
- Custom embedding strategies per database
- Advanced retrieval strategies (hybrid, reranked)
- Complex parser configurations
- Custom extractor pipelines

## 📝 Creating Custom Strategies

1. Create a YAML file following the schema:
```yaml
databases:
  - name: "custom_db"
    type: "ChromaStore"
    # ... database config

data_processing_strategies:
  - name: "custom_processing"
    directory_config:
      # File filtering rules
    parsers:
      # Parser configurations
    extractors:
      # Extractor pipeline
```

2. Use with CLI:
```bash
python cli.py --strategy-file my_strategies.yaml \
    ingest documents/ --strategy custom_processing_custom_db
```

## 🤝 Contributing

Contributions are welcome! Please ensure:
- New parsers follow the `{Type}_{Implementation}` naming convention
- Strategies follow the new RAG schema format
- Tests are added for new functionality
- Documentation is updated

## 📄 License

[License information here]

## 🙏 Acknowledgments

Built with:
- LlamaIndex for document parsing
- ChromaDB for vector storage
- Ollama for local embeddings
- Rich for beautiful CLI output