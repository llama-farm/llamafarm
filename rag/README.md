# RAG Toolkit

Python helpers used by the server and CLI for ingestion, parsing, and retrieval. Day-to-day usage is through the `lf` CLI—this package exists so you can run workers/tests locally and extend the pipeline.

## Run the Celery Worker (development)
```bash
uv sync
uv run python cli.py worker
```

> You normally don’t need to start this manually. `lf start` and `lf datasets process` will launch a worker via Docker automatically.

### Starting Services with Nx (from repo root)
```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# All-in-one
nx dev

# Separate terminals
nx start rag    # Terminal 1
nx start server # Terminal 2
```

## CLI Utilities (Advanced)
There is still a thin Python CLI in `cli.py` for low-level debugging, but the recommended interface is the Go-based `lf` command. Use the Go CLI for:
- Creating/uploading/processing datasets (`lf datasets …`)
- Querying documents (`lf rag query …`)
- Managing health/stats (`lf rag health|stats`)

## Configuration
The ingestion pipeline is driven by the `rag` section of `llamafarm.yaml` (see `../config/schema.yaml` and `../docs/website/docs/rag/index.md`). Key concepts:
- **Databases** (`ChromaStore`, `QdrantStore`, …) with embedding/retrieval strategy definitions.
- **Data processing strategies** specifying parsers, extractors, metadata processors.
- **Datasets** referencing a processing strategy + database pair.

Update `rag/schema.yaml` when adding new parsers, extractors, or stores, then regenerate types via `config/generate-types.sh`.

## Tests
```bash
uv run python cli.py test   # Smoke tests for strategies
uv run pytest tests/
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
      allowed_extensions: [".pdf"]
    parsers:
      - type: "PDFParser_LlamaIndex"
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

   > You no longer specify `mime_types` in project configs; DirectoryParser
   > handles MIME detection internally. Use `file_extensions` or
   > `file_include_patterns` on parsers to describe the files they accept.
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

## Documentation
User-facing instructions for ingestion, queries, and extending RAG live in `docs/website/docs/rag/index.md`.

Those ensure strategies defined in the templates and schema remain valid.

## Documentation
User-facing instructions for ingestion, queries, and extending RAG live in `docs/website/docs/rag/index.md`.

