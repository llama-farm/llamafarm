# Universal RAG Example

This example demonstrates **zero-config RAG** using LlamaFarm's built-in `universal_rag` strategy.

## What is Universal RAG?

When you define a RAG database but **don't specify any `data_processing_strategies`**, LlamaFarm automatically uses `universal_rag`, which:

- **Parses 90%+ of document formats** (PDF, DOCX, MD, TXT, HTML, CSV, JSON, XLSX, PPTX, etc.)
- **Chunks intelligently** using semantic chunking that respects content boundaries
- **Extracts rich metadata** including keywords, entities, language detection, and more
- **Handles OCR** for scanned documents and images (via LlamaFarm API)

**Zero config. Just point at files and go.**

## Quick Start

### Prerequisites

1. LlamaFarm server running (`nx start server`)
2. Universal runtime running (`nx start universal-runtime`)
3. LlamaFarm CLI built (`nx build cli`)

### Run the Demo

```bash
bash examples/universal_rag/demo.sh
```

### What the Demo Shows

1. **Create a dataset** without specifying a data processing strategy
2. **Upload mixed-format documents** (MD, TXT in this example)
3. **Process automatically** with UniversalParser and UniversalExtractor
4. **Query with citations** showing source document names
5. **View metadata** added by the universal extractor

## Configuration

The `llamafarm.yaml` in this example intentionally has **NO** `data_processing_strategies`:

```yaml
rag:
  databases:
    - name: universal_db
      type: ChromaStore
      # ... embedding and retrieval config ...
  # NOTE: No data_processing_strategies!
  # This triggers automatic universal_rag usage.

datasets:
  - name: universal_docs
    # No explicit data_processing_strategy
    database: universal_db
```

## Supported File Types

UniversalParser supports:

| Format | Extensions | Notes |
|--------|-----------|-------|
| Text | `.txt` | Plain text files |
| Markdown | `.md` | With header-based chunking |
| PDF | `.pdf` | Text extraction, OCR for images |
| Word | `.docx` | Full content extraction |
| Excel | `.xlsx`, `.xls` | Tabular data as text |
| PowerPoint | `.pptx` | Slide content extraction |
| HTML | `.html`, `.htm` | Clean text extraction |
| CSV | `.csv` | Row-based chunking |
| JSON | `.json` | Structured content |
| Images | `.png`, `.jpg` | OCR via LlamaFarm API |

## Metadata Extracted

UniversalExtractor automatically adds:

- **chunk_index**: Position in document (0, 1, 2, ...)
- **chunk_label**: Human-readable position ("1/5", "2/5", ...)
- **total_chunks**: Total chunks from this document
- **document_name**: Source filename
- **document_type**: File type detected
- **word_count**: Words in this chunk
- **character_count**: Characters in this chunk
- **keywords**: Top keywords extracted (via YAKE)
- **language**: Detected language (en, es, fr, etc.)
- **processed_at**: ISO timestamp

## Backward Compatibility

Universal RAG is **100% backward compatible**:

- Existing configs with explicit `data_processing_strategies` work unchanged
- Legacy parsers (`PDFParser_PyPDF2`, `TextParser_LlamaIndex`, etc.) still work
- You can mix universal and legacy strategies in the same project

## API Usage

You can also use the LlamaFarm API directly:

```bash
# Create dataset
curl -X POST http://localhost:8000/v1/datasets \
  -H "Content-Type: application/json" \
  -d '{"name": "my_docs", "database": "universal_db"}'

# Upload document
curl -X POST http://localhost:8000/v1/datasets/my_docs/upload \
  -F "file=@document.pdf"

# Process (uses universal_rag automatically)
curl -X POST http://localhost:8000/v1/datasets/my_docs/process

# Query
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -d '{"query": "your question", "database": "universal_db", "top_k": 5}'
```

## Learn More

- [RAG Documentation](../../rag/README.md)
- [Schema Reference](../../schema/README.md)
- [Quick RAG Example](../quick_rag/README.md) - Using explicit strategies
