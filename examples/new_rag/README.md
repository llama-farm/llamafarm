# New RAG System Examples

This directory contains examples and demos for the new RAG parsing system that uses **Docling** and **MarkItDown** parsers with the **SimpleChunker** module.

## Quick Start

```bash
# Run all local demos (no server required)
bash run_all_demos.sh

# Run with API demos (requires server)
bash run_all_demos.sh --api

# Run everything
bash run_all_demos.sh --all
```

## New Parsers

### DoclingParser (Recommended for PDFs)

IBM's Docling parser provides AI-powered document understanding:
- **97.9% accuracy** on table extraction
- AI-powered layout analysis
- Built-in HybridChunker for smart, tokenizer-aware chunking
- Supports PDF, DOCX, PPTX, XLSX

```yaml
parsers:
  - type: DoclingParser
    file_include_patterns: ["*.pdf"]
    config:
      chunk_size: 512
      chunk_strategy: hybrid  # AI-aware chunking
      output_format: markdown
```

### MarkItDownParser (Recommended for Office Docs)

Microsoft's MarkItDown provides fast document conversion:
- Clean markdown output
- Excellent Office document support
- Fast and reliable

```yaml
parsers:
  - type: MarkItDownParser
    file_include_patterns: ["*.docx", "*.pptx", "*.xlsx"]
    config:
      output_format: markdown
```

## SimpleChunker

Standalone chunking module for flexible text splitting:

```python
from components.chunkers import SimpleChunker, ChunkingStrategy

chunker = SimpleChunker(
    strategy=ChunkingStrategy.SENTENCES,
    chunk_size=512,
    overlap=50,
)

chunks = chunker.chunk(text, metadata={"source": "doc.pdf"})
```

**Strategies:**
- `characters` - Fixed character count
- `sentences` - Sentence boundaries (NLTK)
- `paragraphs` - Paragraph boundaries
- `sections` - Markdown headers
- `pages` - Page metadata
- `tokens` - Token-based (tiktoken)

## Demo Scripts

| Demo | Description |
|------|-------------|
| `demo_01_parser_basics.py` | Basic parsing with Docling and MarkItDown |
| `demo_02_chunking_strategies.py` | Compare different chunking strategies |
| `demo_03_docling_full_pipeline.py` | Full Docling pipeline with HybridChunker |
| `demo_04_simple_config.py` | Configuration comparison (old vs new) |
| `demo_05_parse_api.sh` | Parse-only API endpoint demo |
| `demo_06_full_pipeline.py` | Complete pipeline integration |

## API Endpoints

### Parse Without Storing

```bash
# Parse a PDF and get markdown
curl -X POST "http://localhost:8005/v1/projects/test/demo/rag/parse" \
  -F "file=@document.pdf" \
  -F "output_format=markdown" \
  -F "chunk_strategy=hybrid" \
  -F "chunk_size=512"
```

**Response:**
```json
{
  "content": "# Document Title\n\nParsed content...",
  "format": "markdown",
  "chunks": [
    {"content": "...", "metadata": {"chunk_index": 0}}
  ],
  "metadata": {"page_count": 5, "filename": "document.pdf"},
  "processing_time_ms": 1234.5,
  "parser_used": "DoclingParser"
}
```

### Query Options

| Parameter | Values | Default |
|-----------|--------|---------|
| `output_format` | markdown, text, json | markdown |
| `chunk_strategy` | none, characters, sentences, paragraphs, sections, hybrid | none |
| `chunk_size` | 50-10000 | 512 |
| `chunk_overlap` | 0-1000 | 0 |

## Configuration

See `llamafarm.yaml` for a complete configuration example:

```yaml
rag:
  data_processing_strategies:
    - name: docling_default
      parsers:
        - type: DoclingParser
          file_include_patterns: ["*.pdf"]
          config:
            chunk_strategy: hybrid
            chunk_size: 512
```

## Migration from LlamaIndex

### Before (LlamaIndex)
```yaml
parsers:
  - type: PDFParser_LlamaIndex
    config:
      chunk_size: 1000
      chunk_overlap: 100
      fallback_strategies:
        - llama_pdf_reader
```

### After (Docling)
```yaml
parsers:
  - type: DoclingParser
    config:
      chunk_size: 512  # In tokens
      chunk_strategy: hybrid
```

**Key Changes:**
1. `chunk_size` is now in tokens (not characters)
2. No need for `fallback_strategies`
3. `chunk_strategy: hybrid` provides better results
4. No external API dependencies

## Backward Compatibility

All existing parsers continue to work:

```yaml
parsers:
  # New parser (try first)
  - type: DoclingParser
    file_include_patterns: ["*.pdf"]
    priority: 0

  # Legacy fallback
  - type: PDFParser_PyPDF2
    file_include_patterns: ["*.pdf"]
    priority: 2
```

## Running Tests

```bash
# All tests
cd rag && uv run pytest tests/ -v

# Specific test files
uv run pytest tests/components/parsers/test_docling_parser.py -v
uv run pytest tests/components/chunkers/test_simple_chunker.py -v
uv run pytest tests/test_full_pipeline_integration.py -v
uv run pytest tests/test_schema_validation.py -v
```

## Files in This Directory

```
examples/new_rag/
├── README.md              # This file
├── llamafarm.yaml         # Sample configuration
├── run_all_demos.sh       # Run all demos
├── demo_01_parser_basics.py
├── demo_02_chunking_strategies.py
├── demo_03_docling_full_pipeline.py
├── demo_04_simple_config.py
├── demo_05_parse_api.sh
├── demo_06_full_pipeline.py
├── files/                 # Sample files for demos
└── tests/
    └── test_api_integration.py
```
