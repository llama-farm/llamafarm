# LlamaFarm Parser Guide

This guide explains the different parser options available in LlamaFarm RAG.

## Parser Comparison

| Parser | Dependencies | Performance | Use Case |
|--------|-------------|-------------|----------|
| **UniversalParser** | Lightweight | Fast | Simple docs, minimal deps |
| **DoclingParser** | Heavy | Accurate | Complex layouts, tables |
| **MarkItDownParser** | Light | Very Fast | Quick conversion |
| **PyPDF2Parser** | Light | Medium | PDFs, basic extraction |
| **LlamaIndexParser** | Very Heavy | Feature-rich | Advanced workflows |

## UniversalParser (Recommended for Most Users)

The UniversalParser combines MarkItDown + SemChunk for lightweight, structure-preserving document processing.

### Features
- Supports PDF, DOCX, XLSX, HTML, JSON, XML, TXT, MD, and images
- Semantic chunking that respects sentence boundaries
- Optional OCR fallback for scanned documents
- Minimal dependencies (no llama-index, no docling)

### Configuration
```yaml
rag:
  parser:
    type: "universal"
    chunk_size: 512           # Target chunk size in tokens
    chunk_strategy: "semantic" # Options: semantic, page
    use_ocr: true             # Enable OCR fallback
    ocr_min_text_threshold: 50 # Min chars before OCR triggers
```

### When to Use
- Basic document processing
- Quick prototyping
- Resource-constrained environments
- Most common use cases (90% of users)

## DoclingParser

IBM's Docling provides advanced document understanding with layout analysis.

### Features
- Deep layout analysis
- Table extraction
- Figure detection
- Academic paper processing

### Configuration
```yaml
rag:
  parser:
    type: "DoclingParser"
    chunk_size: 1000
    extract_tables: true
    extract_headings: true
    ocr_enabled: true
```

### When to Use
- Complex PDF layouts
- Scanned documents with tables
- Academic papers
- Financial documents

### Installation
```bash
uv add docling
```

## MarkItDownParser

Microsoft's MarkItDown for simple document-to-markdown conversion.

### Features
- Fast conversion
- Preserves basic structure
- Supports many formats

### Configuration
```yaml
rag:
  parser:
    type: "MarkItDownParser"
```

### When to Use
- Simple format conversion
- Markdown-first workflows
- Lightweight processing

## PyPDF2Parser

Basic PDF parsing using PyPDF2.

### Features
- Text extraction
- Metadata extraction
- Page-by-page processing

### Configuration
```yaml
rag:
  parser:
    type: "PDFParser_PyPDF2"
    chunk_size: 1000
    chunk_strategy: "paragraphs"
    preserve_layout: true
```

### When to Use
- Simple PDFs
- Text-heavy documents
- When other parsers aren't needed

## LlamaIndexParser

Full LlamaIndex integration for advanced workflows.

### Features
- Multiple reader types
- Automatic format detection
- Integration with LlamaIndex pipelines

### Configuration
```yaml
rag:
  parser:
    type: "LlamaIndexParser"
```

### When to Use
- Existing LlamaIndex workflows
- Advanced RAG pipelines
- Multiple document types

### Installation
```bash
uv add "llama-rag[llamaindex]"
```

## Choosing a Parser

### Quick Decision Tree

1. **Just need basic document processing?** → UniversalParser
2. **Complex PDFs with tables?** → DoclingParser
3. **Already using LlamaIndex?** → LlamaIndexParser
4. **Simple PDFs only?** → PyPDF2Parser
5. **Quick markdown conversion?** → MarkItDownParser

### Performance Considerations

- **Memory**: UniversalParser < PyPDF2 < MarkItDown < Docling < LlamaIndex
- **Speed**: MarkItDown > UniversalParser > PyPDF2 > Docling > LlamaIndex
- **Accuracy**: Docling > LlamaIndex > UniversalParser > MarkItDown > PyPDF2

## Custom Parsers

You can create custom parsers by extending `BaseParser`:

```python
from components.parsers.base.base_parser import BaseParser, ParserConfig
from core.base import Document, ProcessingResult

class MyCustomParser(BaseParser):
    def _load_metadata(self) -> ParserConfig:
        return ParserConfig(
            name="MyCustomParser",
            display_name="My Custom Parser",
            version="1.0.0",
            supported_extensions=[".custom"],
            mime_types=["application/custom"],
            capabilities=["custom_extraction"],
            dependencies={},
            default_config={},
        )

    def can_parse(self, file_path: str) -> bool:
        return file_path.endswith(".custom")

    def parse(self, source: str) -> ProcessingResult:
        # Your parsing logic here
        pass
```

See `rag/components/parsers/` for more examples.
