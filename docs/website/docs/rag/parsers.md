---
title: Parsers Reference
sidebar_position: 2
---

# Parsers Reference

Parsers transform raw documents into text chunks suitable for embedding and retrieval. LlamaFarm provides multiple parsers for each file format, allowing you to choose based on your performance and capability needs.

## Parser Configuration

Parsers are defined within `data_processing_strategies` in your `llamafarm.yaml`:

```yaml
rag:
  data_processing_strategies:
    - name: my_processor
      description: "Process various document types"
      parsers:
        - type: PDFParser_LlamaIndex
          file_include_patterns:
            - "*.pdf"
          priority: 0  # Lower = try first
          config:
            chunk_size: 1000
            chunk_overlap: 100
```

### Common Parser Properties

| Property | Required | Description |
|----------|----------|-------------|
| `type` | Yes | Parser type identifier (e.g., `PDFParser_PyPDF2`) |
| `file_include_patterns` | No | Glob patterns for files to process (e.g., `["*.pdf"]`) |
| `file_extensions` | No | File extensions this parser handles (e.g., `[".pdf"]`) |
| `priority` | No | Parser priority (lower = try first, default: 50) |
| `mime_types` | No | MIME types this parser handles |
| `fallback_parser` | No | Parser to use if this one fails |
| `config` | Yes | Parser-specific configuration |

---

## Auto Parser

The `auto` parser automatically detects file types and applies the appropriate parser.

### Configuration

```yaml
parsers:
  - type: auto
    config:
      chunk_size: 1000      # Chunk size for text splitting (100-10000, default: 1000)
      chunk_overlap: 200    # Overlap between chunks (0-500, default: 200)
```

### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-10000 | Chunk size for text splitting |
| `chunk_overlap` | integer | 200 | 0-500 | Overlap between chunks |

---

## PDF Parsers

### PDFParser_PyPDF2

Enhanced PDF parser using PyPDF2 with comprehensive text and metadata extraction.

**Best for:** Simple PDFs, form extraction, annotation extraction

```yaml
parsers:
  - type: PDFParser_PyPDF2
    file_include_patterns:
      - "*.pdf"
      - "*.PDF"
    config:
      chunk_size: 1000
      chunk_overlap: 100
      chunk_strategy: paragraphs
      extract_metadata: true
      preserve_layout: true
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Chunk size in characters |
| `chunk_overlap` | integer | 100 | 0-5000 | Overlap between chunks |
| `chunk_strategy` | string | `paragraphs` | `paragraphs`, `sentences`, `characters` | Chunking strategy |
| `extract_metadata` | boolean | `true` | - | Extract PDF metadata |
| `preserve_layout` | boolean | `true` | - | Use layout-preserving extraction |
| `extract_page_info` | boolean | `true` | - | Extract page numbers and rotation |
| `extract_annotations` | boolean | `false` | - | Extract PDF annotations |
| `extract_links` | boolean | `false` | - | Extract hyperlinks |
| `extract_form_fields` | boolean | `false` | - | Extract form fields |
| `extract_outlines` | boolean | `false` | - | Extract document outlines/bookmarks |
| `extract_images` | boolean | `false` | - | Extract embedded images |
| `extract_xmp_metadata` | boolean | `false` | - | Extract XMP metadata |
| `clean_text` | boolean | `true` | - | Clean extracted text |
| `combine_pages` | boolean | `false` | - | Combine all pages (must be `false` for chunking) |

### PDFParser_LlamaIndex

Advanced PDF parser using LlamaIndex with multiple fallback strategies and semantic chunking.

**Best for:** Complex PDFs, scanned documents, semantic chunking

```yaml
parsers:
  - type: PDFParser_LlamaIndex
    file_include_patterns:
      - "*.pdf"
    priority: 0  # Lower = try first (preferred parser)
    config:
      chunk_size: 1200
      chunk_overlap: 150
      chunk_strategy: semantic
      extract_metadata: true
      extract_tables: true
      fallback_strategies:
        - llama_pdf_reader
        - llama_pymupdf_reader
        - direct_pymupdf
        - pypdf2_fallback
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Chunk size in characters |
| `chunk_overlap` | integer | 100 | 0-5000 | Overlap between chunks |
| `chunk_strategy` | string | `sentences` | `sentences`, `paragraphs`, `pages`, `semantic` | Chunking strategy |
| `extract_metadata` | boolean | `true` | - | Extract PDF metadata |
| `extract_images` | boolean | `false` | - | Extract images from PDF |
| `extract_tables` | boolean | `true` | - | Extract tables from PDF |
| `fallback_strategies` | array | All strategies | See below | Fallback strategies in order |

**Fallback Strategies:**
- `llama_pdf_reader` - LlamaIndex PDFReader
- `llama_pymupdf_reader` - LlamaIndex PyMuPDFReader
- `direct_pymupdf` - Direct PyMuPDF
- `pypdf2_fallback` - PyPDF2 fallback

---

## CSV Parsers

### CSVParser_Pandas

Advanced CSV parser using Pandas with data analysis capabilities.

**Best for:** Data analysis, large CSV files, complex data handling

```yaml
parsers:
  - type: CSVParser_Pandas
    file_include_patterns:
      - "*.csv"
    config:
      chunk_size: 1000
      chunk_strategy: rows
      extract_metadata: true
      encoding: utf-8
      delimiter: ","
      na_values:
        - ""
        - "NA"
        - "N/A"
        - "null"
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 1000 | Number of rows per chunk |
| `chunk_strategy` | string | `rows` | `rows`, `columns`, `full` |
| `extract_metadata` | boolean | `true` | Extract data statistics |
| `encoding` | string | `utf-8` | File encoding |
| `delimiter` | string | `,` | CSV delimiter |
| `na_values` | array | `["", "NA", "N/A", "null", "None"]` | Values to treat as NaN |

### CSVParser_Python

Simple CSV parser using native Python csv module.

**Best for:** Simple CSV files, minimal dependencies

```yaml
parsers:
  - type: CSVParser_Python
    file_include_patterns:
      - "*.csv"
    config:
      chunk_size: 1000
      encoding: utf-8
      delimiter: ","
      quotechar: '"'
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 1000 | Number of rows per chunk |
| `encoding` | string | `utf-8` | File encoding |
| `delimiter` | string | `,` | CSV delimiter |
| `quotechar` | string | `"` | Quote character |

### CSVParser_LlamaIndex

CSV parser using LlamaIndex with Pandas backend for advanced processing.

**Best for:** Semantic chunking, field mapping, integration with LlamaIndex

```yaml
parsers:
  - type: CSVParser_LlamaIndex
    file_include_patterns:
      - "*.csv"
      - "*.tsv"
    config:
      chunk_size: 1000
      chunk_strategy: rows
      field_mapping:
        title: name
        content: description
      combine_fields: true
      skiprows: 0
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Number of rows per chunk |
| `chunk_strategy` | string | `rows` | `rows`, `semantic`, `full` | Chunking strategy |
| `field_mapping` | object | - | - | Map CSV columns to standard fields |
| `extract_metadata` | boolean | `true` | - | Extract metadata from CSV |
| `combine_fields` | boolean | `true` | - | Combine fields into text content |
| `skiprows` | integer | 0 | 0+ | Number of rows to skip |
| `na_values` | array | `["", "NA", "N/A", "null", "None"]` | - | Values to treat as missing |

---

## Excel Parsers

### ExcelParser_OpenPyXL

Excel parser using OpenPyXL for XLSX files with formula support.

**Best for:** XLSX files, formula extraction, workbook metadata

```yaml
parsers:
  - type: ExcelParser_OpenPyXL
    file_include_patterns:
      - "*.xlsx"
    config:
      chunk_size: 1000
      extract_formulas: false
      extract_metadata: true
      data_only: true
      sheets: null  # Process all sheets
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 1000 | Number of rows per chunk |
| `extract_formulas` | boolean | `false` | Extract cell formulas |
| `extract_metadata` | boolean | `true` | Extract workbook metadata |
| `sheets` | array/null | `null` | Specific sheets to process (null = all) |
| `data_only` | boolean | `true` | Extract values instead of formulas |

### ExcelParser_Pandas

Excel parser using Pandas with data analysis capabilities.

**Best for:** Data analysis, statistical processing

```yaml
parsers:
  - type: ExcelParser_Pandas
    file_include_patterns:
      - "*.xlsx"
      - "*.xls"
    config:
      chunk_size: 1000
      sheets: null
      extract_metadata: true
      skiprows: null
      na_values:
        - ""
        - "NA"
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 1000 | Number of rows per chunk |
| `sheets` | array/null | `null` | Specific sheets to process |
| `extract_metadata` | boolean | `true` | Extract data statistics |
| `skiprows` | integer/null | `null` | Rows to skip at beginning |
| `na_values` | array | `["", "NA", "N/A", "null", "None"]` | Values to treat as NaN |

### ExcelParser_LlamaIndex

Excel parser using LlamaIndex with Pandas backend for advanced processing.

**Best for:** Semantic chunking, combining sheets, advanced data handling

```yaml
parsers:
  - type: ExcelParser_LlamaIndex
    file_include_patterns:
      - "*.xlsx"
      - "*.xls"
    config:
      chunk_size: 1000
      chunk_strategy: rows
      sheets: null
      combine_sheets: false
      extract_metadata: true
      extract_formulas: false
      header_row: 0
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Number of rows per chunk |
| `chunk_strategy` | string | `rows` | `rows`, `semantic`, `full` | Chunking strategy |
| `sheets` | array/null | `null` | - | Specific sheets to parse |
| `combine_sheets` | boolean | `false` | - | Combine all sheets into one document |
| `extract_metadata` | boolean | `true` | - | Extract metadata |
| `extract_formulas` | boolean | `false` | - | Extract formulas instead of values |
| `header_row` | integer | 0 | 0+ | Row index for headers |
| `skiprows` | integer | 0 | 0+ | Number of rows to skip |
| `na_values` | array | `["", "NA", "N/A", "null", "None"]` | - | Values to treat as missing |

---

## Word Document Parsers

### DocxParser_PythonDocx

Word document parser using python-docx library.

**Best for:** Simple DOCX files, table extraction

```yaml
parsers:
  - type: DocxParser_PythonDocx
    file_include_patterns:
      - "*.docx"
    config:
      chunk_size: 1000
      chunk_strategy: paragraphs
      extract_metadata: true
      extract_tables: true
      extract_headers: true
      extract_footers: false
      extract_comments: false
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 1000 | Chunk size in characters |
| `chunk_strategy` | string | `paragraphs` | `paragraphs`, `sentences`, `characters` |
| `extract_metadata` | boolean | `true` | Extract document metadata |
| `extract_tables` | boolean | `true` | Extract tables |
| `extract_headers` | boolean | `true` | Extract headers |
| `extract_footers` | boolean | `false` | Extract footers |
| `extract_comments` | boolean | `false` | Extract comments |

### DocxParser_LlamaIndex

Advanced DOCX parser using LlamaIndex with enhanced chunking.

**Best for:** Semantic chunking, image extraction, complex documents

```yaml
parsers:
  - type: DocxParser_LlamaIndex
    file_include_patterns:
      - "*.docx"
    config:
      chunk_size: 1000
      chunk_overlap: 100
      chunk_strategy: paragraphs
      extract_metadata: true
      extract_tables: true
      extract_images: false
      preserve_formatting: true
      include_header_footer: false
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Chunk size in characters |
| `chunk_overlap` | integer | 100 | 0-5000 | Overlap between chunks |
| `chunk_strategy` | string | `paragraphs` | `paragraphs`, `sentences`, `semantic` | Chunking strategy |
| `extract_metadata` | boolean | `true` | - | Extract document metadata |
| `extract_tables` | boolean | `true` | - | Extract tables |
| `extract_images` | boolean | `false` | - | Extract images |
| `preserve_formatting` | boolean | `true` | - | Preserve text formatting |
| `include_header_footer` | boolean | `false` | - | Include header and footer |

---

## Markdown Parsers

### MarkdownParser_Python

Markdown parser using native Python with regex parsing.

**Best for:** Simple Markdown files, minimal dependencies

```yaml
parsers:
  - type: MarkdownParser_Python
    file_include_patterns:
      - "*.md"
      - "*.markdown"
    config:
      chunk_size: 1000
      chunk_strategy: sections
      extract_metadata: true
      extract_code_blocks: true
      extract_links: true
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 1000 | Chunk size in characters |
| `chunk_strategy` | string | `sections` | `sections`, `paragraphs`, `characters` |
| `extract_metadata` | boolean | `true` | Extract YAML frontmatter |
| `extract_code_blocks` | boolean | `true` | Extract code blocks |
| `extract_links` | boolean | `true` | Extract markdown links |

### MarkdownParser_LlamaIndex

Advanced markdown parser using LlamaIndex with semantic chunking.

**Best for:** Complex Markdown, semantic chunking, table extraction

```yaml
parsers:
  - type: MarkdownParser_LlamaIndex
    file_include_patterns:
      - "*.md"
      - "*.markdown"
    config:
      chunk_size: 800
      chunk_overlap: 80
      chunk_strategy: headings
      extract_metadata: true
      extract_code_blocks: true
      extract_tables: true
      extract_links: true
      preserve_structure: true
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Chunk size in characters |
| `chunk_overlap` | integer | 100 | 0-5000 | Overlap between chunks |
| `chunk_strategy` | string | `headings` | `headings`, `paragraphs`, `sentences`, `semantic` | Chunking strategy |
| `extract_metadata` | boolean | `true` | - | Extract frontmatter metadata |
| `extract_code_blocks` | boolean | `true` | - | Extract code blocks separately |
| `extract_tables` | boolean | `true` | - | Extract markdown tables |
| `extract_links` | boolean | `true` | - | Extract links and references |
| `preserve_structure` | boolean | `true` | - | Preserve heading hierarchy |

---

## Text Parsers

### TextParser_Python

Text parser using native Python with encoding detection.

**Best for:** Plain text files, minimal dependencies

```yaml
parsers:
  - type: TextParser_Python
    file_include_patterns:
      - "*.txt"
      - "*.log"
    config:
      chunk_size: 1000
      chunk_overlap: 100
      chunk_strategy: sentences
      encoding: utf-8
      clean_text: true
      extract_metadata: true
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `chunk_size` | integer | 1000 | Chunk size in characters |
| `chunk_overlap` | integer | 100 | Overlap between chunks |
| `chunk_strategy` | string | `sentences` | `sentences`, `paragraphs`, `characters` |
| `encoding` | string | `utf-8` | Text encoding (or auto-detect) |
| `clean_text` | boolean | `true` | Remove excessive whitespace |
| `extract_metadata` | boolean | `true` | Extract file statistics |

### TextParser_LlamaIndex

Advanced text parser using LlamaIndex with semantic splitting and code parsing.

**Best for:** Semantic chunking, code files, multi-format support

```yaml
parsers:
  - type: TextParser_LlamaIndex
    file_include_patterns:
      - "*.txt"
      - "*.py"
      - "*.js"
    config:
      chunk_size: 800
      chunk_overlap: 80
      chunk_strategy: semantic
      encoding: utf-8
      clean_text: true
      extract_metadata: true
      preserve_code_structure: true
      detect_language: true
      include_prev_next_rel: true
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Chunk size in characters |
| `chunk_overlap` | integer | 100 | 0-5000 | Overlap between chunks |
| `chunk_strategy` | string | `semantic` | `characters`, `sentences`, `paragraphs`, `tokens`, `semantic`, `code` | Chunking strategy |
| `encoding` | string | `utf-8` | - | Text encoding |
| `clean_text` | boolean | `true` | - | Clean extracted text |
| `extract_metadata` | boolean | `true` | - | Extract comprehensive metadata |
| `semantic_buffer_size` | integer | 1 | 1-10 | Buffer size for semantic chunking |
| `semantic_breakpoint_percentile_threshold` | integer | 95 | 50-99 | Percentile threshold for breakpoints |
| `token_model` | string | `gpt-3.5-turbo` | - | Tokenizer model for token chunking |
| `preserve_code_structure` | boolean | `true` | - | Preserve code syntax and structure |
| `detect_language` | boolean | `true` | - | Auto-detect programming language |
| `include_prev_next_rel` | boolean | `true` | - | Include relationships between chunks |

---

## Email Parser

### MsgParser_ExtractMsg

Outlook MSG file parser using extract-msg library.

**Best for:** Outlook emails, attachment extraction

```yaml
parsers:
  - type: MsgParser_ExtractMsg
    file_include_patterns:
      - "*.msg"
    config:
      chunk_size: 1000
      chunk_overlap: 100
      chunk_strategy: email_sections
      extract_metadata: true
      extract_attachments: true
      extract_headers: true
      include_attachment_content: true
      clean_text: true
      preserve_formatting: false
      encoding: utf-8
```

#### Options

| Option | Type | Default | Range | Description |
|--------|------|---------|-------|-------------|
| `chunk_size` | integer | 1000 | 100-50000 | Chunk size in characters |
| `chunk_overlap` | integer | 100 | 0-5000 | Overlap between chunks |
| `chunk_strategy` | string | `email_sections` | `sentences`, `paragraphs`, `characters`, `email_sections` | Chunking strategy |
| `extract_metadata` | boolean | `true` | - | Extract email metadata |
| `extract_attachments` | boolean | `true` | - | Extract attachments |
| `extract_headers` | boolean | `true` | - | Extract email headers |
| `include_attachment_content` | boolean | `true` | - | Include attachment content |
| `clean_text` | boolean | `true` | - | Clean text |
| `preserve_formatting` | boolean | `false` | - | Preserve HTML formatting |
| `encoding` | string | `utf-8` | - | Text encoding |

---

## Directory Parser

### DirectoryParser

Recursively parses all files in a directory, automatically detecting file types.

**Best for:** Batch processing, directory ingestion, multi-format processing

```yaml
parsers:
  - type: DirectoryParser
    file_include_patterns:
      - "data/**/*"
    config:
      recursive: true
      exclude_patterns:
        - "*.tmp"
        - "*.bak"
      max_depth: 5
```

#### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `recursive` | boolean | `true` | Recursively process subdirectories |
| `exclude_patterns` | array | `[]` | Glob patterns to exclude |
| `max_depth` | integer | 10 | Maximum directory depth |
| `follow_symlinks` | boolean | `false` | Follow symbolic links |

---

## Complete Example

Here's a comprehensive example processing multiple file types:

```yaml
rag:
  data_processing_strategies:
    - name: universal_processor
      description: "Handles multiple document formats"
      parsers:
        # PDF files with LlamaIndex (try first - lower priority value)
        - type: PDFParser_LlamaIndex
          file_include_patterns:
            - "*.pdf"
            - "*.PDF"
          priority: 0
          config:
            chunk_size: 1200
            chunk_overlap: 150
            chunk_strategy: semantic
            extract_metadata: true
            extract_tables: true

        # PDF fallback with PyPDF2 (try second - higher priority value)
        - type: PDFParser_PyPDF2
          file_include_patterns:
            - "*.pdf"
          priority: 50
          fallback_parser: null
          config:
            chunk_size: 1000
            chunk_overlap: 100
            chunk_strategy: paragraphs

        # Markdown files
        - type: MarkdownParser_LlamaIndex
          file_include_patterns:
            - "*.md"
            - "*.markdown"
          priority: 0
          config:
            chunk_size: 800
            chunk_overlap: 80
            chunk_strategy: headings
            extract_code_blocks: true

        # CSV files
        - type: CSVParser_Pandas
          file_include_patterns:
            - "*.csv"
          priority: 0
          config:
            chunk_size: 500
            chunk_strategy: rows
            extract_metadata: true

        # Excel files
        - type: ExcelParser_LlamaIndex
          file_include_patterns:
            - "*.xlsx"
            - "*.xls"
          priority: 0
          config:
            chunk_size: 500
            chunk_strategy: rows
            combine_sheets: false

        # Word documents
        - type: DocxParser_LlamaIndex
          file_include_patterns:
            - "*.docx"
          priority: 0
          config:
            chunk_size: 1000
            chunk_overlap: 100
            chunk_strategy: paragraphs
            extract_tables: true

        # Plain text and code
        - type: TextParser_LlamaIndex
          file_include_patterns:
            - "*.txt"
            - "*.py"
            - "*.js"
            - "*.html"
          priority: 0
          config:
            chunk_size: 800
            chunk_overlap: 80
            chunk_strategy: semantic
            preserve_code_structure: true

        # Outlook emails
        - type: MsgParser_ExtractMsg
          file_include_patterns:
            - "*.msg"
          priority: 0
          config:
            chunk_strategy: email_sections
            extract_attachments: true
```

## Chunking Strategy Guidelines

| Strategy | Best For | Considerations |
|----------|----------|----------------|
| `sentences` | General text, documentation | Good balance of granularity |
| `paragraphs` | Articles, reports | Preserves natural breaks |
| `characters` | Fixed-size needs | Predictable chunk sizes |
| `semantic` | Technical docs, varied content | Content-aware splitting |
| `sections` / `headings` | Markdown, structured docs | Respects document structure |
| `pages` | PDFs, page-based docs | One chunk per page |
| `rows` | CSV, Excel | Data-oriented chunking |
| `code` | Source code files | Preserves syntax |
| `email_sections` | Emails | Headers/body/signature |

## Next Steps

- [Embedders Reference](./embedders.md) - Configure embedding strategies
- [Extractors Reference](./extractors.md) - Add metadata extraction
- [RAG Guide](./index.md) - Full RAG configuration overview
