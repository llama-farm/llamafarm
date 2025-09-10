# Parser System Documentation

Comprehensive guide to the RAG system's parser architecture, including DirectoryParser, file type parsers, and configuration options.

## 🎯 Parser Architecture Overview

The parser system follows a two-tier architecture:

1. **DirectoryParser** (Always Active) - Strategy-level file detection and routing
2. **File Type Parsers** - Specialized parsers for different document types

## 🚀 DirectoryParser: The Gateway

DirectoryParser is **ALWAYS ACTIVE** at the strategy level and handles:
- File/directory scanning
- MIME type detection
- File filtering based on strategy rules
- Routing files to appropriate parsers

### How DirectoryParser Works

```yaml
data_processing_strategies:
  - name: "my_strategy"
    # DirectoryParser configuration (ALWAYS ACTIVE)
    directory_config:
      recursive: true
      supported_files: ["*.pdf", "*.txt"]  # Glob patterns for accepted files
      exclude_patterns: ["*.tmp", ".*"]     # Files to exclude
    parsers:
      # Individual file parsers
```

DirectoryParser automatically:
1. Scans the input path (file or directory)
2. Filters files based on `supported_files` glob patterns in `directory_config`
3. Routes each file to the appropriate parser based on mime_types/file_extensions
4. Handles both single files and directories seamlessly

## 📋 Parser Naming Convention

All parsers follow the standardized naming pattern:

```
{ParserType}_{Implementation}
```

Examples:
- `PDFParser_LlamaIndex` - PDF parser using LlamaIndex
- `TextParser_Python` - Text parser using pure Python
- `CSVParser_Pandas` - CSV parser using Pandas

## 🔧 Available Parsers

### PDF Parsers

#### PDFParser_LlamaIndex
Advanced PDF parsing using LlamaIndex's capabilities.

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
    extract_outline: true
    include_page_numbers: true
```

**Features:**
- Semantic chunking for intelligent text splitting
- Equation preservation for technical documents
- Page-level metadata extraction
- Document outline extraction

#### PDFParser_PyPDF2
Robust PDF parsing using PyPDF2 library.

```yaml
- type: "PDFParser_PyPDF2"
  mime_types: ["application/pdf"]
  file_extensions: [".pdf"]
  config:
    chunk_size: 1000
    chunk_overlap: 150
    chunk_strategy: "paragraphs"  # paragraphs, pages, fixed
    extract_metadata: true
    combine_pages: true
    page_separator: "\n\n---\n\n"
    min_text_length: 50
```

**Features:**
- Paragraph-aware chunking
- Page combination options
- Metadata extraction
- Text quality filtering

### Text Parsers

#### TextParser_LlamaIndex
LlamaIndex-based text parsing with advanced features.

```yaml
- type: "TextParser_LlamaIndex"
  mime_types: ["text/plain"]
  file_extensions: [".txt", ".text", ".log"]
  config:
    chunk_size: 1000
    chunk_overlap: 150
    extract_metadata: true
```

#### TextParser_Python
Pure Python text parser with full control.

```yaml
- type: "TextParser_Python"
  mime_types: ["text/plain"]
  file_extensions: [".txt", ".log", ".text"]
  config:
    encoding: "utf-8"
    chunk_size: 1200
    chunk_overlap: 200
    chunk_strategy: "sentences"  # sentences, paragraphs, words, characters
    respect_sentence_boundaries: true
    respect_paragraph_boundaries: true
    min_chunk_size: 100
    clean_text: true
    preserve_line_breaks: true
    strip_empty_lines: true
    detect_structure: true
```

**Features:**
- Multiple chunking strategies
- Boundary-aware splitting
- Text cleaning options
- Structure detection

### CSV Parsers

#### CSVParser_LlamaIndex
LlamaIndex CSV parser for structured data.

```yaml
- type: "CSVParser_LlamaIndex"
  mime_types: ["text/csv", "text/tab-separated-values"]
  file_extensions: [".csv", ".tsv"]
  config:
    chunk_size: 500
    chunk_overlap: 50
    extract_metadata: true
```

#### CSVParser_Pandas
Pandas-based CSV parser with advanced features.

```yaml
- type: "CSVParser_Pandas"
  mime_types: ["text/csv"]
  file_extensions: [".csv", ".CSV"]
  config:
    content_fields: ["description", "content", "text"]
    metadata_fields: ["id", "date", "category", "author"]
    chunk_size: 500
    chunk_strategy: "rows"  # rows, columns
    delimiter: ","
    encoding: "utf-8"
    priority_mapping:
      "High": 1
      "Medium": 2
      "Low": 3
```

**Features:**
- Field mapping for content/metadata
- Custom delimiters
- Priority mapping
- Row/column chunking

#### CSVParser_Python
Lightweight Python CSV parser.

```yaml
- type: "CSVParser_Python"
  mime_types: ["text/csv"]
  file_extensions: [".csv"]
  config:
    delimiter: ","
    encoding: "utf-8"
    has_header: true
    chunk_size: 500
```

### Document Parsers

#### DocxParser_LlamaIndex
Word document parser using LlamaIndex.

```yaml
- type: "DocxParser_LlamaIndex"
  mime_types: ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
  file_extensions: [".docx", ".DOCX"]
  config:
    chunk_size: 1500
    chunk_overlap: 200
    extract_metadata: true
    extract_tables: true
    chunk_strategy: "sections"  # sections, paragraphs, pages
```

#### DocxParser_PythonDocx
Python-docx based Word parser.

```yaml
- type: "DocxParser_PythonDocx"
  mime_types: ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
  file_extensions: [".docx"]
  config:
    chunk_size: 1600
    chunk_overlap: 200
    extract_metadata: true
    extract_tables: true
    extract_comments: false
    extract_headers_footers: true
```

### Excel Parsers

#### ExcelParser_LlamaIndex
Excel parser using LlamaIndex.

```yaml
- type: "ExcelParser_LlamaIndex"
  mime_types: ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
  file_extensions: [".xlsx", ".XLSX"]
  config:
    chunk_size: 500
    extract_metadata: true
```

#### ExcelParser_Pandas
Pandas-based Excel parser.

```yaml
- type: "ExcelParser_Pandas"
  mime_types: ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
  file_extensions: [".xlsx", ".xls"]
  config:
    sheet_name: null  # null for all sheets
    header_row: 0
    chunk_size: 500
    extract_formulas: false
```

#### ExcelParser_OpenPyXL
OpenPyXL-based Excel parser.

```yaml
- type: "ExcelParser_OpenPyXL"
  mime_types: ["application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]
  file_extensions: [".xlsx"]
  config:
    data_only: true  # Get values, not formulas
    read_only: true  # Memory efficient
    chunk_size: 500
```

### Markdown Parsers

#### MarkdownParser_LlamaIndex
Markdown parser with structure preservation.

```yaml
- type: "MarkdownParser_LlamaIndex"
  mime_types: ["text/markdown", "text/x-markdown"]
  file_extensions: [".md", ".markdown", ".mdown"]
  config:
    chunk_size: 1200
    chunk_overlap: 150
    extract_metadata: true
```

#### MarkdownParser_Python
Python markdown parser with advanced features.

```yaml
- type: "MarkdownParser_Python"
  mime_types: ["text/markdown"]
  file_extensions: [".md", ".markdown"]
  config:
    chunk_size: 1000
    chunk_strategy: "headings"  # headings, sections, paragraphs
    extract_metadata: true
    extract_code_blocks: true
    extract_links: true
    extract_headings: true
    preserve_formatting: false
    min_heading_level: 1
    max_heading_level: 6
```

## 🎨 Parser Configuration Patterns

### Multi-Format Strategy
Handle multiple file types in one strategy:

```yaml
data_processing_strategies:
  - name: "multi_format"
    directory_config:
      recursive: true
      supported_files: ["*"]  # Accept all files
      exclude_patterns: ["*.tmp", ".*"]
    parsers:
      - type: "PDFParser_LlamaIndex"
        mime_types: ["application/pdf"]
        file_extensions: [".pdf"]
        config:
          chunk_size: 1500
      - type: "DocxParser_LlamaIndex"
        mime_types: ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
        file_extensions: [".docx"]
        config:
          chunk_size: 1500
      - type: "TextParser_Python"
        mime_types: ["text/plain"]
        file_extensions: [".txt", ".log"]
        config:
          chunk_size: 1200
```

### Specialized Strategy
Optimized for specific file type:

```yaml
data_processing_strategies:
  - name: "pdf_only"
    directory_config:
      recursive: true
      supported_files: ["*.pdf", "*.PDF"]  # PDF files only
      exclude_patterns: ["*.tmp"]
    parsers:
      - type: "PDFParser_LlamaIndex"
        mime_types: ["application/pdf"]
        file_extensions: [".pdf"]
        config:
          chunk_strategy: "semantic"
          preserve_equations: true
          extract_images: true
```

## 🔄 Parser Selection Logic

1. **DirectoryParser** filters files based on `supported_files` glob patterns in `directory_config`
2. For each accepted file:
   - Check MIME type against parser `mime_types`
   - Check extension against parser `file_extensions`
   - Select first matching parser
3. If no parser matches, file is skipped

## 📊 Chunking Strategies

Different chunking strategies for different use cases:

| Strategy | Best For | Description |
|----------|----------|-------------|
| `sentences` | General text | Splits on sentence boundaries |
| `paragraphs` | Documents | Splits on paragraph boundaries |
| `headings` | Structured docs | Splits on heading levels |
| `sections` | Technical docs | Splits on document sections |
| `pages` | PDFs | Splits on page boundaries |
| `semantic` | Advanced | AI-powered semantic splitting |
| `fixed` | Simple | Fixed character count |
| `rows` | Tabular data | Splits on row count |

## 🛠️ Creating Custom Parsers

To create a custom parser:

1. Inherit from base parser class
2. Follow naming convention: `{Type}Parser_{Implementation}`
3. Implement required methods:
   - `parse(file_path) -> List[Document]`
   - `can_parse(file_path) -> bool`
4. Register in parser factory

Example:
```python
class JSONParser_Custom:
    def __init__(self, config):
        self.config = config
    
    def parse(self, file_path):
        # Custom parsing logic
        return documents
```

## 🔍 Troubleshooting

### File Not Being Processed

Check:
1. File matches a pattern in `supported_files` glob patterns
2. File doesn't match any `exclude_patterns`
3. A parser exists with matching `mime_types` or `file_extensions`

### Wrong Parser Selected

Ensure:
1. Parser order in strategy (first match wins)
2. MIME types and extensions are specific
3. No overlapping parser definitions

### Chunking Issues

Verify:
1. Chunk size is appropriate for content
2. Chunk overlap maintains context
3. Chunking strategy matches document structure

## 📚 See Also

- [Schema Documentation](SCHEMA.md) - Complete schema reference
- [Strategy System](STRATEGY_SYSTEM.md) - How strategies work
- [Component Guide](COMPONENTS.md) - All system components
- [config/templates/](../config/templates/) - Example configurations