/**
 * AUTO-GENERATED FILE - DO NOT EDIT
 * 
 * Generated from rag/schema.yaml by generate-ui-types.py
 * Run: cd rag && ./generate-ui-types.sh
 */

// ============================================================================
// Parser Types
// ============================================================================

export const PARSER_TYPES = [
  "CSVParser_LlamaIndex",
  "CSVParser_Pandas",
  "CSVParser_Python",
  "DocxParser_LlamaIndex",
  "DocxParser_PythonDocx",
  "ExcelParser_LlamaIndex",
  "ExcelParser_OpenPyXL",
  "ExcelParser_Pandas",
  "MSGParser_ExtractMsg",
  "MarkdownParser_LlamaIndex",
  "MarkdownParser_Python",
  "PDFParser_LlamaIndex",
  "PDFParser_PyPDF2",
  "TextParser_LlamaIndex",
  "TextParser_Python",
  "auto",
] as const

export type ParserType = typeof PARSER_TYPES[number]

// ============================================================================
// Extractor Types
// ============================================================================

export const EXTRACTOR_TYPES = [
  "ContentStatisticsExtractor",
  "DateTimeExtractor",
  "EntityExtractor",
  "HeadingExtractor",
  "KeywordExtractor",
  "LinkExtractor",
  "PathExtractor",
  "PatternExtractor",
  "RAKEExtractor",
  "SummaryExtractor",
  "TFIDFExtractor",
  "TableExtractor",
  "YAKEExtractor",
] as const

export type ExtractorType = typeof EXTRACTOR_TYPES[number]

// ============================================================================
// Default Configurations
// ============================================================================

export function getDefaultParserConfig(parserType: ParserType): Record<string, any> {
  const configs: Record<ParserType, Record<string, any>> = {
    "CSVParser_LlamaIndex":       {
          "chunk_size": 1000,
          "chunk_strategy": "rows",
          "extract_metadata": true,
          "combine_fields": true,
          "na_values": [
              "",
              "NA",
              "N/A",
              "null",
              "None"
          ]
      },
    "CSVParser_Pandas":       {
          "chunk_size": 1000,
          "chunk_strategy": "rows",
          "extract_metadata": true,
          "encoding": "utf-8",
          "delimiter": ",",
          "na_values": [
              "",
              "NA",
              "N/A",
              "null",
              "None"
          ]
      },
    "CSVParser_Python":       {
          "chunk_size": 1000,
          "encoding": "utf-8",
          "delimiter": ",",
          "quotechar": "\""
      },
    "DocxParser_LlamaIndex":       {
          "chunk_size": 1000,
          "chunk_overlap": 100,
          "chunk_strategy": "paragraphs",
          "extract_metadata": true,
          "extract_tables": true,
          "extract_images": false,
          "preserve_formatting": true,
          "include_header_footer": false
      },
    "DocxParser_PythonDocx":       {
          "chunk_size": 1000,
          "chunk_strategy": "paragraphs",
          "extract_metadata": true,
          "extract_tables": true,
          "extract_headers": true,
          "extract_footers": false,
          "extract_comments": false
      },
    "ExcelParser_LlamaIndex":       {
          "chunk_size": 1000,
          "chunk_strategy": "rows",
          "combine_sheets": false,
          "extract_metadata": true,
          "extract_formulas": false,
          "header_row": 0,
          "na_values": [
              "",
              "NA",
              "N/A",
              "null",
              "None"
          ]
      },
    "ExcelParser_OpenPyXL":       {
          "chunk_size": 1000,
          "extract_formulas": false,
          "extract_metadata": true,
          "sheets": null,
          "data_only": true
      },
    "ExcelParser_Pandas":       {
          "chunk_size": 1000,
          "sheets": null,
          "extract_metadata": true,
          "skiprows": null,
          "na_values": [
              "",
              "NA",
              "N/A",
              "null",
              "None"
          ]
      },
    "MSGParser_ExtractMsg":       {
          "chunk_size": 1000,
          "chunk_overlap": 100,
          "chunk_strategy": "email_sections",
          "extract_metadata": true,
          "extract_attachments": true,
          "extract_headers": true,
          "include_attachment_content": true,
          "clean_text": true,
          "preserve_formatting": false,
          "encoding": "utf-8"
      },
    "MarkdownParser_LlamaIndex":       {
          "chunk_size": 1000,
          "chunk_overlap": 100,
          "chunk_strategy": "headings",
          "extract_metadata": true,
          "extract_code_blocks": true,
          "extract_tables": true,
          "extract_links": true,
          "preserve_structure": true
      },
    "MarkdownParser_Python":       {
          "chunk_size": 1000,
          "chunk_strategy": "sections",
          "extract_metadata": true,
          "extract_code_blocks": true,
          "extract_links": true
      },
    "PDFParser_LlamaIndex":       {
          "chunk_size": 1000,
          "chunk_overlap": 100,
          "chunk_strategy": "sentences",
          "extract_metadata": true,
          "extract_images": false,
          "extract_tables": true,
          "fallback_strategies": [
              "llama_pdf_reader",
              "llama_pymupdf_reader",
              "direct_pymupdf",
              "pypdf2_fallback"
          ]
      },
    "PDFParser_PyPDF2":       {
          "chunk_size": 1000,
          "chunk_overlap": 100,
          "chunk_strategy": "paragraphs",
          "extract_metadata": true,
          "preserve_layout": true,
          "extract_page_info": true,
          "extract_annotations": false,
          "extract_links": false,
          "extract_form_fields": false,
          "extract_outlines": false,
          "extract_images": false,
          "extract_xmp_metadata": false,
          "clean_text": true,
          "combine_pages": false
      },
    "TextParser_LlamaIndex":       {
          "chunk_size": 1000,
          "chunk_overlap": 100,
          "chunk_strategy": "semantic",
          "encoding": "utf-8",
          "clean_text": true,
          "extract_metadata": true,
          "semantic_buffer_size": 1,
          "semantic_breakpoint_percentile_threshold": 95,
          "token_model": "gpt-3.5-turbo",
          "preserve_code_structure": true,
          "detect_language": true,
          "include_prev_next_rel": true
      },
    "TextParser_Python":       {
          "chunk_size": 1000,
          "chunk_overlap": 100,
          "chunk_strategy": "sentences",
          "encoding": "utf-8",
          "clean_text": true,
          "extract_metadata": true
      },
    "auto":       {
          "chunk_size": 1000,
          "chunk_overlap": 200
      },
  }
  return configs[parserType] || {}
}

export function getDefaultExtractorConfig(extractorType: ExtractorType): Record<string, any> {
  const configs: Record<ExtractorType, Record<string, any>> = {
    "ContentStatisticsExtractor":       {
          "include_readability": true,
          "include_vocabulary": true,
          "include_structure": true,
          "include_sentiment_indicators": false
      },
    "DateTimeExtractor":       {
          "fuzzy_parsing": true,
          "extract_relative": true,
          "extract_times": true,
          "extract_durations": true,
          "default_timezone": "UTC",
          "date_format": "ISO",
          "prefer_dates_from": "current"
      },
    "EntityExtractor":       {
          "model": "en_core_web_sm",
          "entity_types": [
              "PERSON",
              "ORG",
              "GPE",
              "DATE",
              "TIME",
              "MONEY",
              "EMAIL",
              "PHONE",
              "URL",
              "PERCENT",
              "PRODUCT",
              "EVENT"
          ],
          "use_fallback": true,
          "min_entity_length": 2,
          "merge_entities": true,
          "confidence_threshold": 0.7
      },
    "HeadingExtractor":       {
          "max_level": 6,
          "include_hierarchy": true,
          "extract_outline": true,
          "min_heading_length": 3,
          "enabled": true
      },
    "KeywordExtractor":       {
          "algorithm": "rake",
          "max_keywords": 10,
          "min_length": 1,
          "max_length": 4,
          "min_frequency": 1,
          "language": "en",
          "max_ngram_size": 3,
          "deduplication_threshold": 0.9
      },
    "LinkExtractor":       {
          "extract_urls": true,
          "extract_emails": true,
          "extract_domains": true,
          "validate_urls": false,
          "resolve_redirects": false,
          "enabled": true
      },
    "PathExtractor":       {
          "extract_file_paths": true,
          "extract_urls": true,
          "extract_s3_paths": true,
          "validate_paths": false,
          "normalize_paths": true,
          "enabled": true
      },
    "PatternExtractor":       {
          "predefined_patterns": [],
          "custom_patterns": [],
          "case_sensitive": false,
          "return_positions": false,
          "include_context": false,
          "max_matches_per_pattern": 100,
          "deduplicate_matches": true
      },
    "RAKEExtractor":       {
          "algorithm": "rake",
          "max_keywords": 10,
          "min_length": 1,
          "max_length": 4,
          "min_frequency": 1,
          "language": "en",
          "max_ngram_size": 3,
          "deduplication_threshold": 0.9
      },
    "SummaryExtractor":       {
          "summary_sentences": 3,
          "algorithm": "textrank",
          "include_key_phrases": true,
          "include_statistics": true,
          "min_sentence_length": 10,
          "max_sentence_length": 500
      },
    "TFIDFExtractor":       {
          "algorithm": "rake",
          "max_keywords": 10,
          "min_length": 1,
          "max_length": 4,
          "min_frequency": 1,
          "language": "en",
          "max_ngram_size": 3,
          "deduplication_threshold": 0.9
      },
    "TableExtractor":       {
          "output_format": "dict",
          "extract_headers": true,
          "merge_cells": true,
          "min_rows": 2,
          "enabled": true
      },
    "YAKEExtractor":       {
          "max_keywords": 10,
          "language": "en",
          "max_ngram_size": 3,
          "deduplication_threshold": 0.9
      },
  }
  return configs[extractorType] || {}
}

// ============================================================================
// Schema Metadata
// ============================================================================

export type PrimitiveType = 'integer' | 'number' | 'string' | 'boolean' | 'array'

export type SchemaField = {
  type: PrimitiveType
  title?: string
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  enum?: string[]
  items?: { type: PrimitiveType }
  nullable?: boolean
}

export interface ParserSchema {
  type: ParserType
  title: string
  description: string
  defaultExtensions: string[]
  properties: Record<string, SchemaField>
  required?: string[]
}

export interface ExtractorSchema {
  type: ExtractorType
  title: string
  description: string
  properties: Record<string, SchemaField>
  required?: string[]
}

export const PARSER_SCHEMAS: Record<ParserType, ParserSchema> = {
  "CSVParser_LlamaIndex": {
    type: "CSVParser_LlamaIndex",
    title: "CSV Parser (LlamaIndex) Configuration",
    description: "CSV parser using LlamaIndex with Pandas backend for advanced processing",
    defaultExtensions: [".csv"],
    properties: {
      chunk_size: { type: "integer", description: "Number of rows per chunk", default: 1000, minimum: 100, maximum: 50000 },
      chunk_strategy: { type: "string", description: "Chunking strategy", default: "rows", enum: ["rows", "semantic", "full"] },
      field_mapping: { type: "string", description: "Map CSV columns to standard fields" },
      extract_metadata: { type: "boolean", description: "Extract metadata from CSV", default: true },
      combine_fields: { type: "boolean", description: "Combine fields into text content", default: true },
      skiprows: { type: "integer", description: "Number of rows to skip at beginning", minimum: 0 },
      na_values: { type: "array", description: "Values to treat as missing", default: ["", "NA", "N/A", "null", "None"], items: { type: "string" } }
    },
  },
  "CSVParser_Pandas": {
    type: "CSVParser_Pandas",
    title: "CSV Parser (Pandas) Configuration",
    description: "Advanced CSV parser using Pandas with data analysis capabilities",
    defaultExtensions: [".csv"],
    properties: {
      chunk_size: { type: "integer", description: "Number of rows per chunk", default: 1000, minimum: 100 },
      chunk_strategy: { type: "string", description: "How to chunk the CSV data", default: "rows", enum: ["rows", "columns", "full"] },
      extract_metadata: { type: "boolean", description: "Extract data statistics and metadata", default: true },
      encoding: { type: "string", description: "File encoding", default: "utf-8" },
      delimiter: { type: "string", description: "CSV delimiter", default: "," },
      na_values: { type: "array", description: "Values to treat as NaN", default: ["", "NA", "N/A", "null", "None"], items: { type: "string" } }
    },
  },
  "CSVParser_Python": {
    type: "CSVParser_Python",
    title: "CSV Parser (Python) Configuration",
    description: "Simple CSV parser using native Python csv module",
    defaultExtensions: [".csv"],
    properties: {
      chunk_size: { type: "integer", description: "Number of rows per chunk", default: 1000, minimum: 100 },
      encoding: { type: "string", description: "File encoding", default: "utf-8" },
      delimiter: { type: "string", description: "CSV delimiter", default: "," },
      quotechar: { type: "string", description: "Quote character", default: "\"" }
    },
  },
  "DocxParser_LlamaIndex": {
    type: "DocxParser_LlamaIndex",
    title: "DOCX Parser (LlamaIndex) Configuration",
    description: "Advanced DOCX parser using LlamaIndex with enhanced chunking",
    defaultExtensions: [".docx"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100, maximum: 50000 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks", default: 100, minimum: 0, maximum: 5000 },
      chunk_strategy: { type: "string", description: "Chunking strategy", default: "paragraphs", enum: ["paragraphs", "sentences", "semantic"] },
      extract_metadata: { type: "boolean", description: "Extract document metadata", default: true },
      extract_tables: { type: "boolean", description: "Extract tables from document", default: true },
      extract_images: { type: "boolean", description: "Extract images from document", default: false },
      preserve_formatting: { type: "boolean", description: "Preserve text formatting", default: true },
      include_header_footer: { type: "boolean", description: "Include header and footer content", default: false }
    },
  },
  "DocxParser_PythonDocx": {
    type: "DocxParser_PythonDocx",
    title: "DOCX Parser (python-docx) Configuration",
    description: "Word document parser using python-docx library",
    defaultExtensions: [".docx"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100 },
      chunk_strategy: { type: "string", description: "Chunking strategy", default: "paragraphs", enum: ["paragraphs", "sentences", "characters"] },
      extract_metadata: { type: "boolean", description: "Extract document metadata", default: true },
      extract_tables: { type: "boolean", description: "Extract tables using python-docx", default: true },
      extract_headers: { type: "boolean", description: "Extract headers", default: true },
      extract_footers: { type: "boolean", description: "Extract footers", default: false },
      extract_comments: { type: "boolean", description: "Extract comments", default: false }
    },
  },
  "ExcelParser_LlamaIndex": {
    type: "ExcelParser_LlamaIndex",
    title: "Excel Parser (LlamaIndex) Configuration",
    description: "Excel parser using LlamaIndex with Pandas backend for advanced processing",
    defaultExtensions: [".xlsx", ".xls"],
    properties: {
      chunk_size: { type: "integer", description: "Number of rows per chunk", default: 1000, minimum: 100, maximum: 50000 },
      chunk_strategy: { type: "string", description: "Chunking strategy", default: "rows", enum: ["rows", "semantic", "full"] },
      sheets: { type: "string", description: "Specific sheets to parse (null for all)" },
      combine_sheets: { type: "boolean", description: "Combine all sheets into one document", default: false },
      extract_metadata: { type: "boolean", description: "Extract metadata from Excel", default: true },
      extract_formulas: { type: "boolean", description: "Extract formulas instead of values", default: false },
      header_row: { type: "integer", description: "Row index for headers", default: 0, minimum: 0 },
      skiprows: { type: "integer", description: "Number of rows to skip", minimum: 0 },
      na_values: { type: "array", description: "Values to treat as missing", default: ["", "NA", "N/A", "null", "None"], items: { type: "string" } }
    },
  },
  "ExcelParser_OpenPyXL": {
    type: "ExcelParser_OpenPyXL",
    title: "Excel Parser (OpenPyXL) Configuration",
    description: "Excel parser using OpenPyXL for XLSX files with formula support",
    defaultExtensions: [".xlsx", ".xls"],
    properties: {
      chunk_size: { type: "integer", description: "Number of rows per chunk", default: 1000, minimum: 100 },
      extract_formulas: { type: "boolean", description: "Extract cell formulas using OpenPyXL", default: false },
      extract_metadata: { type: "boolean", description: "Extract workbook metadata", default: true },
      sheets: { type: "string", description: "Specific sheets to process (null = all)", default: null },
      data_only: { type: "boolean", description: "Extract values instead of formulas", default: true }
    },
  },
  "ExcelParser_Pandas": {
    type: "ExcelParser_Pandas",
    title: "Excel Parser (Pandas) Configuration",
    description: "Excel parser using Pandas with data analysis capabilities",
    defaultExtensions: [".xlsx", ".xls"],
    properties: {
      chunk_size: { type: "integer", description: "Number of rows per chunk", default: 1000, minimum: 100 },
      sheets: { type: "string", description: "Specific sheets to process (null = all)", default: null },
      extract_metadata: { type: "boolean", description: "Extract data statistics", default: true },
      skiprows: { type: "string", description: "Rows to skip at beginning", default: null },
      na_values: { type: "array", description: "Values to treat as NaN", default: ["", "NA", "N/A", "null", "None"], items: { type: "string" } }
    },
  },
  "MSGParser_ExtractMsg": {
    type: "MSGParser_ExtractMsg",
    title: "MSG Parser (extract-msg) Configuration",
    description: "",
    defaultExtensions: [".msg"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100, maximum: 50000 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks", default: 100, minimum: 0, maximum: 5000 },
      chunk_strategy: { type: "string", description: "Chunking strategy", default: "email_sections", enum: ["sentences", "paragraphs", "characters", "email_sections"] },
      extract_metadata: { type: "boolean", description: "Extract metadata", default: true },
      extract_attachments: { type: "boolean", description: "Extract attachments", default: true },
      extract_headers: { type: "boolean", description: "Extract headers", default: true },
      include_attachment_content: { type: "boolean", description: "Include attachment content", default: true },
      clean_text: { type: "boolean", description: "Clean text", default: true },
      preserve_formatting: { type: "boolean", description: "Preserve formatting", default: false },
      encoding: { type: "string", description: "Encoding", default: "utf-8" }
    },
  },
  "MarkdownParser_LlamaIndex": {
    type: "MarkdownParser_LlamaIndex",
    title: "Markdown Parser (LlamaIndex) Configuration",
    description: "Advanced markdown parser using LlamaIndex with semantic chunking",
    defaultExtensions: [".md", ".markdown"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100, maximum: 50000 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks", default: 100, minimum: 0, maximum: 5000 },
      chunk_strategy: { type: "string", description: "Chunking strategy for markdown", default: "headings", enum: ["headings", "paragraphs", "sentences", "semantic"] },
      extract_metadata: { type: "boolean", description: "Extract frontmatter metadata", default: true },
      extract_code_blocks: { type: "boolean", description: "Extract code blocks separately", default: true },
      extract_tables: { type: "boolean", description: "Extract markdown tables", default: true },
      extract_links: { type: "boolean", description: "Extract links and references", default: true },
      preserve_structure: { type: "boolean", description: "Preserve heading hierarchy", default: true }
    },
  },
  "MarkdownParser_Python": {
    type: "MarkdownParser_Python",
    title: "Markdown Parser (Python) Configuration",
    description: "Markdown parser using native Python with regex parsing",
    defaultExtensions: [".md", ".markdown"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100 },
      chunk_strategy: { type: "string", description: "Chunking strategy - sections uses markdown headers", default: "sections", enum: ["sections", "paragraphs", "characters"] },
      extract_metadata: { type: "boolean", description: "Extract YAML frontmatter", default: true },
      extract_code_blocks: { type: "boolean", description: "Extract code blocks", default: true },
      extract_links: { type: "boolean", description: "Extract markdown links", default: true }
    },
  },
  "PDFParser_LlamaIndex": {
    type: "PDFParser_LlamaIndex",
    title: "PDF Parser (LlamaIndex) Configuration",
    description: "Advanced PDF parser using LlamaIndex with multiple fallback strategies",
    defaultExtensions: [".pdf"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100, maximum: 50000 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks", default: 100, minimum: 0, maximum: 5000 },
      chunk_strategy: { type: "string", description: "Chunking strategy for PDF content", default: "sentences", enum: ["sentences", "paragraphs", "pages", "semantic"] },
      extract_metadata: { type: "boolean", description: "Extract PDF metadata", default: true },
      extract_images: { type: "boolean", description: "Extract images from PDF", default: false },
      extract_tables: { type: "boolean", description: "Extract tables from PDF", default: true },
      fallback_strategies: { type: "array", description: "Fallback strategies to try in order", default: ["llama_pdf_reader", "llama_pymupdf_reader", "direct_pymupdf", "pypdf2_fallback"], items: { type: "string" } }
    },
  },
  "PDFParser_PyPDF2": {
    type: "PDFParser_PyPDF2",
    title: "PDF Parser (PyPDF2) Configuration",
    description: "Enhanced PDF parser using PyPDF2 with comprehensive capabilities",
    defaultExtensions: [".pdf"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100, maximum: 50000 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks in characters", default: 100, minimum: 0, maximum: 5000 },
      chunk_strategy: { type: "string", description: "Chunking strategy using PyPDF2 text structure", default: "paragraphs", enum: ["paragraphs", "sentences", "characters"] },
      extract_metadata: { type: "boolean", description: "Extract PDF metadata using PyPDF2", default: true },
      preserve_layout: { type: "boolean", description: "Use PyPDF2 layout-preserving extraction mode", default: true },
      extract_page_info: { type: "boolean", description: "Extract page numbers and rotation info", default: true },
      extract_annotations: { type: "boolean", description: "Extract PDF annotations using PyPDF2", default: false },
      extract_links: { type: "boolean", description: "Extract hyperlinks", default: false },
      extract_form_fields: { type: "boolean", description: "Extract form fields using PyPDF2", default: false },
      extract_outlines: { type: "boolean", description: "Extract document outlines/bookmarks", default: false },
      extract_images: { type: "boolean", description: "Extract embedded images using PyPDF2", default: false },
      extract_xmp_metadata: { type: "boolean", description: "Extract XMP metadata using PyPDF2", default: false },
      clean_text: { type: "boolean", description: "Clean extracted text", default: true },
      combine_pages: { type: "boolean", description: "Combine all pages into a single document. MUST be false to enable chunking.", default: false }
    },
  },
  "TextParser_LlamaIndex": {
    type: "TextParser_LlamaIndex",
    title: "Text Parser (LlamaIndex) Configuration",
    description: "Advanced text parser using LlamaIndex with semantic splitting, code parsing, and multi-format support",
    defaultExtensions: [".txt"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100, maximum: 50000 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks", default: 100, minimum: 0, maximum: 5000 },
      chunk_strategy: { type: "string", description: "Advanced chunking strategy - semantic uses content-based splitting, code preserves syntax", default: "semantic", enum: ["characters", "sentences", "paragraphs", "tokens", "semantic", "code"] },
      encoding: { type: "string", description: "Text encoding", default: "utf-8" },
      clean_text: { type: "boolean", description: "Clean extracted text", default: true },
      extract_metadata: { type: "boolean", description: "Extract comprehensive file and content metadata", default: true },
      semantic_buffer_size: { type: "integer", description: "Buffer size for semantic chunking", default: 1, minimum: 1, maximum: 10 },
      semantic_breakpoint_percentile_threshold: { type: "integer", description: "Percentile threshold for semantic breakpoints", default: 95, minimum: 50, maximum: 99 },
      token_model: { type: "string", description: "Tokenizer model for token-based chunking", default: "gpt-3.5-turbo" },
      preserve_code_structure: { type: "boolean", description: "Preserve code syntax and structure when parsing code files", default: true },
      detect_language: { type: "boolean", description: "Automatically detect programming language for code files", default: true },
      include_prev_next_rel: { type: "boolean", description: "Include relationships between chunks for better context", default: true }
    },
  },
  "TextParser_Python": {
    type: "TextParser_Python",
    title: "Text Parser (Python) Configuration",
    description: "Text parser using native Python with encoding detection",
    defaultExtensions: [".txt"],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size in characters", default: 1000, minimum: 100 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks", default: 100, minimum: 0 },
      chunk_strategy: { type: "string", description: "Text chunking strategy", default: "sentences", enum: ["sentences", "paragraphs", "characters"] },
      encoding: { type: "string", description: "Text encoding (utf-8 or auto-detect)", default: "utf-8" },
      clean_text: { type: "boolean", description: "Remove excessive whitespace", default: true },
      extract_metadata: { type: "boolean", description: "Extract file statistics", default: true }
    },
  },
  "auto": {
    type: "auto",
    title: "Auto Parser Configuration",
    description: "",
    defaultExtensions: [],
    properties: {
      chunk_size: { type: "integer", description: "Chunk size for text splitting", default: 1000, minimum: 100, maximum: 10000 },
      chunk_overlap: { type: "integer", description: "Overlap between chunks", default: 200, minimum: 0, maximum: 500 }
    },
  },
}

export const EXTRACTOR_SCHEMAS: Record<ExtractorType, ExtractorSchema> = {
  "ContentStatisticsExtractor": {
    type: "ContentStatisticsExtractor",
    title: "Content Statistics Extractor Configuration",
    description: "",
    properties: {
      include_readability: { type: "boolean", description: "Calculate readability scores", default: true },
      include_vocabulary: { type: "boolean", description: "Analyze vocabulary", default: true },
      include_structure: { type: "boolean", description: "Analyze text structure", default: true },
      include_sentiment_indicators: { type: "boolean", description: "Include detailed sentiment indicators", default: false }
    },
  },
  "DateTimeExtractor": {
    type: "DateTimeExtractor",
    title: "DateTime Extractor Configuration",
    description: "",
    properties: {
      fuzzy_parsing: { type: "boolean", description: "Enable fuzzy parsing", default: true },
      extract_relative: { type: "boolean", description: "Extract relative dates", default: true },
      extract_times: { type: "boolean", description: "Extract time expressions", default: true },
      extract_durations: { type: "boolean", description: "Extract durations", default: true },
      default_timezone: { type: "string", description: "Default timezone", default: "UTC" },
      date_format: { type: "string", description: "Output date format", default: "ISO" },
      prefer_dates_from: { type: "string", description: "Preference for ambiguous dates", default: "current", enum: ["past", "future", "current"] }
    },
  },
  "EntityExtractor": {
    type: "EntityExtractor",
    title: "Entity Extractor Configuration",
    description: "",
    properties: {
      model: { type: "string", description: "NER model name", default: "en_core_web_sm" },
      entity_types: { type: "array", description: "Entity types to extract", default: ["PERSON", "ORG", "GPE", "DATE", "TIME", "MONEY", "EMAIL", "PHONE", "URL", "PERCENT", "PRODUCT", "EVENT"], items: { type: "string" } },
      use_fallback: { type: "boolean", description: "Use regex fallback", default: true },
      min_entity_length: { type: "integer", description: "Minimum entity length", default: 2, minimum: 1 },
      merge_entities: { type: "boolean", description: "Merge adjacent entities", default: true },
      confidence_threshold: { type: "number", description: "Minimum confidence score", default: 0.7, minimum: 0.0, maximum: 1.0 }
    },
  },
  "HeadingExtractor": {
    type: "HeadingExtractor",
    title: "Heading Extractor Configuration",
    description: "",
    properties: {
      max_level: { type: "integer", description: "Maximum heading level", default: 6, minimum: 1, maximum: 6 },
      include_hierarchy: { type: "boolean", description: "Include hierarchy structure", default: true },
      extract_outline: { type: "boolean", description: "Generate document outline", default: true },
      min_heading_length: { type: "integer", description: "Minimum heading length", default: 3, minimum: 1 },
      enabled: { type: "boolean", description: "Enable this extractor", default: true }
    },
  },
  "KeywordExtractor": {
    type: "KeywordExtractor",
    title: "Keyword Extractor Configuration",
    description: "",
    properties: {
      extractor_type: { type: "string", description: "Extractor type discriminator" },
      algorithm: { type: "string", description: "Extraction algorithm", default: "rake", enum: ["rake", "yake", "tfidf", "textrank"] },
      max_keywords: { type: "integer", description: "Maximum keywords to extract", default: 10, minimum: 1, maximum: 100 },
      min_length: { type: "integer", description: "Minimum word length for keywords", default: 1, minimum: 1 },
      max_length: { type: "integer", description: "Maximum word length for keywords", default: 4, minimum: 1 },
      min_frequency: { type: "integer", description: "Minimum frequency for keywords", default: 1, minimum: 1 },
      stop_words: { type: "array", description: "Custom stop words", items: { type: "string" } },
      language: { type: "string", description: "Language for YAKE algorithm", default: "en" },
      max_ngram_size: { type: "integer", description: "Maximum n-gram size for YAKE", default: 3, minimum: 1, maximum: 5 },
      deduplication_threshold: { type: "number", description: "Deduplication threshold for YAKE", default: 0.9, minimum: 0.0, maximum: 1.0 }
    },
  },
  "LinkExtractor": {
    type: "LinkExtractor",
    title: "Link Extractor Configuration",
    description: "",
    properties: {
      extract_urls: { type: "boolean", description: "Extract URLs", default: true },
      extract_emails: { type: "boolean", description: "Extract email addresses", default: true },
      extract_domains: { type: "boolean", description: "Extract unique domains", default: true },
      validate_urls: { type: "boolean", description: "Validate URL format", default: false },
      resolve_redirects: { type: "boolean", description: "Resolve URL redirects", default: false },
      enabled: { type: "boolean", description: "Enable this extractor", default: true }
    },
  },
  "PathExtractor": {
    type: "PathExtractor",
    title: "Path Extractor Configuration",
    description: "",
    properties: {
      extract_file_paths: { type: "boolean", description: "Extract file paths", default: true },
      extract_urls: { type: "boolean", description: "Extract URL paths", default: true },
      extract_s3_paths: { type: "boolean", description: "Extract S3 paths", default: true },
      validate_paths: { type: "boolean", description: "Validate path existence", default: false },
      normalize_paths: { type: "boolean", description: "Normalize path formats", default: true },
      enabled: { type: "boolean", description: "Enable this extractor", default: true }
    },
  },
  "PatternExtractor": {
    type: "PatternExtractor",
    title: "Pattern Extractor Configuration",
    description: "",
    properties: {
      predefined_patterns: { type: "array", description: "Use built-in patterns for common data types (e.g., email, phone). Takes precedence over 'patterns' field for matching pattern names", default: [], items: { type: "string" } },
      custom_patterns: { type: "array", description: "Custom regex patterns", default: [], items: { type: "object" } },
      case_sensitive: { type: "boolean", description: "Case-sensitive matching", default: false },
      return_positions: { type: "boolean", description: "Return match positions", default: false },
      include_context: { type: "boolean", description: "Include surrounding context in results", default: false },
      max_matches_per_pattern: { type: "integer", description: "Maximum matches per pattern", default: 100, minimum: 1 },
      deduplicate_matches: { type: "boolean", description: "Remove duplicate matches", default: true }
    },
  },
  "RAKEExtractor": {
    type: "RAKEExtractor",
    title: "Keyword Extractor Configuration",
    description: "",
    properties: {
      extractor_type: { type: "string", description: "Extractor type discriminator" },
      algorithm: { type: "string", description: "Extraction algorithm", default: "rake", enum: ["rake", "yake", "tfidf", "textrank"] },
      max_keywords: { type: "integer", description: "Maximum keywords to extract", default: 10, minimum: 1, maximum: 100 },
      min_length: { type: "integer", description: "Minimum word length for keywords", default: 1, minimum: 1 },
      max_length: { type: "integer", description: "Maximum word length for keywords", default: 4, minimum: 1 },
      min_frequency: { type: "integer", description: "Minimum frequency for keywords", default: 1, minimum: 1 },
      stop_words: { type: "array", description: "Custom stop words", items: { type: "string" } },
      language: { type: "string", description: "Language for YAKE algorithm", default: "en" },
      max_ngram_size: { type: "integer", description: "Maximum n-gram size for YAKE", default: 3, minimum: 1, maximum: 5 },
      deduplication_threshold: { type: "number", description: "Deduplication threshold for YAKE", default: 0.9, minimum: 0.0, maximum: 1.0 }
    },
  },
  "SummaryExtractor": {
    type: "SummaryExtractor",
    title: "Summary Extractor Configuration",
    description: "",
    properties: {
      summary_sentences: { type: "integer", description: "Number of summary sentences", default: 3, minimum: 1, maximum: 10 },
      algorithm: { type: "string", description: "Summarization algorithm", default: "textrank", enum: ["textrank", "lsa", "luhn", "lexrank"] },
      include_key_phrases: { type: "boolean", description: "Extract key phrases", default: true },
      include_statistics: { type: "boolean", description: "Include text statistics", default: true },
      min_sentence_length: { type: "integer", description: "Minimum sentence length for summary", default: 10, minimum: 1 },
      max_sentence_length: { type: "integer", description: "Maximum sentence length for summary", default: 500, minimum: 10 }
    },
  },
  "TFIDFExtractor": {
    type: "TFIDFExtractor",
    title: "Keyword Extractor Configuration",
    description: "",
    properties: {
      extractor_type: { type: "string", description: "Extractor type discriminator" },
      algorithm: { type: "string", description: "Extraction algorithm", default: "rake", enum: ["rake", "yake", "tfidf", "textrank"] },
      max_keywords: { type: "integer", description: "Maximum keywords to extract", default: 10, minimum: 1, maximum: 100 },
      min_length: { type: "integer", description: "Minimum word length for keywords", default: 1, minimum: 1 },
      max_length: { type: "integer", description: "Maximum word length for keywords", default: 4, minimum: 1 },
      min_frequency: { type: "integer", description: "Minimum frequency for keywords", default: 1, minimum: 1 },
      stop_words: { type: "array", description: "Custom stop words", items: { type: "string" } },
      language: { type: "string", description: "Language for YAKE algorithm", default: "en" },
      max_ngram_size: { type: "integer", description: "Maximum n-gram size for YAKE", default: 3, minimum: 1, maximum: 5 },
      deduplication_threshold: { type: "number", description: "Deduplication threshold for YAKE", default: 0.9, minimum: 0.0, maximum: 1.0 }
    },
  },
  "TableExtractor": {
    type: "TableExtractor",
    title: "Table Extractor Configuration",
    description: "",
    properties: {
      output_format: { type: "string", description: "Output format", default: "dict", enum: ["dict", "list", "csv", "markdown"] },
      extract_headers: { type: "boolean", description: "Extract table headers", default: true },
      merge_cells: { type: "boolean", description: "Handle merged cells", default: true },
      min_rows: { type: "integer", description: "Minimum rows for table", default: 2, minimum: 1 },
      enabled: { type: "boolean", description: "Enable this extractor", default: true }
    },
  },
  "YAKEExtractor": {
    type: "YAKEExtractor",
    title: "YAKE Extractor Configuration",
    description: "",
    properties: {
      extractor_type: { type: "string", description: "Extractor type discriminator" },
      max_keywords: { type: "integer", description: "Maximum keywords to extract", default: 10, minimum: 1, maximum: 100 },
      language: { type: "string", description: "Language for YAKE algorithm", default: "en" },
      max_ngram_size: { type: "integer", description: "Maximum n-gram size for YAKE", default: 3, minimum: 1, maximum: 5 },
      deduplication_threshold: { type: "number", description: "Deduplication threshold for YAKE", default: 0.9, minimum: 0.0, maximum: 1.0 }
    },
  },
}
