/**
 * AUTO-GENERATED FILE - DO NOT EDIT
 *
 * Generated from rag/schema.yaml by designer/generate-types.ts
 * Run: cd designer && ./generate-types.sh
 */

// ============================================================================
// Parser Types
// ============================================================================

export const PARSER_TYPES = ["CSVParser_LlamaIndex","CSVParser_Pandas","CSVParser_Python","DOCXParser_LlamaIndex","DOCXParser_PythonDocx","EXCELParser_LlamaIndex","EXCELParser_OpenPyXL","EXCELParser_Pandas","MARKDOWNParser_LlamaIndex","MARKDOWNParser_Python","MSGParser_ExtractMsg","PDFParser_LlamaIndex","PDFParser_PyPDF2","TEXTParser_LlamaIndex","TEXTParser_Python","auto"] as const

export type ParserType = typeof PARSER_TYPES[number]

// ============================================================================
// Extractor Types
// ============================================================================

export const EXTRACTOR_TYPES = ["ContentStatisticsExtractor","DateTimeExtractor","EntityExtractor","HeadingExtractor","KeywordExtractor","LinkExtractor","PathExtractor","PatternExtractor","RAKEExtractor","SummaryExtractor","TFIDFExtractor","TableExtractor","YAKEExtractor"] as const

export type ExtractorType = typeof EXTRACTOR_TYPES[number]

// ============================================================================
// Default Configurations
// ============================================================================

const PARSER_DEFAULTS = {
  "CSVParser_LlamaIndex": {
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
  "CSVParser_Pandas": {
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
  "CSVParser_Python": {
    "chunk_size": 1000,
    "encoding": "utf-8",
    "delimiter": ",",
    "quotechar": "\""
  },
  "DOCXParser_LlamaIndex": {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "chunk_strategy": "paragraphs",
    "extract_metadata": true,
    "extract_tables": true,
    "extract_images": false,
    "preserve_formatting": true,
    "include_header_footer": false
  },
  "DOCXParser_PythonDocx": {
    "chunk_size": 1000,
    "chunk_strategy": "paragraphs",
    "extract_metadata": true,
    "extract_tables": true,
    "extract_headers": true,
    "extract_footers": false,
    "extract_comments": false
  },
  "EXCELParser_LlamaIndex": {
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
  "EXCELParser_OpenPyXL": {
    "chunk_size": 1000,
    "extract_formulas": false,
    "extract_metadata": true,
    "sheets": null,
    "data_only": true
  },
  "EXCELParser_Pandas": {
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
  "MARKDOWNParser_LlamaIndex": {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "chunk_strategy": "headings",
    "extract_metadata": true,
    "extract_code_blocks": true,
    "extract_tables": true,
    "extract_links": true,
    "preserve_structure": true
  },
  "MARKDOWNParser_Python": {
    "chunk_size": 1000,
    "chunk_strategy": "sections",
    "extract_metadata": true,
    "extract_code_blocks": true,
    "extract_links": true
  },
  "MSGParser_ExtractMsg": {
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
  "PDFParser_LlamaIndex": {
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
  "PDFParser_PyPDF2": {
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
  "TEXTParser_LlamaIndex": {
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
  "TEXTParser_Python": {
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "chunk_strategy": "sentences",
    "encoding": "utf-8",
    "clean_text": true,
    "extract_metadata": true
  },
  "auto": {
    "chunk_size": 1000,
    "chunk_overlap": 200
  }
} as const

export function getDefaultParserConfig(parserType: ParserType): Record<string, any> {
  return (PARSER_DEFAULTS as any)[parserType] || {}
}

const EXTRACTOR_DEFAULTS = {
  "ContentStatisticsExtractor": {
    "include_readability": true,
    "include_vocabulary": true,
    "include_structure": true,
    "include_sentiment_indicators": false
  },
  "DateTimeExtractor": {
    "fuzzy_parsing": true,
    "extract_relative": true,
    "extract_times": true,
    "extract_durations": true,
    "default_timezone": "UTC",
    "date_format": "ISO",
    "prefer_dates_from": "current"
  },
  "EntityExtractor": {
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
  "HeadingExtractor": {
    "max_level": 6,
    "include_hierarchy": true,
    "extract_outline": true,
    "min_heading_length": 3,
    "enabled": true
  },
  "KeywordExtractor": {
    "algorithm": "rake",
    "max_keywords": 10,
    "min_length": 1,
    "max_length": 4,
    "min_frequency": 1,
    "language": "en",
    "max_ngram_size": 3,
    "deduplication_threshold": 0.9
  },
  "LinkExtractor": {
    "extract_urls": true,
    "extract_emails": true,
    "extract_domains": true,
    "validate_urls": false,
    "resolve_redirects": false,
    "enabled": true
  },
  "PathExtractor": {
    "extract_file_paths": true,
    "extract_urls": true,
    "extract_s3_paths": true,
    "validate_paths": false,
    "normalize_paths": true,
    "enabled": true
  },
  "PatternExtractor": {
    "predefined_patterns": [],
    "custom_patterns": [],
    "case_sensitive": false,
    "return_positions": false,
    "include_context": false,
    "max_matches_per_pattern": 100,
    "deduplicate_matches": true
  },
  "RAKEExtractor": {
    "algorithm": "rake",
    "max_keywords": 10,
    "min_length": 1,
    "max_length": 4,
    "min_frequency": 1,
    "language": "en",
    "max_ngram_size": 3,
    "deduplication_threshold": 0.9
  },
  "SummaryExtractor": {
    "summary_sentences": 3,
    "algorithm": "textrank",
    "include_key_phrases": true,
    "include_statistics": true,
    "min_sentence_length": 10,
    "max_sentence_length": 500
  },
  "TFIDFExtractor": {
    "algorithm": "rake",
    "max_keywords": 10,
    "min_length": 1,
    "max_length": 4,
    "min_frequency": 1,
    "language": "en",
    "max_ngram_size": 3,
    "deduplication_threshold": 0.9
  },
  "TableExtractor": {
    "output_format": "dict",
    "extract_headers": true,
    "merge_cells": true,
    "min_rows": 2,
    "enabled": true
  },
  "YAKEExtractor": {
    "algorithm": "rake",
    "max_keywords": 10,
    "min_length": 1,
    "max_length": 4,
    "min_frequency": 1,
    "language": "en",
    "max_ngram_size": 3,
    "deduplication_threshold": 0.9
  }
} as const

export function getDefaultExtractorConfig(extractorType: ExtractorType): Record<string, any> {
  return (EXTRACTOR_DEFAULTS as any)[extractorType] || {}
}

// ============================================================================
// Schema Metadata
// ============================================================================

export type PrimitiveType = 'integer' | 'number' | 'string' | 'boolean' | 'array'

export interface SchemaField {
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
  defaultExtensions?: string[]
  properties: Record<string, any>
  required?: string[]
}

export interface ExtractorSchema {
  type: ExtractorType
  title: string
  description: string
  properties: Record<string, any>
  required?: string[]
}

export const PARSER_SCHEMAS: Record<ParserType, ParserSchema> = {
  "CSVParser_LlamaIndex": {
    "type": "CSVParser_LlamaIndex",
    "title": "CSV Parser (LlamaIndex) Configuration",
    "description": "CSV parser using LlamaIndex with Pandas backend for advanced processing",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Number of rows per chunk"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "rows",
          "semantic",
          "full"
        ],
        "default": "rows",
        "description": "Chunking strategy"
      },
      "field_mapping": {
        "type": "object",
        "additionalProperties": {
          "type": "string"
        },
        "description": "Map CSV columns to standard fields"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract metadata from CSV"
      },
      "combine_fields": {
        "type": "boolean",
        "default": true,
        "description": "Combine fields into text content"
      },
      "skiprows": {
        "type": "integer",
        "minimum": 0,
        "description": "Number of rows to skip at beginning"
      },
      "na_values": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "default": [
          "",
          "NA",
          "N/A",
          "null",
          "None"
        ],
        "description": "Values to treat as missing"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".csv"
    ]
  },
  "CSVParser_Pandas": {
    "type": "CSVParser_Pandas",
    "title": "CSV Parser (Pandas) Configuration",
    "description": "Advanced CSV parser using Pandas with data analysis capabilities",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "description": "Number of rows per chunk"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "rows",
          "columns",
          "full"
        ],
        "default": "rows",
        "description": "How to chunk the CSV data"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract data statistics and metadata"
      },
      "encoding": {
        "type": "string",
        "default": "utf-8",
        "description": "File encoding"
      },
      "delimiter": {
        "type": "string",
        "default": ",",
        "description": "CSV delimiter"
      },
      "na_values": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "default": [
          "",
          "NA",
          "N/A",
          "null",
          "None"
        ],
        "description": "Values to treat as NaN"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".csv"
    ]
  },
  "CSVParser_Python": {
    "type": "CSVParser_Python",
    "title": "CSV Parser (Python) Configuration",
    "description": "Simple CSV parser using native Python csv module",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "description": "Number of rows per chunk"
      },
      "encoding": {
        "type": "string",
        "default": "utf-8",
        "description": "File encoding"
      },
      "delimiter": {
        "type": "string",
        "default": ",",
        "description": "CSV delimiter"
      },
      "quotechar": {
        "type": "string",
        "default": "\"",
        "description": "Quote character"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".csv"
    ]
  },
  "DOCXParser_LlamaIndex": {
    "type": "DOCXParser_LlamaIndex",
    "title": "DOCX Parser (LlamaIndex) Configuration",
    "description": "Advanced DOCX parser using LlamaIndex with enhanced chunking",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Chunk size in characters"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 100,
        "minimum": 0,
        "maximum": 5000,
        "description": "Overlap between chunks"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "paragraphs",
          "sentences",
          "semantic"
        ],
        "default": "paragraphs",
        "description": "Chunking strategy"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract document metadata"
      },
      "extract_tables": {
        "type": "boolean",
        "default": true,
        "description": "Extract tables from document"
      },
      "extract_images": {
        "type": "boolean",
        "default": false,
        "description": "Extract images from document"
      },
      "preserve_formatting": {
        "type": "boolean",
        "default": true,
        "description": "Preserve text formatting"
      },
      "include_header_footer": {
        "type": "boolean",
        "default": false,
        "description": "Include header and footer content"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".docx"
    ]
  },
  "DOCXParser_PythonDocx": {
    "type": "DOCXParser_PythonDocx",
    "title": "DOCX Parser (python-docx) Configuration",
    "description": "Word document parser using python-docx library",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "description": "Chunk size in characters"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "paragraphs",
          "sentences",
          "characters"
        ],
        "default": "paragraphs",
        "description": "Chunking strategy"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract document metadata"
      },
      "extract_tables": {
        "type": "boolean",
        "default": true,
        "description": "Extract tables using python-docx"
      },
      "extract_headers": {
        "type": "boolean",
        "default": true,
        "description": "Extract headers"
      },
      "extract_footers": {
        "type": "boolean",
        "default": false,
        "description": "Extract footers"
      },
      "extract_comments": {
        "type": "boolean",
        "default": false,
        "description": "Extract comments"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".docx"
    ]
  },
  "EXCELParser_LlamaIndex": {
    "type": "EXCELParser_LlamaIndex",
    "title": "Excel Parser (LlamaIndex) Configuration",
    "description": "Excel parser using LlamaIndex with Pandas backend for advanced processing",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Number of rows per chunk"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "rows",
          "semantic",
          "full"
        ],
        "default": "rows",
        "description": "Chunking strategy"
      },
      "sheets": {
        "type": [
          "array",
          "null"
        ],
        "items": {
          "type": "string"
        },
        "description": "Specific sheets to parse (null for all)"
      },
      "combine_sheets": {
        "type": "boolean",
        "default": false,
        "description": "Combine all sheets into one document"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract metadata from Excel"
      },
      "extract_formulas": {
        "type": "boolean",
        "default": false,
        "description": "Extract formulas instead of values"
      },
      "header_row": {
        "type": "integer",
        "default": 0,
        "minimum": 0,
        "description": "Row index for headers"
      },
      "skiprows": {
        "type": "integer",
        "minimum": 0,
        "description": "Number of rows to skip"
      },
      "na_values": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "default": [
          "",
          "NA",
          "N/A",
          "null",
          "None"
        ],
        "description": "Values to treat as missing"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".xlsx",
      ".xls"
    ]
  },
  "EXCELParser_OpenPyXL": {
    "type": "EXCELParser_OpenPyXL",
    "title": "Excel Parser (OpenPyXL) Configuration",
    "description": "Excel parser using OpenPyXL for XLSX files with formula support",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "description": "Number of rows per chunk"
      },
      "extract_formulas": {
        "type": "boolean",
        "default": false,
        "description": "Extract cell formulas using OpenPyXL"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract workbook metadata"
      },
      "sheets": {
        "type": [
          "array",
          "null"
        ],
        "items": {
          "type": "string"
        },
        "default": null,
        "description": "Specific sheets to process (null = all)"
      },
      "data_only": {
        "type": "boolean",
        "default": true,
        "description": "Extract values instead of formulas"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".xlsx",
      ".xls"
    ]
  },
  "EXCELParser_Pandas": {
    "type": "EXCELParser_Pandas",
    "title": "Excel Parser (Pandas) Configuration",
    "description": "Excel parser using Pandas with data analysis capabilities",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "description": "Number of rows per chunk"
      },
      "sheets": {
        "type": [
          "array",
          "null"
        ],
        "items": {
          "type": "string"
        },
        "default": null,
        "description": "Specific sheets to process (null = all)"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract data statistics"
      },
      "skiprows": {
        "type": [
          "integer",
          "null"
        ],
        "default": null,
        "description": "Rows to skip at beginning"
      },
      "na_values": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "default": [
          "",
          "NA",
          "N/A",
          "null",
          "None"
        ],
        "description": "Values to treat as NaN"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".xlsx",
      ".xls"
    ]
  },
  "MARKDOWNParser_LlamaIndex": {
    "type": "MARKDOWNParser_LlamaIndex",
    "title": "Markdown Parser (LlamaIndex) Configuration",
    "description": "Advanced markdown parser using LlamaIndex with semantic chunking",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Chunk size in characters"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 100,
        "minimum": 0,
        "maximum": 5000,
        "description": "Overlap between chunks"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "headings",
          "paragraphs",
          "sentences",
          "semantic"
        ],
        "default": "headings",
        "description": "Chunking strategy for markdown"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract frontmatter metadata"
      },
      "extract_code_blocks": {
        "type": "boolean",
        "default": true,
        "description": "Extract code blocks separately"
      },
      "extract_tables": {
        "type": "boolean",
        "default": true,
        "description": "Extract markdown tables"
      },
      "extract_links": {
        "type": "boolean",
        "default": true,
        "description": "Extract links and references"
      },
      "preserve_structure": {
        "type": "boolean",
        "default": true,
        "description": "Preserve heading hierarchy"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".md",
      ".markdown"
    ]
  },
  "MARKDOWNParser_Python": {
    "type": "MARKDOWNParser_Python",
    "title": "Markdown Parser (Python) Configuration",
    "description": "Markdown parser using native Python with regex parsing",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "description": "Chunk size in characters"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "sections",
          "paragraphs",
          "characters"
        ],
        "default": "sections",
        "description": "Chunking strategy - sections uses markdown headers"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract YAML frontmatter"
      },
      "extract_code_blocks": {
        "type": "boolean",
        "default": true,
        "description": "Extract code blocks"
      },
      "extract_links": {
        "type": "boolean",
        "default": true,
        "description": "Extract markdown links"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".md",
      ".markdown"
    ]
  },
  "MSGParser_ExtractMsg": {
    "type": "MSGParser_ExtractMsg",
    "title": "MSG Parser (extract-msg) Configuration",
    "description": "",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Chunk size in characters"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 100,
        "minimum": 0,
        "maximum": 5000,
        "description": "Overlap between chunks"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "sentences",
          "paragraphs",
          "characters",
          "email_sections"
        ],
        "default": "email_sections",
        "description": "Chunking strategy"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract metadata"
      },
      "extract_attachments": {
        "type": "boolean",
        "default": true,
        "description": "Extract attachments"
      },
      "extract_headers": {
        "type": "boolean",
        "default": true,
        "description": "Extract headers"
      },
      "include_attachment_content": {
        "type": "boolean",
        "default": true,
        "description": "Include attachment content"
      },
      "clean_text": {
        "type": "boolean",
        "default": true,
        "description": "Clean text"
      },
      "preserve_formatting": {
        "type": "boolean",
        "default": false,
        "description": "Preserve formatting"
      },
      "encoding": {
        "type": "string",
        "default": "utf-8",
        "description": "Encoding"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".msg"
    ]
  },
  "PDFParser_LlamaIndex": {
    "type": "PDFParser_LlamaIndex",
    "title": "PDF Parser (LlamaIndex) Configuration",
    "description": "Advanced PDF parser using LlamaIndex with multiple fallback strategies",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Chunk size in characters"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 100,
        "minimum": 0,
        "maximum": 5000,
        "description": "Overlap between chunks"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "sentences",
          "paragraphs",
          "pages",
          "semantic"
        ],
        "default": "sentences",
        "description": "Chunking strategy for PDF content"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract PDF metadata"
      },
      "extract_images": {
        "type": "boolean",
        "default": false,
        "description": "Extract images from PDF"
      },
      "extract_tables": {
        "type": "boolean",
        "default": true,
        "description": "Extract tables from PDF"
      },
      "fallback_strategies": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": [
            "llama_pdf_reader",
            "llama_pymupdf_reader",
            "direct_pymupdf",
            "pypdf2_fallback"
          ]
        },
        "default": [
          "llama_pdf_reader",
          "llama_pymupdf_reader",
          "direct_pymupdf",
          "pypdf2_fallback"
        ],
        "description": "Fallback strategies to try in order"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".pdf"
    ]
  },
  "PDFParser_PyPDF2": {
    "type": "PDFParser_PyPDF2",
    "title": "PDF Parser (PyPDF2) Configuration",
    "description": "Enhanced PDF parser using PyPDF2 with comprehensive capabilities",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Chunk size in characters"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 100,
        "minimum": 0,
        "maximum": 5000,
        "description": "Overlap between chunks in characters"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "paragraphs",
          "sentences",
          "characters"
        ],
        "default": "paragraphs",
        "description": "Chunking strategy using PyPDF2 text structure"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract PDF metadata using PyPDF2"
      },
      "preserve_layout": {
        "type": "boolean",
        "default": true,
        "description": "Use PyPDF2 layout-preserving extraction mode"
      },
      "extract_page_info": {
        "type": "boolean",
        "default": true,
        "description": "Extract page numbers and rotation info"
      },
      "extract_annotations": {
        "type": "boolean",
        "default": false,
        "description": "Extract PDF annotations using PyPDF2"
      },
      "extract_links": {
        "type": "boolean",
        "default": false,
        "description": "Extract hyperlinks"
      },
      "extract_form_fields": {
        "type": "boolean",
        "default": false,
        "description": "Extract form fields using PyPDF2"
      },
      "extract_outlines": {
        "type": "boolean",
        "default": false,
        "description": "Extract document outlines/bookmarks"
      },
      "extract_images": {
        "type": "boolean",
        "default": false,
        "description": "Extract embedded images using PyPDF2"
      },
      "extract_xmp_metadata": {
        "type": "boolean",
        "default": false,
        "description": "Extract XMP metadata using PyPDF2"
      },
      "clean_text": {
        "type": "boolean",
        "default": true,
        "description": "Clean extracted text"
      },
      "combine_pages": {
        "type": "boolean",
        "default": false,
        "description": "Combine all pages into a single document. MUST be false to enable chunking."
      }
    },
    "required": [],
    "defaultExtensions": [
      ".pdf"
    ]
  },
  "TEXTParser_LlamaIndex": {
    "type": "TEXTParser_LlamaIndex",
    "title": "Text Parser (LlamaIndex) Configuration",
    "description": "Advanced text parser using LlamaIndex with semantic splitting, code parsing, and multi-format support",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 50000,
        "description": "Chunk size in characters"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 100,
        "minimum": 0,
        "maximum": 5000,
        "description": "Overlap between chunks"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "characters",
          "sentences",
          "paragraphs",
          "tokens",
          "semantic",
          "code"
        ],
        "default": "semantic",
        "description": "Advanced chunking strategy - semantic uses content-based splitting, code preserves syntax"
      },
      "encoding": {
        "type": "string",
        "default": "utf-8",
        "description": "Text encoding"
      },
      "clean_text": {
        "type": "boolean",
        "default": true,
        "description": "Clean extracted text"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract comprehensive file and content metadata"
      },
      "semantic_buffer_size": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "maximum": 10,
        "description": "Buffer size for semantic chunking"
      },
      "semantic_breakpoint_percentile_threshold": {
        "type": "integer",
        "default": 95,
        "minimum": 50,
        "maximum": 99,
        "description": "Percentile threshold for semantic breakpoints"
      },
      "token_model": {
        "type": "string",
        "default": "gpt-3.5-turbo",
        "description": "Tokenizer model for token-based chunking"
      },
      "preserve_code_structure": {
        "type": "boolean",
        "default": true,
        "description": "Preserve code syntax and structure when parsing code files"
      },
      "detect_language": {
        "type": "boolean",
        "default": true,
        "description": "Automatically detect programming language for code files"
      },
      "include_prev_next_rel": {
        "type": "boolean",
        "default": true,
        "description": "Include relationships between chunks for better context"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".txt"
    ]
  },
  "TEXTParser_Python": {
    "type": "TEXTParser_Python",
    "title": "Text Parser (Python) Configuration",
    "description": "Text parser using native Python with encoding detection",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "description": "Chunk size in characters"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 100,
        "minimum": 0,
        "description": "Overlap between chunks"
      },
      "chunk_strategy": {
        "type": "string",
        "enum": [
          "sentences",
          "paragraphs",
          "characters"
        ],
        "default": "sentences",
        "description": "Text chunking strategy"
      },
      "encoding": {
        "type": "string",
        "default": "utf-8",
        "description": "Text encoding (utf-8 or auto-detect)"
      },
      "clean_text": {
        "type": "boolean",
        "default": true,
        "description": "Remove excessive whitespace"
      },
      "extract_metadata": {
        "type": "boolean",
        "default": true,
        "description": "Extract file statistics"
      }
    },
    "required": [],
    "defaultExtensions": [
      ".txt"
    ]
  },
  "auto": {
    "type": "auto",
    "title": "Auto Parser Configuration",
    "description": "",
    "properties": {
      "chunk_size": {
        "type": "integer",
        "default": 1000,
        "minimum": 100,
        "maximum": 10000,
        "description": "Chunk size for text splitting"
      },
      "chunk_overlap": {
        "type": "integer",
        "default": 200,
        "minimum": 0,
        "maximum": 500,
        "description": "Overlap between chunks"
      }
    },
    "required": []
  }
}

export const EXTRACTOR_SCHEMAS: Record<ExtractorType, ExtractorSchema> = {
  "ContentStatisticsExtractor": {
    "type": "ContentStatisticsExtractor",
    "title": "Content Statistics Extractor Configuration",
    "description": "",
    "properties": {
      "include_readability": {
        "type": "boolean",
        "default": true,
        "description": "Calculate readability scores"
      },
      "include_vocabulary": {
        "type": "boolean",
        "default": true,
        "description": "Analyze vocabulary"
      },
      "include_structure": {
        "type": "boolean",
        "default": true,
        "description": "Analyze text structure"
      },
      "include_sentiment_indicators": {
        "type": "boolean",
        "default": false,
        "description": "Include detailed sentiment indicators"
      }
    },
    "required": []
  },
  "DateTimeExtractor": {
    "type": "DateTimeExtractor",
    "title": "DateTime Extractor Configuration",
    "description": "",
    "properties": {
      "fuzzy_parsing": {
        "type": "boolean",
        "default": true,
        "description": "Enable fuzzy parsing"
      },
      "extract_relative": {
        "type": "boolean",
        "default": true,
        "description": "Extract relative dates"
      },
      "extract_times": {
        "type": "boolean",
        "default": true,
        "description": "Extract time expressions"
      },
      "extract_durations": {
        "type": "boolean",
        "default": true,
        "description": "Extract durations"
      },
      "default_timezone": {
        "type": "string",
        "default": "UTC",
        "description": "Default timezone"
      },
      "date_format": {
        "type": "string",
        "default": "ISO",
        "description": "Output date format"
      },
      "prefer_dates_from": {
        "type": "string",
        "enum": [
          "past",
          "future",
          "current"
        ],
        "default": "current",
        "description": "Preference for ambiguous dates"
      }
    },
    "required": []
  },
  "EntityExtractor": {
    "type": "EntityExtractor",
    "title": "Entity Extractor Configuration",
    "description": "",
    "properties": {
      "model": {
        "type": "string",
        "default": "en_core_web_sm",
        "description": "NER model name"
      },
      "entity_types": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": [
            "PERSON",
            "ORG",
            "GPE",
            "DATE",
            "TIME",
            "MONEY",
            "EMAIL",
            "PHONE",
            "URL",
            "LAW",
            "PERCENT",
            "PRODUCT",
            "EVENT",
            "VERSION",
            "FAC",
            "LOC"
          ]
        },
        "default": [
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
        "description": "Entity types to extract"
      },
      "use_fallback": {
        "type": "boolean",
        "default": true,
        "description": "Use regex fallback"
      },
      "min_entity_length": {
        "type": "integer",
        "default": 2,
        "minimum": 1,
        "description": "Minimum entity length"
      },
      "merge_entities": {
        "type": "boolean",
        "default": true,
        "description": "Merge adjacent entities"
      },
      "confidence_threshold": {
        "type": "number",
        "default": 0.7,
        "minimum": 0,
        "maximum": 1,
        "description": "Minimum confidence score"
      }
    },
    "required": []
  },
  "HeadingExtractor": {
    "type": "HeadingExtractor",
    "title": "Heading Extractor Configuration",
    "description": "",
    "properties": {
      "max_level": {
        "type": "integer",
        "default": 6,
        "minimum": 1,
        "maximum": 6,
        "description": "Maximum heading level"
      },
      "include_hierarchy": {
        "type": "boolean",
        "default": true,
        "description": "Include hierarchy structure"
      },
      "extract_outline": {
        "type": "boolean",
        "default": true,
        "description": "Generate document outline"
      },
      "min_heading_length": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "description": "Minimum heading length"
      },
      "enabled": {
        "type": "boolean",
        "default": true,
        "description": "Enable this extractor"
      }
    },
    "required": []
  },
  "KeywordExtractor": {
    "type": "KeywordExtractor",
    "title": "Keyword Extractor Configuration",
    "description": "",
    "properties": {
      "extractor_type": {
        "type": "string",
        "const": "keyword",
        "description": "Extractor type discriminator"
      },
      "algorithm": {
        "type": "string",
        "enum": [
          "rake",
          "yake",
          "tfidf",
          "textrank"
        ],
        "default": "rake",
        "description": "Extraction algorithm"
      },
      "max_keywords": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 100,
        "description": "Maximum keywords to extract"
      },
      "min_length": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum word length for keywords"
      },
      "max_length": {
        "type": "integer",
        "default": 4,
        "minimum": 1,
        "description": "Maximum word length for keywords"
      },
      "min_frequency": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum frequency for keywords"
      },
      "stop_words": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Custom stop words"
      },
      "language": {
        "type": "string",
        "default": "en",
        "description": "Language for YAKE algorithm"
      },
      "max_ngram_size": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 5,
        "description": "Maximum n-gram size for YAKE"
      },
      "deduplication_threshold": {
        "type": "number",
        "default": 0.9,
        "minimum": 0,
        "maximum": 1,
        "description": "Deduplication threshold for YAKE"
      }
    },
    "required": []
  },
  "LinkExtractor": {
    "type": "LinkExtractor",
    "title": "Link Extractor Configuration",
    "description": "",
    "properties": {
      "extract_urls": {
        "type": "boolean",
        "default": true,
        "description": "Extract URLs"
      },
      "extract_emails": {
        "type": "boolean",
        "default": true,
        "description": "Extract email addresses"
      },
      "extract_domains": {
        "type": "boolean",
        "default": true,
        "description": "Extract unique domains"
      },
      "validate_urls": {
        "type": "boolean",
        "default": false,
        "description": "Validate URL format"
      },
      "resolve_redirects": {
        "type": "boolean",
        "default": false,
        "description": "Resolve URL redirects"
      },
      "enabled": {
        "type": "boolean",
        "default": true,
        "description": "Enable this extractor"
      }
    },
    "required": []
  },
  "PathExtractor": {
    "type": "PathExtractor",
    "title": "Path Extractor Configuration",
    "description": "",
    "properties": {
      "extract_file_paths": {
        "type": "boolean",
        "default": true,
        "description": "Extract file paths"
      },
      "extract_urls": {
        "type": "boolean",
        "default": true,
        "description": "Extract URL paths"
      },
      "extract_s3_paths": {
        "type": "boolean",
        "default": true,
        "description": "Extract S3 paths"
      },
      "validate_paths": {
        "type": "boolean",
        "default": false,
        "description": "Validate path existence"
      },
      "normalize_paths": {
        "type": "boolean",
        "default": true,
        "description": "Normalize path formats"
      },
      "enabled": {
        "type": "boolean",
        "default": true,
        "description": "Enable this extractor"
      }
    },
    "required": []
  },
  "PatternExtractor": {
    "type": "PatternExtractor",
    "title": "Pattern Extractor Configuration",
    "description": "",
    "properties": {
      "predefined_patterns": {
        "type": "array",
        "items": {
          "type": "string",
          "enum": [
            "email",
            "phone",
            "url",
            "ip",
            "ip_address",
            "ssn",
            "credit_card",
            "zip_code",
            "file_path",
            "version",
            "date"
          ]
        },
        "default": [],
        "description": "Use built-in patterns for common data types (e.g., email, phone). Takes precedence over 'patterns' field for matching pattern names"
      },
      "custom_patterns": {
        "type": "array",
        "items": {
          "type": "object",
          "required": [
            "name",
            "pattern"
          ],
          "additionalProperties": false,
          "properties": {
            "name": {
              "type": "string",
              "description": "Pattern name"
            },
            "pattern": {
              "type": "string",
              "description": "Regex pattern"
            },
            "description": {
              "type": "string",
              "description": "Pattern description"
            }
          }
        },
        "default": [],
        "description": "Custom regex patterns"
      },
      "case_sensitive": {
        "type": "boolean",
        "default": false,
        "description": "Case-sensitive matching"
      },
      "return_positions": {
        "type": "boolean",
        "default": false,
        "description": "Return match positions"
      },
      "include_context": {
        "type": "boolean",
        "default": false,
        "description": "Include surrounding context in results"
      },
      "max_matches_per_pattern": {
        "type": "integer",
        "default": 100,
        "minimum": 1,
        "description": "Maximum matches per pattern"
      },
      "deduplicate_matches": {
        "type": "boolean",
        "default": true,
        "description": "Remove duplicate matches"
      }
    },
    "required": []
  },
  "RAKEExtractor": {
    "type": "RAKEExtractor",
    "title": "Keyword Extractor Configuration",
    "description": "",
    "properties": {
      "extractor_type": {
        "type": "string",
        "const": "keyword",
        "description": "Extractor type discriminator"
      },
      "algorithm": {
        "type": "string",
        "enum": [
          "rake",
          "yake",
          "tfidf",
          "textrank"
        ],
        "default": "rake",
        "description": "Extraction algorithm"
      },
      "max_keywords": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 100,
        "description": "Maximum keywords to extract"
      },
      "min_length": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum word length for keywords"
      },
      "max_length": {
        "type": "integer",
        "default": 4,
        "minimum": 1,
        "description": "Maximum word length for keywords"
      },
      "min_frequency": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum frequency for keywords"
      },
      "stop_words": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Custom stop words"
      },
      "language": {
        "type": "string",
        "default": "en",
        "description": "Language for YAKE algorithm"
      },
      "max_ngram_size": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 5,
        "description": "Maximum n-gram size for YAKE"
      },
      "deduplication_threshold": {
        "type": "number",
        "default": 0.9,
        "minimum": 0,
        "maximum": 1,
        "description": "Deduplication threshold for YAKE"
      }
    },
    "required": []
  },
  "SummaryExtractor": {
    "type": "SummaryExtractor",
    "title": "Summary Extractor Configuration",
    "description": "",
    "properties": {
      "summary_sentences": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 10,
        "description": "Number of summary sentences"
      },
      "algorithm": {
        "type": "string",
        "enum": [
          "textrank",
          "lsa",
          "luhn",
          "lexrank"
        ],
        "default": "textrank",
        "description": "Summarization algorithm"
      },
      "include_key_phrases": {
        "type": "boolean",
        "default": true,
        "description": "Extract key phrases"
      },
      "include_statistics": {
        "type": "boolean",
        "default": true,
        "description": "Include text statistics"
      },
      "min_sentence_length": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "description": "Minimum sentence length for summary"
      },
      "max_sentence_length": {
        "type": "integer",
        "default": 500,
        "minimum": 10,
        "description": "Maximum sentence length for summary"
      }
    },
    "required": []
  },
  "TFIDFExtractor": {
    "type": "TFIDFExtractor",
    "title": "Keyword Extractor Configuration",
    "description": "",
    "properties": {
      "extractor_type": {
        "type": "string",
        "const": "keyword",
        "description": "Extractor type discriminator"
      },
      "algorithm": {
        "type": "string",
        "enum": [
          "rake",
          "yake",
          "tfidf",
          "textrank"
        ],
        "default": "rake",
        "description": "Extraction algorithm"
      },
      "max_keywords": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 100,
        "description": "Maximum keywords to extract"
      },
      "min_length": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum word length for keywords"
      },
      "max_length": {
        "type": "integer",
        "default": 4,
        "minimum": 1,
        "description": "Maximum word length for keywords"
      },
      "min_frequency": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum frequency for keywords"
      },
      "stop_words": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Custom stop words"
      },
      "language": {
        "type": "string",
        "default": "en",
        "description": "Language for YAKE algorithm"
      },
      "max_ngram_size": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 5,
        "description": "Maximum n-gram size for YAKE"
      },
      "deduplication_threshold": {
        "type": "number",
        "default": 0.9,
        "minimum": 0,
        "maximum": 1,
        "description": "Deduplication threshold for YAKE"
      }
    },
    "required": []
  },
  "TableExtractor": {
    "type": "TableExtractor",
    "title": "Table Extractor Configuration",
    "description": "",
    "properties": {
      "output_format": {
        "type": "string",
        "enum": [
          "dict",
          "list",
          "csv",
          "markdown"
        ],
        "default": "dict",
        "description": "Output format"
      },
      "extract_headers": {
        "type": "boolean",
        "default": true,
        "description": "Extract table headers"
      },
      "merge_cells": {
        "type": "boolean",
        "default": true,
        "description": "Handle merged cells"
      },
      "min_rows": {
        "type": "integer",
        "default": 2,
        "minimum": 1,
        "description": "Minimum rows for table"
      },
      "enabled": {
        "type": "boolean",
        "default": true,
        "description": "Enable this extractor"
      }
    },
    "required": []
  },
  "YAKEExtractor": {
    "type": "YAKEExtractor",
    "title": "Keyword Extractor Configuration",
    "description": "",
    "properties": {
      "extractor_type": {
        "type": "string",
        "const": "keyword",
        "description": "Extractor type discriminator"
      },
      "algorithm": {
        "type": "string",
        "enum": [
          "rake",
          "yake",
          "tfidf",
          "textrank"
        ],
        "default": "rake",
        "description": "Extraction algorithm"
      },
      "max_keywords": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 100,
        "description": "Maximum keywords to extract"
      },
      "min_length": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum word length for keywords"
      },
      "max_length": {
        "type": "integer",
        "default": 4,
        "minimum": 1,
        "description": "Maximum word length for keywords"
      },
      "min_frequency": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "description": "Minimum frequency for keywords"
      },
      "stop_words": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Custom stop words"
      },
      "language": {
        "type": "string",
        "default": "en",
        "description": "Language for YAKE algorithm"
      },
      "max_ngram_size": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 5,
        "description": "Maximum n-gram size for YAKE"
      },
      "deduplication_threshold": {
        "type": "number",
        "default": 0.9,
        "minimum": 0,
        "maximum": 1,
        "description": "Deduplication threshold for YAKE"
      }
    },
    "required": []
  }
}
