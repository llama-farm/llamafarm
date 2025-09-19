export type PrimitiveType =
  | 'integer'
  | 'number'
  | 'string'
  | 'boolean'
  | 'array'

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

export type ParserSchema = {
  title: string
  description?: string
  properties: Record<string, SchemaField>
  required?: string[]
}

// Mapping between visible parser names used in the UI and the parser schemas
// derived from rag/schema.yaml. This is a curated subset focused on the
// parsers currently surfaced in StrategyView defaults. Additional parsers can
// be added here following the same structure.
export const PARSER_SCHEMAS: Record<string, ParserSchema> = {
  // Excel (LlamaIndex)
  ExcelParser_LlamaIndex: {
    title: 'Excel Parser (LlamaIndex) Configuration',
    description:
      'Excel parser using LlamaIndex with Pandas backend for advanced processing',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        maximum: 50000,
        description: 'Number of rows per chunk',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['rows', 'semantic', 'full'],
        default: 'rows',
        description: 'Chunking strategy',
      },
      sheets: {
        type: 'array',
        items: { type: 'string' },
        nullable: true,
        description: 'Specific sheets to parse (null for all)',
      },
      combine_sheets: {
        type: 'boolean',
        default: false,
        description: 'Combine all sheets into one document',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract metadata from Excel',
      },
      extract_formulas: {
        type: 'boolean',
        default: false,
        description: 'Extract formulas instead of values',
      },
      header_row: {
        type: 'integer',
        default: 0,
        minimum: 0,
        description: 'Row index for headers',
      },
      skiprows: {
        type: 'integer',
        minimum: 0,
        description: 'Number of rows to skip',
      },
      na_values: {
        type: 'array',
        items: { type: 'string' },
        default: ['', 'NA', 'N/A', 'null', 'None'],
        description: 'Values to treat as missing',
      },
    },
  },

  // Text (LlamaIndex)
  TextParser_LlamaIndex: {
    title: 'Text Parser (LlamaIndex) Configuration',
    description:
      'Advanced text parser using LlamaIndex with semantic splitting and code support',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        maximum: 50000,
        description: 'Chunk size in characters',
      },
      chunk_overlap: {
        type: 'integer',
        default: 100,
        minimum: 0,
        maximum: 5000,
        description: 'Overlap between chunks',
      },
      chunk_strategy: {
        type: 'string',
        enum: [
          'characters',
          'sentences',
          'paragraphs',
          'tokens',
          'semantic',
          'code',
        ],
        default: 'semantic',
        description: 'Advanced chunking strategy',
      },
      encoding: {
        type: 'string',
        default: 'utf-8',
        description: 'Text encoding',
      },
      clean_text: {
        type: 'boolean',
        default: true,
        description: 'Clean extracted text',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract comprehensive file and content metadata',
      },
      semantic_buffer_size: {
        type: 'integer',
        default: 1,
        minimum: 1,
        maximum: 10,
        description: 'Buffer size for semantic chunking',
      },
      semantic_breakpoint_percentile_threshold: {
        type: 'integer',
        default: 95,
        minimum: 50,
        maximum: 99,
        description: 'Percentile threshold for semantic breakpoints',
      },
      token_model: {
        type: 'string',
        default: 'gpt-3.5-turbo',
        description: 'Tokenizer model for token-based chunking',
      },
      preserve_code_structure: {
        type: 'boolean',
        default: true,
        description: 'Preserve code syntax and structure for code files',
      },
      detect_language: {
        type: 'boolean',
        default: true,
        description: 'Auto-detect programming language for code files',
      },
      include_prev_next_rel: {
        type: 'boolean',
        default: true,
        description: 'Include relationships between chunks for better context',
      },
    },
  },

  // Markdown (LlamaIndex)
  MarkdownParser_LlamaIndex: {
    title: 'Markdown Parser (LlamaIndex) Configuration',
    description:
      'Advanced markdown parser using LlamaIndex with semantic chunking',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        maximum: 50000,
        description: 'Chunk size in characters',
      },
      chunk_overlap: {
        type: 'integer',
        default: 100,
        minimum: 0,
        maximum: 5000,
        description: 'Overlap between chunks',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['headings', 'paragraphs', 'sentences', 'semantic'],
        default: 'headings',
        description: 'Chunking strategy for markdown',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract frontmatter metadata',
      },
      extract_code_blocks: {
        type: 'boolean',
        default: true,
        description: 'Extract code blocks separately',
      },
      extract_tables: {
        type: 'boolean',
        default: true,
        description: 'Extract markdown tables',
      },
      extract_links: {
        type: 'boolean',
        default: true,
        description: 'Extract links and references',
      },
      preserve_structure: {
        type: 'boolean',
        default: true,
        description: 'Preserve heading hierarchy',
      },
    },
  },

  // Docx (python-docx)
  DocxParser_PythonDocx: {
    title: 'DOCX Parser (python-docx) Configuration',
    description: 'Word document parser using python-docx library',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        description: 'Chunk size in characters',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['paragraphs', 'sentences', 'characters'],
        default: 'paragraphs',
        description: 'Chunking strategy',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract document metadata',
      },
      extract_tables: {
        type: 'boolean',
        default: true,
        description: 'Extract tables using python-docx',
      },
      extract_headers: {
        type: 'boolean',
        default: true,
        description: 'Extract headers',
      },
      extract_footers: {
        type: 'boolean',
        default: false,
        description: 'Extract footers',
      },
      extract_comments: {
        type: 'boolean',
        default: false,
        description: 'Extract comments',
      },
    },
  },

  // CSV (LlamaIndex)
  CSVParser_LlamaIndex: {
    title: 'CSV Parser (LlamaIndex) Configuration',
    description:
      'CSV parser using LlamaIndex with Pandas backend for advanced processing',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        maximum: 50000,
        description: 'Number of rows per chunk',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['rows', 'semantic', 'full'],
        default: 'rows',
        description: 'Chunking strategy',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract metadata from CSV',
      },
      combine_fields: {
        type: 'boolean',
        default: true,
        description: 'Combine fields into text content',
      },
      skiprows: {
        type: 'integer',
        minimum: 0,
        description: 'Number of rows to skip at beginning',
      },
      na_values: {
        type: 'array',
        items: { type: 'string' },
        default: ['', 'NA', 'N/A', 'null', 'None'],
        description: 'Values to treat as missing',
      },
    },
  },

  // Excel (OpenPyXL)
  ExcelParser_OpenPyXL: {
    title: 'Excel Parser (OpenPyXL) Configuration',
    description:
      'Excel parser using OpenPyXL for XLSX files with formula support',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        description: 'Number of rows per chunk',
      },
      extract_formulas: {
        type: 'boolean',
        default: false,
        description: 'Extract cell formulas using OpenPyXL',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract workbook metadata',
      },
      sheets: {
        type: 'array',
        items: { type: 'string' },
        nullable: true,
        description: 'Specific sheets to process (null = all)',
      },
      data_only: {
        type: 'boolean',
        default: true,
        description: 'Extract values instead of formulas',
      },
    },
  },

  // Excel (Pandas)
  ExcelParser_Pandas: {
    title: 'Excel Parser (Pandas) Configuration',
    description: 'Excel parser using Pandas with data analysis capabilities',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        description: 'Number of rows per chunk',
      },
      sheets: {
        type: 'array',
        items: { type: 'string' },
        nullable: true,
        description: 'Specific sheets to process (null = all)',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract data statistics',
      },
      skiprows: {
        type: 'integer',
        nullable: true,
        description: 'Rows to skip at beginning',
      },
      na_values: {
        type: 'array',
        items: { type: 'string' },
        default: ['', 'NA', 'N/A', 'null', 'None'],
        description: 'Values to treat as NaN',
      },
    },
  },

  // CSV (Pandas)
  CSVParser_Pandas: {
    title: 'CSV Parser (Pandas) Configuration',
    description:
      'Advanced CSV parser using Pandas with data analysis capabilities',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        description: 'Number of rows per chunk',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['rows', 'semantic', 'full'],
        default: 'rows',
        description: 'How to chunk the CSV data',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract data statistics and metadata',
      },
      encoding: {
        type: 'string',
        default: 'utf-8',
        description: 'File encoding',
      },
      delimiter: {
        type: 'string',
        default: ',',
        description: 'CSV delimiter',
      },
      na_values: {
        type: 'array',
        items: { type: 'string' },
        default: ['', 'NA', 'N/A', 'null', 'None'],
        description: 'Values to treat as NaN',
      },
    },
  },

  // CSV (Python)
  CSVParser_Python: {
    title: 'CSV Parser (Python) Configuration',
    description: 'Simple CSV parser using native Python csv module',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        description: 'Number of rows per chunk',
      },
      encoding: {
        type: 'string',
        default: 'utf-8',
        description: 'File encoding',
      },
      delimiter: {
        type: 'string',
        default: ',',
        description: 'CSV delimiter',
      },
      quotechar: {
        type: 'string',
        default: '"',
        description: 'Quote character',
      },
    },
  },

  // PDF (LlamaIndex)
  PDFParser_LlamaIndex: {
    title: 'PDF Parser (LlamaIndex) Configuration',
    description:
      'Advanced PDF parser using LlamaIndex with multiple fallback strategies',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        maximum: 50000,
        description: 'Chunk size in characters',
      },
      chunk_overlap: {
        type: 'integer',
        default: 100,
        minimum: 0,
        maximum: 5000,
        description: 'Overlap between chunks',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['sentences', 'paragraphs', 'pages', 'semantic'],
        default: 'sentences',
        description: 'Chunking strategy for PDF content',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract PDF metadata',
      },
      extract_images: {
        type: 'boolean',
        default: false,
        description: 'Extract images from PDF',
      },
      extract_tables: {
        type: 'boolean',
        default: true,
        description: 'Extract tables from PDF',
      },
    },
  },

  // PDF (PyPDF2)
  PDFParser_PyPDF2: {
    title: 'PDF Parser (PyPDF2) Configuration',
    description:
      'Enhanced PDF parser using PyPDF2 with comprehensive capabilities',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        maximum: 50000,
        description: 'Chunk size in characters',
      },
      chunk_overlap: {
        type: 'integer',
        default: 100,
        minimum: 0,
        maximum: 5000,
        description: 'Overlap between chunks in characters',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['paragraphs', 'sentences', 'characters'],
        default: 'paragraphs',
        description: 'Chunking strategy',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract PDF metadata using PyPDF2',
      },
      preserve_layout: {
        type: 'boolean',
        default: true,
        description: 'Use layout-preserving extraction mode',
      },
      extract_page_info: {
        type: 'boolean',
        default: true,
        description: 'Extract page numbers and rotation info',
      },
      clean_text: {
        type: 'boolean',
        default: true,
        description: 'Clean extracted text',
      },
    },
  },

  // Docx (LlamaIndex)
  DocxParser_LlamaIndex: {
    title: 'DOCX Parser (LlamaIndex) Configuration',
    description: 'Advanced DOCX parser using LlamaIndex with enhanced chunking',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        maximum: 50000,
        description: 'Chunk size in characters',
      },
      chunk_overlap: {
        type: 'integer',
        default: 100,
        minimum: 0,
        maximum: 5000,
        description: 'Overlap between chunks',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['paragraphs', 'sentences', 'semantic'],
        default: 'paragraphs',
        description: 'Chunking strategy',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract document metadata',
      },
      extract_tables: {
        type: 'boolean',
        default: true,
        description: 'Extract tables from document',
      },
      extract_images: {
        type: 'boolean',
        default: false,
        description: 'Extract images from document',
      },
      preserve_formatting: {
        type: 'boolean',
        default: true,
        description: 'Preserve text formatting',
      },
    },
  },

  // Markdown (Python)
  MarkdownParser_Python: {
    title: 'Markdown Parser (Python) Configuration',
    description: 'Markdown parser using native Python with regex parsing',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        description: 'Chunk size in characters',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['sections', 'paragraphs', 'characters'],
        default: 'sections',
        description: 'Chunking strategy - sections uses markdown headers',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract YAML frontmatter',
      },
      extract_code_blocks: {
        type: 'boolean',
        default: true,
        description: 'Extract code blocks',
      },
      extract_links: {
        type: 'boolean',
        default: true,
        description: 'Extract markdown links',
      },
    },
  },

  // Text (Python)
  TextParser_Python: {
    title: 'Text Parser (Python) Configuration',
    description: 'Text parser using native Python with encoding detection',
    properties: {
      chunk_size: {
        type: 'integer',
        default: 1000,
        minimum: 100,
        description: 'Chunk size in characters',
      },
      chunk_overlap: {
        type: 'integer',
        default: 100,
        minimum: 0,
        description: 'Overlap between chunks',
      },
      chunk_strategy: {
        type: 'string',
        enum: ['sentences', 'paragraphs', 'characters'],
        default: 'sentences',
        description: 'Text chunking strategy',
      },
      encoding: {
        type: 'string',
        default: 'utf-8',
        description: 'Text encoding (utf-8 or auto-detect)',
      },
      clean_text: {
        type: 'boolean',
        default: true,
        description: 'Remove excessive whitespace',
      },
      extract_metadata: {
        type: 'boolean',
        default: true,
        description: 'Extract file statistics',
      },
    },
  },
}

export const ORDERED_PARSER_TYPES: string[] = [
  'PDFParser_LlamaIndex',
  'PDFParser_PyPDF2',
  'DocxParser_LlamaIndex',
  'DocxParser_PythonDocx',
  'MarkdownParser_Python',
  'MarkdownParser_LlamaIndex',
  'CSVParser_Pandas',
  'CSVParser_LlamaIndex',
  'ExcelParser_Pandas',
  'ExcelParser_OpenPyXL',
  'TextParser_Python',
  'TextParser_LlamaIndex',
  'ExcelParser_LlamaIndex',
]

export function getDefaultConfigForParser(
  parserType: string
): Record<string, unknown> {
  const schema = PARSER_SCHEMAS[parserType]
  if (!schema) return {}
  const cfg: Record<string, unknown> = {}
  Object.entries(schema.properties).forEach(([key, field]) => {
    if (typeof field.default !== 'undefined') {
      cfg[key] = field.default
    } else if (field.type === 'array') {
      // Default to empty array unless nullable implies null by default
      cfg[key] = field.nullable ? null : []
    } else if (field.type === 'boolean') {
      cfg[key] = false
    } else if (field.type === 'integer' || field.type === 'number') {
      // If nullable, honor null default; otherwise use minimum or 0
      cfg[key] = field.nullable ? null : (field.minimum ?? 0)
    } else {
      cfg[key] = ''
    }
  })
  return cfg
}
