export type PrimitiveType =
  | 'integer'
  | 'number'
  | 'string'
  | 'boolean'
  | 'array'
  | 'object'

export type SchemaField = {
  type: PrimitiveType
  title?: string
  description?: string
  default?: unknown
  minimum?: number
  maximum?: number
  enum?: string[]
  items?: { type: PrimitiveType; properties?: Record<string, SchemaField> }
}

export type ExtractorSchema = {
  title: string
  description?: string
  properties: Record<string, SchemaField>
  required?: string[]
}

export const EXTRACTOR_SCHEMAS: Record<string, ExtractorSchema> = {
  KeywordExtractor: {
    title: 'Keyword Extractor Configuration',
    description:
      'Extract keywords using algorithms like RAKE/YAKE/TF-IDF/TextRank',
    properties: {
      extractor_type: {
        type: 'string',
        default: 'keyword',
        description: 'Extractor type discriminator',
      },
      algorithm: {
        type: 'string',
        enum: ['rake', 'yake', 'tfidf', 'textrank'],
        default: 'rake',
        description: 'Extraction algorithm',
      },
      max_keywords: { type: 'integer', default: 10, minimum: 1, maximum: 100 },
      min_length: { type: 'integer', default: 1, minimum: 1 },
      max_length: { type: 'integer', default: 4, minimum: 1 },
      min_frequency: { type: 'integer', default: 1, minimum: 1 },
      stop_words: {
        type: 'array',
        items: { type: 'string' },
        description: 'Custom stop words',
      },
      language: { type: 'string', default: 'en' },
      max_ngram_size: { type: 'integer', default: 3, minimum: 1, maximum: 5 },
      deduplication_threshold: {
        type: 'number',
        default: 0.9,
        minimum: 0.0,
        maximum: 1.0,
      },
    },
  },
  EntityExtractor: {
    title: 'Entity Extractor Configuration',
    description: 'Named entity extraction with model settings',
    properties: {
      model: { type: 'string', default: 'en_core_web_sm' },
      entity_types: {
        type: 'array',
        items: {
          type: 'string',
        },
        description: 'Entity types to extract',
      },
      use_fallback: { type: 'boolean', default: true },
      min_entity_length: { type: 'integer', default: 2, minimum: 1 },
      merge_entities: { type: 'boolean', default: true },
      confidence_threshold: {
        type: 'number',
        default: 0.7,
        minimum: 0.0,
        maximum: 1.0,
      },
    },
  },
  DateTimeExtractor: {
    title: 'DateTime Extractor Configuration',
    description: 'Extract absolute/relative dates, times and durations',
    properties: {
      fuzzy_parsing: { type: 'boolean', default: true },
      extract_relative: { type: 'boolean', default: true },
      extract_times: { type: 'boolean', default: true },
      extract_durations: { type: 'boolean', default: true },
      default_timezone: { type: 'string', default: 'UTC' },
      date_format: { type: 'string', default: 'ISO' },
      prefer_dates_from: {
        type: 'string',
        enum: ['past', 'future', 'current'],
        default: 'current',
      },
    },
  },
  HeadingExtractor: {
    title: 'Heading Extractor Configuration',
    description: 'Extract document headings with optional outline',
    properties: {
      max_level: { type: 'integer', default: 6, minimum: 1, maximum: 6 },
      include_hierarchy: { type: 'boolean', default: true },
      extract_outline: { type: 'boolean', default: true },
      min_heading_length: { type: 'integer', default: 3, minimum: 1 },
      enabled: { type: 'boolean', default: true },
    },
  },
  LinkExtractor: {
    title: 'Link Extractor Configuration',
    description: 'Extract URLs, emails and domains',
    properties: {
      extract_urls: { type: 'boolean', default: true },
      extract_emails: { type: 'boolean', default: true },
      extract_domains: { type: 'boolean', default: true },
      validate_urls: { type: 'boolean', default: false },
      resolve_redirects: { type: 'boolean', default: false },
      enabled: { type: 'boolean', default: true },
    },
  },
  PathExtractor: {
    title: 'Path Extractor Configuration',
    description: 'Extract file, URL, and S3 paths',
    properties: {
      extract_file_paths: { type: 'boolean', default: true },
      extract_urls: { type: 'boolean', default: true },
      extract_s3_paths: { type: 'boolean', default: true },
      validate_paths: { type: 'boolean', default: false },
      normalize_paths: { type: 'boolean', default: true },
      enabled: { type: 'boolean', default: true },
    },
  },
  PatternExtractor: {
    title: 'Pattern Extractor Configuration',
    description: 'Extract predefined and custom regex patterns',
    properties: {
      predefined_patterns: {
        type: 'array',
        items: { type: 'string' },
        description: 'Built-in pattern names',
      },
      custom_patterns: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            name: { type: 'string' },
            pattern: { type: 'string' },
            description: { type: 'string' },
          },
        },
      },
      case_sensitive: { type: 'boolean', default: false },
      return_positions: { type: 'boolean', default: false },
      include_context: { type: 'boolean', default: false },
      max_matches_per_pattern: { type: 'integer', default: 100, minimum: 1 },
      deduplicate_matches: { type: 'boolean', default: true },
    },
  },
  StatisticsExtractor: {
    title: 'Statistics Extractor Configuration',
    description: 'Readability, vocabulary, structure, sentiment indicators',
    properties: {
      include_readability: { type: 'boolean', default: true },
      include_vocabulary: { type: 'boolean', default: true },
      include_structure: { type: 'boolean', default: true },
      include_sentiment: { type: 'boolean', default: false },
      include_sentiment_indicators: { type: 'boolean', default: false },
      include_language: { type: 'boolean', default: true },
    },
  },
  SummaryExtractor: {
    title: 'Summary Extractor Configuration',
    description: 'Unsupervised extractive summarization with options',
    properties: {
      summary_sentences: {
        type: 'integer',
        default: 3,
        minimum: 1,
        maximum: 10,
      },
      algorithm: {
        type: 'string',
        enum: ['textrank', 'lsa', 'luhn', 'lexrank'],
        default: 'textrank',
      },
      include_key_phrases: { type: 'boolean', default: true },
      include_statistics: { type: 'boolean', default: true },
      min_sentence_length: { type: 'integer', default: 10, minimum: 1 },
      max_sentence_length: { type: 'integer', default: 500, minimum: 10 },
    },
  },
  SentimentExtractor: {
    title: 'Sentiment Extractor Configuration',
    description: 'Sentiment analysis with optional business tone',
    properties: {
      analyze_business_tone: { type: 'boolean', default: false },
      extract_confidence: { type: 'boolean', default: true },
      categories: { type: 'array', items: { type: 'string' } },
    },
  },
  YAKEExtractor: {
    title: 'YAKE Extractor Configuration',
    description: 'Keyword extraction (YAKE) settings',
    properties: {
      extractor_type: { type: 'string', default: 'yake' },
      max_keywords: { type: 'integer', default: 10, minimum: 1, maximum: 100 },
      language: { type: 'string', default: 'en' },
      max_ngram_size: { type: 'integer', default: 3, minimum: 1, maximum: 5 },
      deduplication_threshold: {
        type: 'number',
        default: 0.9,
        minimum: 0.0,
        maximum: 1.0,
      },
    },
  },
  ContentStatisticsExtractor: {
    title: 'Content Statistics Extractor Configuration',
    description: 'Text content statistic indicators',
    properties: {
      include_readability: { type: 'boolean', default: true },
      include_vocabulary: { type: 'boolean', default: true },
      include_structure: { type: 'boolean', default: true },
      include_sentiment_indicators: { type: 'boolean', default: false },
    },
  },
  TableExtractor: {
    title: 'Table Extractor Configuration',
    description: 'Extract tabular data from documents',
    properties: {
      output_format: {
        type: 'string',
        enum: ['dict', 'list', 'csv', 'markdown'],
        default: 'dict',
      },
      extract_headers: { type: 'boolean', default: true },
      merge_cells: { type: 'boolean', default: true },
      min_rows: { type: 'integer', default: 2, minimum: 1 },
      enabled: { type: 'boolean', default: true },
    },
  },
}

export const ORDERED_EXTRACTOR_TYPES: string[] = [
  'ContentStatisticsExtractor',
  'EntityExtractor',
  'KeywordExtractor',
  'TableExtractor',
  'DateTimeExtractor',
  'PatternExtractor',
  'HeadingExtractor',
  'LinkExtractor',
  'SummaryExtractor',
  'StatisticsExtractor',
  'YAKEExtractor',
  'SentimentExtractor',
]

export function getDefaultConfigForExtractor(
  type: string
): Record<string, unknown> {
  const schema = EXTRACTOR_SCHEMAS[type]
  if (!schema) return {}
  const cfg: Record<string, unknown> = {}
  Object.entries(schema.properties).forEach(([key, field]) => {
    if (typeof field.default !== 'undefined') {
      cfg[key] = field.default
    } else if (field.type === 'array') {
      cfg[key] = []
    } else if (field.type === 'boolean') {
      cfg[key] = false
    } else if (field.type === 'integer' || field.type === 'number') {
      cfg[key] = field.minimum ?? 0
    } else if (field.type === 'object') {
      cfg[key] = {}
    } else {
      cfg[key] = ''
    }
  })
  return cfg
}
