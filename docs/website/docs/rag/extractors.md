---
title: Extractors Reference
sidebar_position: 4
---

# Extractors Reference

Extractors enrich document chunks with metadata, enabling filtered searches and better retrieval. They run after parsing and add structured data to each chunk.

## Quick Start

Extractors are defined in `data_processing_strategies`:

```yaml
rag:
  data_processing_strategies:
    - name: my_processor
      parsers:
        - type: PDFParser_LlamaIndex
          config:
            chunk_size: 1000
      extractors:
        - type: EntityExtractor
          config:
            entity_types: [PERSON, ORG, DATE]
        - type: KeywordExtractor
          config:
            max_keywords: 10
```

### Common Extractor Properties

| Property | Required | Description |
|----------|----------|-------------|
| `type` | Yes | Extractor type (e.g., `EntityExtractor`) |
| `config` | No | Extractor-specific configuration |
| `priority` | No | Execution order (lower numbers run first) |
| `file_include_patterns` | No | Glob patterns for files to apply to |
| `condition` | No | Condition expression for when to run |

---

## EntityExtractor

Extracts named entities using NER models with regex fallback.

**Extracts:** People, organizations, dates, emails, phone numbers, URLs, products

```yaml
- type: EntityExtractor
  config:
    entity_types: [PERSON, ORG, DATE, EMAIL, PHONE]
    use_fallback: true
    confidence_threshold: 0.7
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `model` | string | `en_core_web_sm` | spaCy NER model |
| `entity_types` | array | See below | Entity types to extract |
| `use_fallback` | boolean | `true` | Use regex fallback |
| `min_entity_length` | integer | 2 | Minimum entity length |
| `merge_entities` | boolean | `true` | Merge adjacent entities |
| `confidence_threshold` | number | 0.7 | Minimum confidence (0-1) |

**Supported Entity Types:**
`PERSON`, `ORG`, `GPE`, `DATE`, `TIME`, `MONEY`, `EMAIL`, `PHONE`, `URL`, `LAW`, `PERCENT`, `PRODUCT`, `EVENT`, `VERSION`, `FAC`, `LOC`

---

## KeywordExtractor

Extracts important keywords using various algorithms.

**Extracts:** Key terms, phrases, n-grams

```yaml
- type: KeywordExtractor
  config:
    algorithm: rake
    max_keywords: 10
    language: en
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `algorithm` | string | `rake` | `rake`, `yake`, `tfidf`, `textrank` |
| `max_keywords` | integer | 10 | Maximum keywords (1-100) |
| `min_length` | integer | 1 | Minimum word length |
| `max_length` | integer | 4 | Maximum word length |
| `min_frequency` | integer | 1 | Minimum frequency |
| `stop_words` | array | - | Custom stop words |
| `language` | string | `en` | Language for YAKE |
| `max_ngram_size` | integer | 3 | Max n-gram size for YAKE |
| `deduplication_threshold` | number | 0.9 | Dedup threshold for YAKE |

---

## DateTimeExtractor

Extracts dates, times, and durations with fuzzy parsing.

**Extracts:** Dates, times, relative expressions, durations

```yaml
- type: DateTimeExtractor
  config:
    fuzzy_parsing: true
    extract_relative: true
    extract_times: true
    default_timezone: UTC
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `fuzzy_parsing` | boolean | `true` | Enable fuzzy date parsing |
| `extract_relative` | boolean | `true` | Extract relative dates |
| `extract_times` | boolean | `true` | Extract time expressions |
| `extract_durations` | boolean | `true` | Extract durations |
| `default_timezone` | string | `UTC` | Default timezone |
| `date_format` | string | `ISO` | Output date format |
| `prefer_dates_from` | string | `current` | `past`, `future`, `current` |

---

## HeadingExtractor

Extracts document headings and builds outline structure.

**Extracts:** Headings, hierarchy, document outline

```yaml
- type: HeadingExtractor
  config:
    max_level: 6
    include_hierarchy: true
    extract_outline: true
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_level` | integer | 6 | Maximum heading level (1-6) |
| `include_hierarchy` | boolean | `true` | Include hierarchy structure |
| `extract_outline` | boolean | `true` | Generate document outline |
| `min_heading_length` | integer | 3 | Minimum heading length |
| `enabled` | boolean | `true` | Enable extractor |

---

## LinkExtractor

Extracts URLs, emails, and domain information.

**Extracts:** URLs, email addresses, domains

```yaml
- type: LinkExtractor
  config:
    extract_urls: true
    extract_emails: true
    extract_domains: true
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extract_urls` | boolean | `true` | Extract URLs |
| `extract_emails` | boolean | `true` | Extract email addresses |
| `extract_domains` | boolean | `true` | Extract unique domains |
| `validate_urls` | boolean | `false` | Validate URL format |
| `resolve_redirects` | boolean | `false` | Resolve URL redirects |

---

## PathExtractor

Extracts file paths, URLs, and S3 paths.

**Extracts:** File paths, URL paths, cloud storage paths

```yaml
- type: PathExtractor
  config:
    extract_file_paths: true
    extract_s3_paths: true
    normalize_paths: true
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `extract_file_paths` | boolean | `true` | Extract file paths |
| `extract_urls` | boolean | `true` | Extract URL paths |
| `extract_s3_paths` | boolean | `true` | Extract S3 paths |
| `validate_paths` | boolean | `false` | Validate path existence |
| `normalize_paths` | boolean | `true` | Normalize path formats |

---

## PatternExtractor

Extracts data using predefined or custom regex patterns.

**Extracts:** Emails, phones, IPs, SSNs, credit cards, versions, custom patterns

```yaml
- type: PatternExtractor
  config:
    predefined_patterns:
      - email
      - phone
      - ip_address
      - version
    custom_patterns:
      - name: order_id
        pattern: "ORD-[0-9]{6}"
        description: "Order identifier"
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `predefined_patterns` | array | `[]` | Built-in patterns to use |
| `custom_patterns` | array | `[]` | Custom regex patterns |
| `case_sensitive` | boolean | `false` | Case-sensitive matching |
| `return_positions` | boolean | `false` | Return match positions |
| `include_context` | boolean | `false` | Include surrounding context |
| `max_matches_per_pattern` | integer | 100 | Max matches per pattern |
| `deduplicate_matches` | boolean | `true` | Remove duplicates |

**Predefined Patterns:**
`email`, `phone`, `url`, `ip`, `ip_address`, `ssn`, `credit_card`, `zip_code`, `file_path`, `version`, `date`

---

## ContentStatisticsExtractor

Calculates readability scores, vocabulary stats, and text structure.

**Extracts:** Word count, readability scores, vocabulary metrics

```yaml
- type: ContentStatisticsExtractor
  config:
    include_readability: true
    include_vocabulary: true
    include_structure: true
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `include_readability` | boolean | `true` | Calculate readability scores |
| `include_vocabulary` | boolean | `true` | Analyze vocabulary |
| `include_structure` | boolean | `true` | Analyze text structure |
| `include_sentiment_indicators` | boolean | `false` | Include sentiment indicators |

---

## SummaryExtractor

Generates extractive summaries using text ranking algorithms.

**Extracts:** Summary sentences, key phrases, text statistics

```yaml
- type: SummaryExtractor
  config:
    summary_sentences: 3
    algorithm: textrank
    include_key_phrases: true
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `summary_sentences` | integer | 3 | Number of sentences (1-10) |
| `algorithm` | string | `textrank` | `textrank`, `lsa`, `luhn`, `lexrank` |
| `include_key_phrases` | boolean | `true` | Extract key phrases |
| `include_statistics` | boolean | `true` | Include text statistics |
| `min_sentence_length` | integer | 10 | Minimum sentence length |
| `max_sentence_length` | integer | 500 | Maximum sentence length |

---

## TableExtractor

Extracts tabular data from documents.

**Extracts:** Tables, headers, cell data

```yaml
- type: TableExtractor
  config:
    output_format: dict
    extract_headers: true
    merge_cells: true
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `output_format` | string | `dict` | `dict`, `list`, `csv`, `markdown` |
| `extract_headers` | boolean | `true` | Extract table headers |
| `merge_cells` | boolean | `true` | Handle merged cells |
| `min_rows` | integer | 2 | Minimum rows for table |

---

## YAKEExtractor

YAKE (Yet Another Keyword Extractor) - unsupervised keyword extraction.

**Extracts:** Keywords using statistical features

```yaml
- type: YAKEExtractor
  config:
    max_keywords: 10
    language: en
    max_ngram_size: 3
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_keywords` | integer | 10 | Maximum keywords (1-100) |
| `language` | string | `en` | Language code |
| `max_ngram_size` | integer | 3 | Max n-gram size (1-5) |
| `deduplication_threshold` | number | 0.9 | Dedup threshold (0-1) |

---

## RAKEExtractor

RAKE (Rapid Automatic Keyword Extraction) algorithm.

**Extracts:** Keywords based on word co-occurrence

```yaml
- type: RAKEExtractor
  config:
    max_keywords: 10
    min_length: 1
    max_length: 4
```

Use via `KeywordExtractor` with `algorithm: rake`.

---

## TFIDFExtractor

TF-IDF based keyword extraction.

**Extracts:** Keywords based on term frequency-inverse document frequency

```yaml
- type: TFIDFExtractor
  config:
    max_keywords: 10
    min_frequency: 1
```

Use via `KeywordExtractor` with `algorithm: tfidf`.

---

## Complete Example

Combine multiple extractors for rich metadata:

```yaml
extractors:
  # High priority - entities first
  - type: EntityExtractor
    priority: 100
    config:
      entity_types: [PERSON, ORG, DATE, PRODUCT]
      use_fallback: true

  # Keywords for searchability
  - type: KeywordExtractor
    priority: 90
    config:
      algorithm: yake
      max_keywords: 15

  # Statistics for filtering
  - type: ContentStatisticsExtractor
    priority: 80
    config:
      include_readability: true

  # Patterns for specific data
  - type: PatternExtractor
    priority: 70
    file_include_patterns: ["*.pdf"]
    config:
      predefined_patterns: [email, phone, date]
      custom_patterns:
        - name: case_number
          pattern: "CASE-[A-Z]{2}-[0-9]{6}"
```

## Using Extracted Metadata

Query with metadata filters:

```bash
# Filter by entity
lf rag query --database main_db --filter "entities.ORG:Acme Corp" "contracts"

# Filter by keyword
lf rag query --database main_db --filter "keywords:merger" "recent news"

# Filter by date
lf rag query --database main_db --filter "dates:2024" "quarterly reports"
```

## Next Steps

- [Databases Reference](./databases.md) - Configure vector stores
- [Retrieval Strategies](./retrieval-strategies.md) - Configure retrieval
- [RAG Guide](./index.md) - Full RAG overview
