---
title: Local only extractors
sidebar_label: Local only extractors
slug: /rag/local-only-extractors
toc_min_heading_level: 2
toc_max_heading_level: 3
---

The RAG system includes 5 local-only extractors that enhance documents with metadata without requiring external LLMs.

## Available extractors

- YAKE: Advanced keyword extraction considering position and context
- RAKE: Fast phrase extraction using stop words as delimiters
- TF-IDF: Term frequency analysis for finding unique terms
- Entities: Person, organization, date, email, phone extraction (spaCy + regex fallbacks)
- DateTime: Date, time, and relative date extraction
- Statistics: Readability metrics, vocabulary analysis, content statistics

## Extractor commands

```bash
# List all available extractors
uv run python cli.py extractors list --detailed

# Test an extractor on sample text
uv run python cli.py extractors test --extractor yake

# Test with your own text
uv run python cli.py extractors test --extractor entities --text "Contact John Doe at john@company.com"

# Test with a file
uv run python cli.py extractors test --extractor statistics --file samples/document.txt
```

## Using extractors during ingestion

```bash
# Apply extractors during CSV ingestion
uv run python cli.py ingest samples/small_sample.csv --extractors yake entities statistics

# Apply with custom configuration
uv run python cli.py ingest document.pdf --extractors rake entities \
  --extractor-config '{"rake": {"max_keywords": 20}, "entities": {"entity_types": ["PERSON", "ORG"]}}'

# Use configuration-based extractors (see config_examples/extractors_demo_config.json)
uv run python cli.py --config config_examples/extractors_demo_config.json ingest samples/document.pdf
```

## Extractor output example

```json
{
  "yake_keywords": [
    "machine learning",
    "artificial intelligence",
    "data analysis"
  ],
  "entities_person": ["John Smith", "Sarah Johnson"],
  "entities_email": ["contact@company.com"],
  "word_count": 1250,
  "reading_time_minutes": 6.25,
  "sentiment_classification": "positive",
  "flesch_reading_ease": 65.2
}
```
