---
title: Data sources
sidebar_label: Data sources
slug: /rag/data-sources
toc_min_heading_level: 2
toc_max_heading_level: 3
---

Connecting to files, databases, and APIs.

## CSV processing

### CSV commands with sample data

```bash
# 1. Test CSV parsing with sample data
uv run python cli.py --config config_examples/basic_config.json \
  test --test-file samples/small_sample.csv

# 2. Ingest CSV data using specific configuration
uv run python cli.py --config config_examples/basic_config.json \
  ingest samples/small_sample.csv

# 3. Search the ingested CSV data
uv run python cli.py --config config_examples/basic_config.json \
  search "login problems"

# 4. Check collection info
uv run python cli.py --config config_examples/basic_config.json info
```

### Custom CSV configuration

```bash
uv run python cli.py --config config_examples/custom_csv_config.json \
  ingest your_custom_data.csv
```

## PDF processing

### Single PDF

```bash
uv run python cli.py --config config_examples/pdf_config.json \
  test --test-file samples/test_document.pdf
uv run python cli.py --config config_examples/pdf_config.json \
  ingest samples/test_document.pdf
uv run python cli.py --config config_examples/pdf_config.json \
  search "specific topic from PDF"
```

### Multiple PDFs

```bash
uv run python cli.py --config config_examples/pdf_config.json \
  ingest samples/document1.pdf
uv run python cli.py --config config_examples/pdf_config.json \
  ingest samples/document2.pdf
uv run python cli.py --config config_examples/pdf_config.json \
  ingest samples/document3.pdf
```

### PDF directory

```bash
for pdf in samples/pdfs/*.pdf; do
  echo "Processing: $pdf"
  uv run python cli.py --config config_examples/pdf_config.json ingest "$pdf"
done

find samples/ -name "*.pdf" -exec uv run python cli.py \
  --config config_examples/pdf_config.json ingest {} \;
```

### Page-by-page processing

```bash
uv run python cli.py --config config_examples/pdf_separate_pages_config.json \
  ingest samples/multi_page_document.pdf
uv run python cli.py --config config_examples/pdf_separate_pages_config.json \
  search "chapter introduction"
```
