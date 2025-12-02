---
title: Specialized ML Models
sidebar_position: 2
---

# Specialized ML Models

Beyond text generation, the Universal Runtime provides a comprehensive suite of specialized ML endpoints for document processing, text analysis, and anomaly detection. These endpoints run on the Universal Runtime server (port 11540).

## Quick Reference

| Capability | Endpoint | Use Case |
|-----------|----------|----------|
| [OCR](#ocr-text-extraction) | `POST /v1/ocr` | Extract text from images/PDFs |
| [Document Extraction](#document-extraction) | `POST /v1/documents/extract` | Extract structured data from forms |
| [Text Classification](#text-classification) | `POST /v1/classify` | Sentiment, spam detection, routing |
| [Named Entity Recognition](#named-entity-recognition-ner) | `POST /v1/ner` | Extract people, places, organizations |
| [Reranking](#reranking-cross-encoder) | `POST /v1/rerank` | Improve RAG retrieval accuracy |
| [Anomaly Detection](#anomaly-detection) | `POST /v1/anomaly/*` | Detect outliers in numeric/mixed data |

## Starting the Universal Runtime

```bash
# Start the runtime server
nx start universal-runtime

# Or with custom port
LF_RUNTIME_PORT=8080 nx start universal-runtime
```

The server runs on `http://localhost:11540` by default.

---

## OCR (Text Extraction)

Extract text from images and PDF documents using multiple OCR backends.

### Supported Backends

| Backend | Description | Best For |
|---------|-------------|----------|
| `surya` | Transformer-based, layout-aware (recommended) | Best accuracy, complex documents |
| `easyocr` | 80+ languages, widely used | Multilingual documents |
| `paddleocr` | Fast, production-optimized | Asian languages, speed |
| `tesseract` | Classic OCR, CPU-only | Simple documents, CPU-only environments |

### Basic Usage

```bash
# OCR with base64 image
curl -X POST http://localhost:11540/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "model": "surya",
    "images": ["'$(base64 -w0 document.png)'"],
    "languages": ["en"]
  }'
```

### PDF Processing Workflow

For multi-page documents, use the file upload workflow:

```bash
# 1. Upload PDF (auto-converts to images)
curl -X POST http://localhost:11540/v1/files \
  -F "file=@document.pdf" \
  -F "convert_pdf=true" \
  -F "pdf_dpi=150"

# Response: {"id": "file_abc123", "page_count": 5, ...}

# 2. Run OCR on all pages
curl -X POST http://localhost:11540/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "model": "surya",
    "file_id": "file_abc123",
    "languages": ["en"],
    "return_boxes": true
  }'
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "text": "Invoice #12345\nDate: 2024-01-15\nTotal: $1,234.56",
      "confidence": 0.95,
      "boxes": [
        {"x1": 10, "y1": 20, "x2": 150, "y2": 40, "text": "Invoice #12345", "confidence": 0.98}
      ]
    }
  ],
  "model": "surya",
  "usage": {"images_processed": 1}
}
```

---

## Document Extraction

Extract structured key-value pairs from forms, invoices, and receipts using vision-language models.

### Supported Models

| Model | Description |
|-------|-------------|
| `naver-clova-ix/donut-base-finetuned-cord-v2` | Receipt/invoice extraction (no OCR needed) |
| `naver-clova-ix/donut-base-finetuned-docvqa` | Document Q&A |
| `microsoft/layoutlmv3-base-finetuned-docvqa` | Document Q&A with layout understanding |

### Extract from Receipt

```bash
curl -X POST http://localhost:11540/v1/documents/extract \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "file_id": "file_abc123",
    "task": "extraction"
  }'
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "confidence": 0.92,
      "fields": [
        {"key": "store_name", "value": "Coffee Shop", "confidence": 0.95, "bbox": [10, 20, 100, 40]},
        {"key": "total", "value": "$15.99", "confidence": 0.98, "bbox": [10, 60, 80, 80]},
        {"key": "date", "value": "2024-01-15", "confidence": 0.94, "bbox": [10, 100, 100, 120]}
      ]
    }
  ]
}
```

### Document Q&A

Ask questions about document content:

```bash
curl -X POST http://localhost:11540/v1/documents/extract \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-clova-ix/donut-base-finetuned-docvqa",
    "file_id": "file_abc123",
    "prompts": ["What is the total amount?", "What is the invoice date?"],
    "task": "vqa"
  }'
```

---

## Text Classification

Classify text using any HuggingFace sequence classification model.

### Popular Models

| Model | Use Case |
|-------|----------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment analysis |
| `facebook/bart-large-mnli` | Zero-shot classification |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Social media sentiment |

### Basic Classification

```bash
curl -X POST http://localhost:11540/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "texts": [
      "I love this product!",
      "This is terrible and broken.",
      "It works okay I guess."
    ]
  }'
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {"index": 0, "label": "POSITIVE", "score": 0.9998, "all_scores": {"POSITIVE": 0.9998, "NEGATIVE": 0.0002}},
    {"index": 1, "label": "NEGATIVE", "score": 0.9995, "all_scores": {"POSITIVE": 0.0005, "NEGATIVE": 0.9995}},
    {"index": 2, "label": "POSITIVE", "score": 0.6234, "all_scores": {"POSITIVE": 0.6234, "NEGATIVE": 0.3766}}
  ],
  "model": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

---

## Named Entity Recognition (NER)

Extract named entities (people, organizations, locations) from text.

### Popular Models

| Model | Description |
|-------|-------------|
| `dslim/bert-base-NER` | English NER (PERSON/ORG/LOC/MISC) |
| `Jean-Baptiste/roberta-large-ner-english` | High-accuracy English NER |
| `xlm-roberta-large-finetuned-conll03-english` | Multilingual NER |

### Basic NER

```bash
curl -X POST http://localhost:11540/v1/ner \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dslim/bert-base-NER",
    "texts": [
      "John Smith works at Google in San Francisco.",
      "Apple CEO Tim Cook announced new products."
    ]
  }'
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "index": 0,
      "entities": [
        {"text": "John Smith", "label": "PER", "start": 0, "end": 10, "score": 0.99},
        {"text": "Google", "label": "ORG", "start": 20, "end": 26, "score": 0.98},
        {"text": "San Francisco", "label": "LOC", "start": 30, "end": 43, "score": 0.97}
      ]
    },
    {
      "index": 1,
      "entities": [
        {"text": "Apple", "label": "ORG", "start": 0, "end": 5, "score": 0.99},
        {"text": "Tim Cook", "label": "PER", "start": 10, "end": 18, "score": 0.98}
      ]
    }
  ]
}
```

---

## Reranking (Cross-Encoder)

Improve RAG retrieval accuracy by reranking candidate documents with a cross-encoder model.

### Why Rerank?

Cross-encoders are **significantly more accurate** than bi-encoder similarity (10-20% improvement) and **10-100x faster** than LLM-based reranking.

### Popular Models

| Model | Description |
|-------|-------------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast, general purpose |
| `BAAI/bge-reranker-v2-m3` | Multilingual, high accuracy |
| `cross-encoder/ms-marco-MiniLM-L-12-v2` | Higher accuracy, slower |

### Basic Reranking

```bash
curl -X POST http://localhost:11540/v1/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "query": "What are the clinical trial requirements?",
    "documents": [
      "Clinical trials must follow FDA regulations for safety.",
      "The weather in California is sunny.",
      "Phase 3 trials require at least 300 participants.",
      "Our company was founded in 2010."
    ],
    "top_k": 2,
    "return_documents": true
  }'
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {"index": 0, "relevance_score": 0.92, "document": "Clinical trials must follow FDA regulations..."},
    {"index": 2, "relevance_score": 0.87, "document": "Phase 3 trials require at least 300 participants."}
  ],
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
}
```

### Integration with RAG

Use reranking to improve your RAG pipeline:

```python
# 1. Get initial candidates from vector search (fast, approximate)
candidates = rag_query(query, top_k=20)

# 2. Rerank with cross-encoder (accurate, slower)
reranked = rerank(query, candidates[:20], top_k=5)

# 3. Use top results for LLM context
context = "\n".join([doc["document"] for doc in reranked])
```

---

## Anomaly Detection

Detect outliers and anomalies in numeric and mixed data using multiple algorithms.

See the dedicated [Anomaly Detection Guide](./anomaly-detection.md) for complete documentation.

### Quick Example

```bash
# 1. Train on normal data
curl -X POST http://localhost:11540/v1/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-monitor",
    "backend": "isolation_forest",
    "data": [[100, 1024], [105, 1100], [98, 980], [102, 1050]],
    "contamination": 0.1
  }'

# 2. Detect anomalies in new data
curl -X POST http://localhost:11540/v1/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-monitor",
    "data": [[100, 1024], [9999, 50000], [103, 1080]]
  }'
```

---

## File Management Endpoints

The Universal Runtime provides file storage for processing documents across multiple requests.

### Upload File

```bash
curl -X POST http://localhost:11540/v1/files \
  -F "file=@document.pdf" \
  -F "convert_pdf=true" \
  -F "pdf_dpi=150"
```

### List Files

```bash
curl http://localhost:11540/v1/files
```

### Get File Info

```bash
curl http://localhost:11540/v1/files/{file_id}
```

### Get File as Images

```bash
curl http://localhost:11540/v1/files/{file_id}/images
```

### Delete File

```bash
curl -X DELETE http://localhost:11540/v1/files/{file_id}
```

Files are stored temporarily (5-minute TTL by default).

---

## Next Steps

- [Anomaly Detection Guide](./anomaly-detection.md) - Complete anomaly detection documentation
- [Universal Runtime Overview](./index.md#universal-runtime) - General runtime configuration
- [API Reference](../api/index.md) - Full API documentation
