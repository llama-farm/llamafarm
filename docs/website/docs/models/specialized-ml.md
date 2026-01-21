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
| [Text Classification](#text-classification-pre-trained) | `POST /v1/classify` | Sentiment, spam detection (pre-trained models) |
| [Custom Classification](#custom-text-classification-setfit) | `POST /v1/classifier/*` | Train your own classifier with few examples |
| [Named Entity Recognition](#named-entity-recognition-ner) | `POST /v1/ner` | Extract people, places, organizations |
| [Reranking](#reranking-cross-encoder) | `POST /v1/rerank` | Improve RAG retrieval accuracy |
| [Anomaly Detection](#anomaly-detection) | `POST /v1/anomaly/*` | Detect outliers in numeric/mixed data |
| [Time-Series Forecasting](#time-series-forecasting) | `POST /v1/ml/timeseries/forecast` | Predict future values with confidence intervals |
| [PII Detection & Redaction](#pii-detection--redaction) | `POST /v1/ml/nlp/pii/*` | Find and redact sensitive information |
| [Language Detection](#language-detection) | `POST /v1/ml/nlp/language` | Identify language of text (20 languages) |
| [Table Question Answering](#table-question-answering) | `POST /v1/ml/analysis/table-qa` | Answer questions about tabular data |
| [Keyword Extraction](#keyword-extraction) | `POST /v1/ml/nlp/keywords` | Extract key phrases from documents |
| [Change Point Detection](#change-point-detection) | `POST /v1/ml/timeseries/changepoints` | Find structural changes in time-series |
| [Drift Detection](#drift-detection) | `POST /v1/ml/analysis/drift` | Monitor for concept drift in data streams |
| [Dataset Quality Audit](#dataset-quality-audit) | `POST /v1/ml/analysis/dataset-audit` | Find label errors and duplicates |
| [Anomaly Explanations](#anomaly-explanations) | `POST /v1/ml/anomaly/explain` | Explain why points are anomalous (SHAP) |

## Starting the Universal Runtime

```bash
# Start the runtime server
nx start universal-runtime

# Or with custom port
LF_RUNTIME_PORT=8080 nx start universal-runtime
```

The server runs on `http://localhost:8000` by default. However, all examples in this guide use the **LlamaFarm API** at `http://localhost:8000` which proxies requests to the runtime with additional features.

### Model Caching

The Universal Runtime caches loaded models in memory for faster inference on repeated requests:

- **Default TTL:** 5 minutes (300 seconds) of inactivity before unloading
- **Environment variable:** Set `MODEL_UNLOAD_TIMEOUT` to customize (in seconds)
- **Flash Attention 2:** Automatically enabled on CUDA devices for compatible models

```bash
# Keep models loaded for 30 minutes
MODEL_UNLOAD_TIMEOUT=1800 nx start universal-runtime
```

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

### Using the LlamaFarm API (Recommended)

The easiest way to use OCR is through the LlamaFarm API, which handles file uploads and PDF-to-image conversion automatically:

```bash
# Upload a PDF or image directly
curl -X POST http://localhost:8000/v1/vision/ocr \
  -F "file=@document.pdf" \
  -F "model=easyocr" \
  -F "languages=en"
```

Or with base64-encoded images:

```bash
curl -X POST http://localhost:8000/v1/vision/ocr \
  -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \
  -F "model=surya" \
  -F "languages=en"
```

**Supported file types:** PDF, PNG, JPG, JPEG, GIF, WebP, BMP, TIFF

### Direct API with Base64 Images

For programmatic access with base64-encoded images:

```bash
# OCR with base64 image
curl -X POST http://localhost:8000/v1/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "model": "surya",
    "images": ["'$(base64 -w0 document.png)'"],
    "languages": ["en"]
  }'
```

### PDF Processing Workflow

For multi-page documents:

```bash
# 1. Upload PDF (auto-converts to images)
curl -X POST http://localhost:8000/v1/files \
  -F "file=@document.pdf" \
  -F "convert_pdf=true" \
  -F "pdf_dpi=150"

# Response: {"id": "file_abc123", "page_count": 5, ...}

# 2. Run OCR on all pages
curl -X POST http://localhost:8000/v1/ocr \
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

### Using the LlamaFarm API (Recommended)

The easiest way to extract data from documents is through the LlamaFarm API:

```bash
# Extract from a receipt (file upload)
curl -X POST http://localhost:8000/v1/vision/documents/extract \
  -F "file=@receipt.pdf" \
  -F "model=naver-clova-ix/donut-base-finetuned-cord-v2" \
  -F "task=extraction"
```

**Supported file types:** PDF, PNG, JPG, JPEG, GIF, WebP, BMP, TIFF

### Extract from Receipt with File ID

Using a previously uploaded file:

```bash
curl -X POST http://localhost:8000/v1/documents/extract \
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

Ask questions about document content using the LlamaFarm API:

```bash
# Document VQA with file upload (LlamaFarm API)
curl -X POST http://localhost:8000/v1/vision/documents/extract \
  -F "file=@invoice.pdf" \
  -F "model=naver-clova-ix/donut-base-finetuned-docvqa" \
  -F "prompts=What is the total amount?,What is the invoice date?" \
  -F "task=vqa"
```

Or with a file ID:

```bash
curl -X POST http://localhost:8000/v1/documents/extract \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-clova-ix/donut-base-finetuned-docvqa",
    "file_id": "file_abc123",
    "prompts": ["What is the total amount?", "What is the invoice date?"],
    "task": "vqa"
  }'
```

---

## Text Classification (Pre-trained)

Use **pre-trained HuggingFace models** for common classification tasks like sentiment analysis. No training required - just pick a model and classify.

:::tip When to Use This vs Custom Classification
- **Use `/v1/classify`** when a pre-trained model exists for your task (sentiment, spam, toxicity)
- **Use `/v1/classifier/*`** when you need custom categories specific to your domain (intent routing, ticket categorization)
:::

### Popular Models

| Model | Use Case |
|-------|----------|
| `distilbert-base-uncased-finetuned-sst-2-english` | Sentiment analysis |
| `facebook/bart-large-mnli` | Zero-shot classification |
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | Social media sentiment |

:::tip Model Quantization
You can use quantized models for faster inference by appending a quantization suffix: `model:Q4_K_M`. For example: `distilbert-base-uncased-finetuned-sst-2-english:Q4_K_M`
:::

### Basic Classification

```bash
curl -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "distilbert-base-uncased-finetuned-sst-2-english",
    "texts": [
      "I love this product!",
      "This is terrible and broken.",
      "It works okay I guess."
    ],
    "max_length": 512
  }'
```

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | Yes | - | HuggingFace model ID |
| `texts` | array | Yes | - | Texts to classify |
| `max_length` | int | No | auto | Max sequence length (auto-detects: 8192 for ModernBERT, 512 for classic BERT) |

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

## Custom Text Classification (SetFit)

Train **your own text classifier** with as few as 8-16 examples per class using [SetFit](https://huggingface.co/docs/setfit) (Sentence Transformer Fine-tuning). Perfect for domain-specific classification tasks.

:::info How SetFit Works
SetFit uses contrastive learning to fine-tune a sentence-transformer model on your examples, then trains a small classification head. This approach achieves strong performance with minimal labeled data and no GPU required.
:::

### When to Use Custom Classification

| Scenario | Use `/v1/classify` | Use `/v1/classifier/*` |
|----------|-------------------|----------------------|
| Sentiment analysis | ✅ Pre-trained models available | ❌ Overkill |
| Intent routing (booking, support, billing) | ❌ No pre-trained model | ✅ Train on your intents |
| Ticket categorization | ❌ Domain-specific | ✅ Train on your categories |
| Content moderation | ✅ Toxicity models exist | ✅ If you need custom rules |
| Document classification | ❌ Domain-specific | ✅ Train on your doc types |

### Workflow Overview

```
1. Fit model     →  2. Predict  →  3. Save (optional)
   /classifier/fit    /classifier/predict    /classifier/save
```

:::tip Using the LlamaFarm API (Recommended)
The LlamaFarm API (`/v1/ml/classifier/*`) provides the same functionality as the Universal Runtime with added features:
- **Model Versioning**: Optional timestamped versions when `overwrite: false` (default is `true` for exact model names)
- **Latest Resolution**: Use `model-name-latest` to auto-resolve to the newest version (when using `overwrite: false`)
- **File Upload Support**: Direct file handling without base64 encoding

```bash
# Via LlamaFarm API (port 8000)
curl -X POST http://localhost:8000/v1/ml/classifier/fit ...

# Via Universal Runtime (port 11540)
curl -X POST http://localhost:11540/v1/classifier/fit ...
```
:::

:::warning Server vs Universal Runtime
- **`/v1/classify`** (pre-trained models) is **only available on Universal Runtime** (port 11540). It is NOT proxied through the LlamaFarm server.
- **`/v1/ml/classifier/*`** (custom SetFit classifiers) is available on the LlamaFarm server (port 8000) and proxies to Universal Runtime.
:::

### Step 1: Train Your Classifier

Provide labeled examples (minimum 2, recommended 8-16 per class):

```bash
curl -X POST http://localhost:8000/v1/classifier/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier",
    "base_model": "sentence-transformers/all-MiniLM-L6-v2",
    "training_data": [
      {"text": "I need to book a flight to NYC", "label": "booking"},
      {"text": "Reserve a hotel room for next week", "label": "booking"},
      {"text": "Can I get a table for two tonight?", "label": "booking"},
      {"text": "Cancel my reservation please", "label": "cancellation"},
      {"text": "I want to cancel my booking", "label": "cancellation"},
      {"text": "Please remove my appointment", "label": "cancellation"},
      {"text": "What is the weather like?", "label": "other"},
      {"text": "Tell me a joke", "label": "other"}
    ],
    "num_iterations": 20
  }'
```

**Response:**
```json
{
  "object": "fit_result",
  "model": "intent-classifier",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "samples_fitted": 8,
  "num_classes": 3,
  "labels": ["booking", "cancellation", "other"],
  "training_time_ms": 1234.56,
  "status": "fitted"
}
```

### Step 2: Classify New Texts

```bash
curl -X POST http://localhost:8000/v1/classifier/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "intent-classifier",
    "texts": [
      "I want to book a car for tomorrow",
      "Please cancel everything",
      "How are you doing?"
    ]
  }'
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {"text": "I want to book a car for tomorrow", "label": "booking", "score": 0.94, "all_scores": {"booking": 0.94, "cancellation": 0.03, "other": 0.03}},
    {"text": "Please cancel everything", "label": "cancellation", "score": 0.91, "all_scores": {"booking": 0.04, "cancellation": 0.91, "other": 0.05}},
    {"text": "How are you doing?", "label": "other", "score": 0.87, "all_scores": {"booking": 0.06, "cancellation": 0.07, "other": 0.87}}
  ],
  "model": "intent-classifier"
}
```

### Step 3: Save for Production

Save your trained model to persist across server restarts:

```bash
curl -X POST http://localhost:8000/v1/classifier/save \
  -H "Content-Type: application/json" \
  -d '{"model": "intent-classifier"}'
```

**Response:**
```json
{
  "object": "save_result",
  "model": "intent-classifier",
  "path": "~/.llamafarm/models/classifier/intent-classifier",
  "status": "saved"
}
```

:::note Storage Structure
SetFit classifiers are stored as **directories** (not files) under `~/.llamafarm/models/classifier/`. Each directory contains:
- Model weights and config
- `labels.txt` - Class labels for the classifier

**Note:** Models are auto-saved immediately after fitting, so explicit save is optional but recommended for adding descriptions.
:::

### Loading Saved Models

After a server restart, load your saved model:

```bash
curl -X POST http://localhost:8000/v1/classifier/load \
  -H "Content-Type: application/json" \
  -d '{"model": "intent-classifier"}'
```

### List & Delete Models

```bash
# List all saved classifiers
curl http://localhost:8000/v1/classifier/models

# Delete a model
curl -X DELETE http://localhost:8000/v1/classifier/models/intent-classifier
```

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/classifier/fit` | POST | Train a classifier on labeled examples |
| `/v1/classifier/predict` | POST | Classify texts using a trained model |
| `/v1/classifier/save` | POST | Save model to disk |
| `/v1/classifier/load` | POST | Load model from disk |
| `/v1/classifier/models` | GET | List saved models |
| `/v1/classifier/models/{name}` | DELETE | Delete a saved model |

### Training Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | required | Unique name for your classifier |
| `base_model` | string | `all-MiniLM-L6-v2` | Sentence transformer to fine-tune |
| `training_data` | array | required | List of `{text, label}` objects |
| `num_iterations` | int | 20 | Contrastive learning iterations |
| `batch_size` | int | 16 | Training batch size |

### Recommended Base Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `sentence-transformers/all-MiniLM-L6-v2` | 80MB | Fast | Good |
| `sentence-transformers/all-mpnet-base-v2` | 420MB | Medium | Better |
| `BAAI/bge-small-en-v1.5` | 130MB | Fast | Good |
| `BAAI/bge-base-en-v1.5` | 440MB | Medium | Better |

### Best Practices

1. **Provide diverse examples**: Include variations in phrasing, not just similar sentences
2. **Balance classes**: Aim for similar numbers of examples per class
3. **Start small**: 8-16 examples per class is often sufficient
4. **Test before saving**: Verify accuracy on held-out examples before saving
5. **Iterate**: Add more examples for classes with lower accuracy

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
curl -X POST http://localhost:8000/v1/ner \
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
curl -X POST http://localhost:8000/v1/rerank \
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
curl -X POST http://localhost:8000/v1/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-monitor",
    "backend": "isolation_forest",
    "data": [[100, 1024], [105, 1100], [98, 980], [102, 1050]],
    "contamination": 0.1
  }'

# 2. Detect anomalies in new data
curl -X POST http://localhost:8000/v1/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-monitor",
    "data": [[100, 1024], [9999, 50000], [103, 1080]]
  }'
```

---

## Time-Series Forecasting

Probabilistic time-series forecasting using Chronos-Bolt transformer models. Get point predictions with confidence intervals for sales, demand, resource planning, and more.

### Model

| Model | Description |
|-------|-------------|
| `amazon/chronos-t5-small` | Fast, accurate forecasting with uncertainty quantification |

### Basic Forecasting

```bash
curl -X POST http://localhost:8000/v1/ml/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "values": [100, 120, 115, 130, 125, 140, 135, 150, 145, 160],
    "horizon": 5
  }'
```

### Response Format

```json
{
  "forecasts": [
    {"step": 1, "point": 155.2, "lower": 148.5, "upper": 162.1},
    {"step": 2, "point": 160.8, "lower": 151.2, "upper": 170.4},
    {"step": 3, "point": 165.3, "lower": 153.1, "upper": 177.5},
    {"step": 4, "point": 170.1, "lower": 155.8, "upper": 184.4},
    {"step": 5, "point": 174.6, "lower": 158.2, "upper": 191.0}
  ],
  "horizon": 5,
  "input_length": 10,
  "quantiles": [0.1, 0.5, 0.9]
}
```

### Sales Forecasting Example

```bash
# Forecast next 7 days of sales
curl -X POST http://localhost:8000/v1/ml/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "values": [1200, 1350, 1280, 1420, 1380, 1500, 1450, 1320, 1400, 1350, 1480, 1420, 1550, 1500],
    "horizon": 7
  }'
```

### Custom Confidence Intervals

Request specific quantiles for your risk tolerance:

```bash
curl -X POST http://localhost:8000/v1/ml/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "values": [100, 105, 98, 112, 108, 120, 115],
    "horizon": 3,
    "quantiles": [0.05, 0.25, 0.5, 0.75, 0.95],
    "num_samples": 100
  }'
```

### Batch Forecasting (Multiple Series)

Forecast multiple time-series in one request:

```bash
curl -X POST http://localhost:8000/v1/ml/timeseries/forecast_batch \
  -H "Content-Type: application/json" \
  -d '{
    "series_list": [
      [100, 110, 105, 120, 115, 130, 125],
      [500, 520, 510, 540, 530, 560, 550],
      [50, 55, 52, 58, 56, 62, 60]
    ],
    "horizon": 3
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | array | required | Historical time-series values (minimum 3) |
| `horizon` | int | 7 | Number of future steps to forecast |
| `quantiles` | array | `[0.1, 0.5, 0.9]` | Quantile levels for confidence intervals |
| `num_samples` | int | 20 | Sample paths for uncertainty estimation |

### Use Cases

- **Sales forecasting**: Predict daily/weekly revenue with confidence bounds
- **Demand prediction**: Plan inventory based on expected demand ranges
- **Resource planning**: Forecast server load, staffing needs
- **Energy consumption**: Predict power usage for capacity planning
- **Financial projections**: Revenue forecasts with uncertainty quantification

### Best Practices

1. **More history = better forecasts**: Provide at least 30 observations when possible
2. **Capture seasonality**: Include full seasonal cycles (e.g., full year for yearly patterns)
3. **Use confidence intervals**: For critical decisions, plan based on upper bounds
4. **Monitor accuracy**: Compare predictions to actuals and retrain periodically

---

## PII Detection & Redaction

Detect and redact personally identifiable information (PII) using GLiNER, a zero-shot named entity recognition model. No training required - works out of the box for common PII types.

### Supported PII Types

| Type | Examples |
|------|----------|
| `person` | John Smith, Dr. Jane Doe |
| `email` | user@example.com |
| `phone number` | 555-123-4567, (800) 555-0199 |
| `social security number` | 123-45-6789 |
| `credit card number` | 4111-1111-1111-1111 |
| `address` | 123 Main St, New York, NY 10001 |
| `date of birth` | 03/15/1985, March 15, 1985 |
| `ip address` | 192.168.1.100 |

### Detect PII

Find PII entities in text without modifying it:

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/pii/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contact John Smith at john.smith@email.com or call 555-123-4567"
  }'
```

**Response:**
```json
{
  "entities": [
    {"text": "John Smith", "label": "person", "start": 8, "end": 18, "score": 0.95},
    {"text": "john.smith@email.com", "label": "email", "start": 22, "end": 42, "score": 1.0},
    {"text": "555-123-4567", "label": "phone", "start": 51, "end": 63, "score": 1.0}
  ]
}
```

### Redact PII

Replace PII with placeholder text:

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/redact \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Customer: Jane Doe\nSSN: 123-45-6789\nEmail: jane@company.com"
  }'
```

**Response:**
```json
{
  "redacted_text": "Customer: [REDACTED]\nSSN: [REDACTED]\nEmail: [REDACTED]",
  "entities": [
    {"text": "Jane Doe", "label": "person", "start": 10, "end": 18, "score": 0.94},
    {"text": "123-45-6789", "label": "ssn", "start": 25, "end": 36, "score": 1.0},
    {"text": "jane@company.com", "label": "email", "start": 44, "end": 60, "score": 1.0}
  ],
  "entity_count": 3
}
```

### Custom Replacement Patterns

Use different placeholders for each PII type:

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/redact \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contact Alice at alice@example.com or 555-0123",
    "replacement_map": {
      "person": "[NAME]",
      "email": "[EMAIL]",
      "phone": "[PHONE]"
    }
  }'
```

**Response:**
```json
{
  "redacted_text": "Contact [NAME] at [EMAIL] or [PHONE]",
  "entities": [...],
  "entity_count": 3
}
```

### Detect Specific Entity Types

Focus on particular PII categories:

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/pii/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Patient Robert Johnson, MRN 12345678, DOB 03/15/1985",
    "entity_types": ["person", "date of birth", "medical record number"],
    "threshold": 0.3
  }'
```

### Batch Processing

Process multiple documents efficiently:

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/pii/detect_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Email from Bob Smith <bob@email.com>",
      "Call Sarah at 555-0199",
      "Invoice for John Doe, 123 Main St"
    ]
  }'
```

### Parameters

**Detection:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to analyze |
| `entity_types` | array | all types | Specific PII types to detect |
| `threshold` | float | 0.5 | Minimum confidence (0-1) |
| `use_regex` | bool | true | Also use regex patterns for common PII |

**Redaction:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Text to redact |
| `replacement` | string | `[REDACTED]` | Default replacement text |
| `replacement_map` | object | null | Per-type replacement strings |
| `entity_types` | array | all types | Types to redact |
| `threshold` | float | 0.5 | Minimum confidence |

### Use Cases

- **Log sanitization**: Remove PII before storing logs
- **Data export**: Anonymize customer data for analytics
- **Compliance**: GDPR, HIPAA, CCPA data handling
- **Chat moderation**: Detect shared personal information
- **Document processing**: Redact sensitive info before sharing

### Example: Data Pipeline Sanitization

```python
import httpx

def sanitize_for_logging(log_entry: str) -> str:
    """Remove PII from log entries before storage."""
    response = httpx.post(
        "http://localhost:8000/v1/ml/nlp/redact",
        json={
            "text": log_entry,
            "replacement": "***"
        }
    )
    return response.json()["redacted_text"]

# Sanitize logs
raw_log = "2024-01-15 10:30:00 User john@email.com logged in from 192.168.1.100"
safe_log = sanitize_for_logging(raw_log)
# Output: "2024-01-15 10:30:00 User *** logged in from ***"
```

### Example: Compliance Audit

```python
import httpx
from collections import Counter

def audit_document(text: str) -> dict:
    """Generate PII audit report for compliance."""
    response = httpx.post(
        "http://localhost:8000/v1/ml/nlp/pii/detect",
        json={"text": text, "threshold": 0.5}
    )

    entities = response.json()["entities"]
    type_counts = Counter(e["label"] for e in entities)

    high_risk = {"social security number", "credit card number"}
    high_risk_count = sum(1 for e in entities if e["label"] in high_risk)

    return {
        "total_pii": len(entities),
        "by_type": dict(type_counts),
        "risk_level": "HIGH" if high_risk_count > 0 else "MEDIUM" if len(entities) > 5 else "LOW"
    }
```

---

## Language Detection

Identify the language of text using XLM-RoBERTa, supporting 20 languages with high accuracy.

### Supported Languages

Arabic, Bulgarian, Chinese, Dutch, English, French, German, Greek, Hindi, Italian, Japanese, Polish, Portuguese, Russian, Spanish, Swahili, Thai, Turkish, Urdu, Vietnamese

### Detect Language

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/language \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, comment allez-vous?"
  }'
```

**Response:**
```json
{
  "language": "fr",
  "language_name": "French",
  "confidence": 0.98,
  "all_scores": {
    "fr": 0.98,
    "en": 0.01,
    "es": 0.005
  }
}
```

### Batch Language Detection

Detect languages for multiple texts:

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/language_batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Hello, how are you?",
      "Hola, ¿cómo estás?",
      "こんにちは、お元気ですか？",
      "Guten Tag, wie geht es Ihnen?"
    ]
  }'
```

### Use Cases

- **Content routing**: Route support tickets to language-appropriate teams
- **Translation preprocessing**: Detect source language before translation
- **Multilingual content moderation**: Apply language-specific rules
- **Analytics**: Analyze language distribution in user-generated content

### Example: Multilingual Content Router

```python
import httpx

def route_by_language(text: str) -> dict:
    response = httpx.post(
        "http://localhost:8000/v1/ml/nlp/language",
        json={"text": text}
    )
    result = response.json()

    language_teams = {
        "en": "english-support",
        "es": "spanish-support",
        "fr": "french-support",
        "de": "german-support",
    }

    return {
        "language": result["language"],
        "confidence": result["confidence"],
        "team": language_teams.get(result["language"], "general-support")
    }
```

---

## Table Question Answering

Answer natural language questions about tabular data using TAPAS (Table Parser). Supports cell selection, aggregation (SUM, AVG, COUNT), and complex queries.

### Model

| Model | Description |
|-------|-------------|
| `google/tapas-base-finetuned-wtq` | Trained on WikiTableQuestions, general-purpose |

### Basic Table QA

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/table-qa \
  -H "Content-Type: application/json" \
  -d '{
    "table": {
      "columns": ["Name", "Department", "Salary"],
      "rows": [
        ["Alice", "Engineering", "120000"],
        ["Bob", "Sales", "85000"],
        ["Carol", "Engineering", "110000"],
        ["David", "Marketing", "75000"]
      ]
    },
    "question": "Who has the highest salary?"
  }'
```

**Response:**
```json
{
  "answer": "Alice",
  "cells": [{"row": 0, "column": 0}],
  "cell_values": ["Alice"],
  "aggregation": "NONE",
  "question": "Who has the highest salary?"
}
```

### Aggregation Queries

TAPAS automatically detects when aggregation is needed:

```bash
# SUM query
curl -X POST http://localhost:8000/v1/ml/analysis/table-qa \
  -H "Content-Type: application/json" \
  -d '{
    "table": {
      "columns": ["Product", "Q1 Sales", "Q2 Sales"],
      "rows": [
        ["Widget A", "15000", "18000"],
        ["Widget B", "22000", "19000"],
        ["Widget C", "8000", "12000"]
      ]
    },
    "question": "What is the total Q1 sales?"
  }'

# Response: {"answer": "45000", "aggregation": "SUM", ...}
```

```bash
# COUNT query
curl -X POST http://localhost:8000/v1/ml/analysis/table-qa \
  -H "Content-Type: application/json" \
  -d '{
    "table": {...},
    "question": "How many employees are in Engineering?"
  }'

# Response: {"answer": "2", "aggregation": "COUNT", ...}
```

### Multiple Questions (Batch)

Ask multiple questions about the same table:

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/table-qa_batch \
  -H "Content-Type: application/json" \
  -d '{
    "table": {
      "columns": ["Country", "Population", "GDP"],
      "rows": [...]
    },
    "questions": [
      "Which country has the largest population?",
      "What is the average GDP?",
      "How many countries are listed?"
    ]
  }'
```

### Use Cases

- **Data exploration**: Query spreadsheets with natural language
- **Report generation**: Extract insights from tables automatically
- **Business intelligence**: Let non-technical users query data
- **Chatbot integration**: Answer questions about structured data

### Example: CSV Query Interface

```python
import httpx
import csv

def query_csv(csv_path: str, question: str) -> str:
    """Query a CSV file with natural language."""
    with open(csv_path) as f:
        reader = csv.reader(f)
        rows = list(reader)

    table = {
        "columns": rows[0],
        "rows": rows[1:]
    }

    response = httpx.post(
        "http://localhost:8000/v1/ml/analysis/table-qa",
        json={"table": table, "question": question}
    )
    return response.json()["answer"]

# Usage
answer = query_csv("sales_data.csv", "What was the best selling product in Q4?")
```

---

## Keyword Extraction

Extract the most important keywords and keyphrases from documents using embedding-based similarity and MMR (Maximal Marginal Relevance) for diversity.

### Basic Extraction

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/keywords \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn from data. Deep learning, a branch of machine learning, uses neural networks with multiple layers.",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "keywords": [
    {"keyword": "machine learning", "score": 0.87},
    {"keyword": "artificial intelligence", "score": 0.82},
    {"keyword": "deep learning", "score": 0.79},
    {"keyword": "neural networks", "score": 0.75},
    {"keyword": "data", "score": 0.68}
  ]
}
```

### Control Diversity

Use MMR diversity to avoid redundant keywords:

```bash
curl -X POST http://localhost:8000/v1/ml/nlp/keywords \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your document text here...",
    "top_k": 10,
    "diversity": 0.7,
    "ngram_range": [1, 3]
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | required | Document to extract keywords from |
| `top_k` | int | 10 | Number of keywords to return |
| `diversity` | float | 0.5 | MMR diversity (0=no diversity, 1=max) |
| `ngram_range` | array | `[1, 3]` | Min/max n-gram length |

### Use Cases

- **SEO optimization**: Extract keywords for meta tags
- **Document indexing**: Generate tags for search
- **Content summarization**: Identify main topics
- **Trend analysis**: Track keyword frequency over time

### Example: Document Tagging Pipeline

```python
import httpx

def auto_tag_document(text: str, max_tags: int = 5) -> list[str]:
    """Automatically generate tags for a document."""
    response = httpx.post(
        "http://localhost:8000/v1/ml/nlp/keywords",
        json={
            "text": text,
            "top_k": max_tags,
            "diversity": 0.6  # Some diversity to avoid similar tags
        }
    )

    # Return keywords with score > 0.5
    return [
        kw["keyword"]
        for kw in response.json()["keywords"]
        if kw["score"] > 0.5
    ]

# Usage
tags = auto_tag_document(article_text)
# ["machine learning", "data science", "python", ...]
```

---

## Change Point Detection

Detect structural changes (regime shifts) in time-series data using the Ruptures library. Find where statistical properties like mean, variance, or trend change significantly.

### Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `pelt` | Optimal with linear complexity | General use, unknown # of changes |
| `binseg` | Binary segmentation (fast) | Large datasets, approximate |
| `window` | Sliding window | Trend changes |
| `bottomup` | Bottom-up segmentation | Many small segments |

### Basic Detection

```bash
curl -X POST http://localhost:8000/v1/ml/timeseries/changepoints \
  -H "Content-Type: application/json" \
  -d '{
    "values": [1, 1, 1, 1, 5, 5, 5, 5, 2, 2, 2, 2],
    "algorithm": "pelt"
  }'
```

**Response:**
```json
{
  "change_points": [4, 8],
  "n_segments": 3,
  "segment_boundaries": [
    {"start": 0, "end": 4},
    {"start": 4, "end": 8},
    {"start": 8, "end": 12}
  ],
  "signal_length": 12,
  "algorithm": "pelt"
}
```

### Known Number of Changes

If you know how many changes to expect:

```bash
curl -X POST http://localhost:8000/v1/ml/timeseries/changepoints \
  -H "Content-Type: application/json" \
  -d '{
    "values": [10, 12, 11, 50, 52, 48, 20, 22, 19],
    "n_changepoints": 2
  }'
```

### Custom Penalty

Control sensitivity with penalty parameter (higher = fewer change points):

```bash
curl -X POST http://localhost:8000/v1/ml/timeseries/changepoints \
  -H "Content-Type: application/json" \
  -d '{
    "values": [...],
    "penalty": 10.0
  }'
```

### Use Cases

- **Anomaly detection**: Find when normal patterns change
- **A/B testing**: Detect when metrics shift after experiments
- **Equipment monitoring**: Identify regime changes in sensor data
- **Financial analysis**: Find trend reversals

### Example: Detect Metric Shifts

```python
import httpx

def find_metric_shifts(daily_values: list[float]) -> list[dict]:
    """Find significant changes in a daily metric."""
    response = httpx.post(
        "http://localhost:8000/v1/ml/timeseries/changepoints",
        json={
            "values": daily_values,
            "algorithm": "pelt",
            "model": "rbf"  # Good for mean/variance changes
        }
    )

    result = response.json()
    shifts = []

    for i, cp in enumerate(result["change_points"]):
        seg_before = result["segment_boundaries"][i]
        seg_after = result["segment_boundaries"][i + 1]

        before_avg = sum(daily_values[seg_before["start"]:seg_before["end"]]) / (seg_before["end"] - seg_before["start"])
        after_avg = sum(daily_values[seg_after["start"]:seg_after["end"]]) / (seg_after["end"] - seg_after["start"])

        shifts.append({
            "day": cp,
            "change": after_avg - before_avg,
            "before_avg": before_avg,
            "after_avg": after_avg
        })

    return shifts
```

---

## Drift Detection

Monitor streaming data for concept drift - when statistical properties change over time. Critical for knowing when ML models need retraining.

### Algorithms

| Algorithm | Description | Best For |
|-----------|-------------|----------|
| `adwin` | Adaptive windowing | Numeric data, general use |
| `page_hinkley` | Page-Hinkley test | Detecting mean changes |
| `kswin` | Kolmogorov-Smirnov windowing | Distribution changes |
| `ddm` | Drift Detection Method | Error rate monitoring |

### Detect Drift in Stream

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/drift \
  -H "Content-Type: application/json" \
  -d '{
    "values": [1.0, 1.1, 0.9, 1.0, 1.1, 5.0, 5.2, 4.9, 5.1, 5.0],
    "algorithm": "adwin"
  }'
```

**Response:**
```json
{
  "total_processed": 10,
  "drift_detected": true,
  "drift_points": [6],
  "final_index": 10,
  "algorithm": "adwin"
}
```

### Monitor Error Rates (DDM)

For monitoring model prediction errors:

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/drift \
  -H "Content-Type: application/json" \
  -d '{
    "values": [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1],
    "algorithm": "ddm"
  }'
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `values` | array | required | Stream of numeric values |
| `algorithm` | string | `adwin` | Detection algorithm |
| `delta` | float | 0.002 | Sensitivity for ADWIN |
| `threshold` | float | 50.0 | Threshold for Page-Hinkley |
| `window_size` | int | 100 | Window size for KSWIN |

### Use Cases

- **Model monitoring**: Detect when predictions degrade
- **Data quality**: Find when input distributions change
- **A/B testing**: Detect when experiments cause shifts
- **Alerting**: Trigger retraining when drift occurs

### Example: Model Performance Monitor

```python
import httpx

class ModelMonitor:
    def __init__(self):
        self.errors = []

    def record_prediction(self, predicted, actual):
        error = 1 if predicted != actual else 0
        self.errors.append(error)

        # Check for drift periodically
        if len(self.errors) % 100 == 0:
            return self.check_drift()
        return None

    def check_drift(self) -> dict:
        response = httpx.post(
            "http://localhost:8000/v1/ml/analysis/drift",
            json={
                "values": self.errors[-500:],  # Last 500 predictions
                "algorithm": "ddm"
            }
        )
        result = response.json()

        if result["drift_detected"]:
            return {
                "alert": "Model drift detected!",
                "drift_points": result["drift_points"],
                "action": "Consider retraining model"
            }
        return {"status": "Model performing normally"}
```

---

## Dataset Quality Audit

Find label errors, duplicates, and quality issues in classification datasets using Cleanlab. Essential for improving training data quality.

### Audit Dataset

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/dataset-audit \
  -H "Content-Type: application/json" \
  -d '{
    "labels": [0, 1, 0, 1, 0, 1, 0, 1],
    "pred_probs": [
      [0.9, 0.1], [0.2, 0.8], [0.85, 0.15], [0.1, 0.9],
      [0.3, 0.7], [0.15, 0.85], [0.95, 0.05], [0.05, 0.95]
    ],
    "label_names": ["negative", "positive"]
  }'
```

**Response:**
```json
{
  "label_issues": [
    {
      "index": 4,
      "given_label": 0,
      "given_label_name": "negative",
      "suggested_label": 1,
      "suggested_label_name": "positive",
      "given_confidence": 0.3,
      "suggested_confidence": 0.7
    }
  ],
  "duplicates": [],
  "summary": {
    "total_samples": 8,
    "label_issue_count": 1,
    "label_issue_rate": 0.125,
    "duplicate_pair_count": 0,
    "unique_labels": 2
  }
}
```

### Find Duplicates

Include feature vectors to detect near-duplicate samples:

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/dataset-audit \
  -H "Content-Type: application/json" \
  -d '{
    "labels": [...],
    "pred_probs": [...],
    "features": [[0.1, 0.2, ...], [0.1, 0.2, ...], ...],
    "check_duplicates": true,
    "duplicate_threshold": 0.95
  }'
```

### Get Label Quality Scores

Get per-sample quality scores:

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/dataset-audit \
  -H "Content-Type: application/json" \
  -d '{
    "labels": [0, 1, 0, 1],
    "pred_probs": [[0.9, 0.1], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]]
  }'

# Response: {"scores": [0.9, 0.8, 0.5, 0.9]}
```

### Suggest Corrections

Get high-confidence correction suggestions:

```bash
curl -X POST http://localhost:8000/v1/ml/analysis/dataset-audit \
  -H "Content-Type: application/json" \
  -d '{
    "labels": [...],
    "pred_probs": [...],
    "min_confidence_diff": 0.3
  }'
```

### Use Cases

- **Training data cleanup**: Find and fix mislabeled examples
- **Data deduplication**: Remove near-duplicate samples
- **Quality metrics**: Track label quality over time
- **Active learning**: Prioritize samples for re-labeling

### Example: Training Data Cleanup Pipeline

```python
import httpx

def cleanup_training_data(texts: list, labels: list, classifier_probs):
    """Find and suggest fixes for label errors."""

    # Audit the dataset
    response = httpx.post(
        "http://localhost:8000/v1/ml/analysis/dataset-audit",
        json={
            "labels": labels,
            "pred_probs": classifier_probs.tolist(),
            "label_names": ["negative", "neutral", "positive"]
        }
    )

    result = response.json()

    print(f"Found {result['summary']['label_issue_count']} potential label errors")
    print(f"Error rate: {result['summary']['label_issue_rate']:.1%}")

    # Show examples to review
    for issue in result["label_issues"][:10]:
        print(f"\nSample {issue['index']}:")
        print(f"  Text: {texts[issue['index']][:100]}...")
        print(f"  Current: {issue['given_label_name']} ({issue['given_confidence']:.1%})")
        print(f"  Suggested: {issue['suggested_label_name']} ({issue['suggested_confidence']:.1%})")

    return result["label_issues"]
```

---

## Anomaly Explanations

Explain why data points are flagged as anomalies using SHAP (SHapley Additive exPlanations). Provides interpretable feature-level explanations.

### Explain Anomalies

```bash
curl -X POST http://localhost:8000/v1/ml/anomaly/explain \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my-anomaly-detector",
    "data": [[95.0, 1024, 500]],
    "feature_names": ["cpu_percent", "memory_mb", "network_io"]
  }'
```

**Response:**
```json
{
  "explanations": [
    {
      "features": [
        {"feature": "cpu_percent", "importance": 0.82, "value": 95.0, "direction": "high", "shap_value": 0.82},
        {"feature": "memory_mb", "importance": 0.15, "value": 1024, "direction": "low", "shap_value": -0.15},
        {"feature": "network_io", "importance": 0.03, "value": 500, "direction": "low", "shap_value": -0.03}
      ],
      "top_contributors": [
        {"feature": "cpu_percent", "importance": 0.82, "value": 95.0, "direction": "high"}
      ],
      "total_shap": 1.0
    }
  ]
}
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | string | required | Name of trained anomaly detector |
| `data` | array | required | Data points to explain |
| `feature_names` | array | null | Names for each feature |
| `n_samples` | int | 100 | SHAP approximation samples |

### Understanding Results

- **importance**: Absolute contribution to anomaly score (higher = more important)
- **direction**: Whether the feature value is "high" or "low" relative to normal
- **shap_value**: Signed contribution (positive = increases anomaly score)
- **top_contributors**: Most important features (top 3)

### Use Cases

- **Root cause analysis**: Understand why alerts fire
- **Debugging**: Investigate false positives/negatives
- **Reporting**: Generate human-readable explanations
- **Model validation**: Verify model uses sensible features

### Example: Alert Explanation System

```python
import httpx

def explain_alert(data_point: list, feature_names: list) -> str:
    """Generate human-readable explanation for an anomaly alert."""

    response = httpx.post(
        "http://localhost:8000/v1/ml/anomaly/explain",
        json={
            "model_name": "server-monitor",
            "data": [data_point],
            "feature_names": feature_names
        }
    )

    explanation = response.json()["explanations"][0]
    top = explanation["top_contributors"]

    parts = []
    for feat in top:
        if feat["direction"] == "high":
            parts.append(f"{feat['feature']} is unusually high ({feat['value']})")
        else:
            parts.append(f"{feat['feature']} is unusually low ({feat['value']})")

    return "Alert triggered because: " + "; ".join(parts)

# Usage
explanation = explain_alert(
    [95.0, 512, 10000],
    ["cpu_percent", "memory_mb", "requests_per_sec"]
)
# "Alert triggered because: cpu_percent is unusually high (95.0); requests_per_sec is unusually high (10000)"
```

### Example: Batch Explanation for Investigation

```python
import httpx

def investigate_anomalies(anomalous_data: list, feature_names: list):
    """Investigate multiple anomalies to find common patterns."""

    response = httpx.post(
        "http://localhost:8000/v1/ml/anomaly/explain",
        json={
            "model_name": "fraud-detector",
            "data": anomalous_data,
            "feature_names": feature_names
        }
    )

    explanations = response.json()["explanations"]

    # Count which features appear most often as top contributors
    from collections import Counter
    feature_counts = Counter()

    for exp in explanations:
        for feat in exp["top_contributors"]:
            feature_counts[feat["feature"]] += 1

    print("Most common anomaly drivers:")
    for feature, count in feature_counts.most_common(5):
        pct = count / len(explanations) * 100
        print(f"  {feature}: {pct:.0f}% of anomalies")
```

---

## File Management Endpoints

The Universal Runtime provides file storage for processing documents across multiple requests.

### Upload File

```bash
curl -X POST http://localhost:8000/v1/files \
  -F "file=@document.pdf" \
  -F "convert_pdf=true" \
  -F "pdf_dpi=150"
```

### List Files

```bash
curl http://localhost:8000/v1/files
```

### Get File Info

```bash
curl http://localhost:8000/v1/files/{file_id}
```

### Get File as Images

```bash
curl http://localhost:8000/v1/files/{file_id}/images
```

### Delete File

```bash
curl -X DELETE http://localhost:8000/v1/files/{file_id}
```

Files are stored temporarily (5-minute TTL by default).

---

## Next Steps

- [Anomaly Detection Guide](./anomaly-detection.md) - Complete anomaly detection documentation
- [Vision Models](./vision-ml.md) - Image classification, object detection, OCR
- [Universal Runtime Overview](./index.md#universal-runtime) - General runtime configuration
- [API Reference](../api/index.md) - Full API documentation
