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
| [Anomaly Detection](#anomaly-detection) | `POST /v1/ml/anomaly/*` | Detect outliers in numeric/mixed data |
| [Time-Series Forecasting](#time-series-forecasting) | `POST /v1/timeseries/*` | Predict future values with Chronos or traditional methods |
| [Time-Series Anomaly Detection](#time-series-anomaly-detection-adtk) | `POST /v1/adtk/*` | Detect level shifts, spikes, stuck values |
| [Data Drift Detection](#data-drift-detection) | `POST /v1/drift/*` | Monitor ML model data distribution changes |
| [CatBoost](#catboost-gradient-boosting) | `POST /v1/catboost/*` | Native categorical support, incremental learning |

## Starting the Universal Runtime

```bash
# Start the runtime server
nx start universal-runtime

# Or with custom port
LF_RUNTIME_PORT=8080 nx start universal-runtime
```

The server runs on `http://localhost:11540` by default.

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
curl -X POST http://localhost:14345/v1/vision/ocr \
  -F "file=@document.pdf" \
  -F "model=easyocr" \
  -F "languages=en"
```

Or with base64-encoded images:

```bash
curl -X POST http://localhost:14345/v1/vision/ocr \
  -F 'images=["data:image/png;base64,iVBORw0KGgo..."]' \
  -F "model=surya" \
  -F "languages=en"
```

**Supported file types:** PDF, PNG, JPG, JPEG, GIF, WebP, BMP, TIFF

### Using the Universal Runtime Directly

For more control, you can use the Universal Runtime directly with base64 images:

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

### PDF Processing Workflow (Universal Runtime)

For multi-page documents using the Universal Runtime directly:

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

### Using the LlamaFarm API (Recommended)

The easiest way to extract data from documents is through the LlamaFarm API:

```bash
# Extract from a receipt (file upload)
curl -X POST http://localhost:14345/v1/vision/documents/extract \
  -F "file=@receipt.pdf" \
  -F "model=naver-clova-ix/donut-base-finetuned-cord-v2" \
  -F "task=extraction"
```

**Supported file types:** PDF, PNG, JPG, JPEG, GIF, WebP, BMP, TIFF

### Extract from Receipt (Universal Runtime)

Using the Universal Runtime directly with a file ID:

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

Ask questions about document content using the LlamaFarm API:

```bash
# Document VQA with file upload (LlamaFarm API)
curl -X POST http://localhost:14345/v1/vision/documents/extract \
  -F "file=@invoice.pdf" \
  -F "model=naver-clova-ix/donut-base-finetuned-docvqa" \
  -F "prompts=What is the total amount?,What is the invoice date?" \
  -F "task=vqa"
```

Or using the Universal Runtime directly:

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
curl -X POST http://localhost:11540/v1/classify \
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
- **Model Versioning**: Automatic timestamped versions when `overwrite: false`
- **Latest Resolution**: Use `model-name-latest` to auto-resolve to the newest version

```bash
# Via LlamaFarm API (port 14345)
curl -X POST http://localhost:14345/v1/ml/classifier/fit ...

# Via Universal Runtime (port 11540)
curl -X POST http://localhost:11540/v1/classifier/fit ...
```
:::

:::warning Server vs Universal Runtime
- **`/v1/classify`** (pre-trained models) is **only available on Universal Runtime** (port 11540). It is NOT proxied through the LlamaFarm server.
- **`/v1/ml/classifier/*`** (custom SetFit classifiers) is available on the LlamaFarm server (port 14345) and proxies to Universal Runtime.
:::

### Step 1: Train Your Classifier

Provide labeled examples (minimum 2, recommended 8-16 per class):

```bash
curl -X POST http://localhost:11540/v1/classifier/fit \
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
curl -X POST http://localhost:11540/v1/classifier/predict \
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
curl -X POST http://localhost:11540/v1/classifier/save \
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
curl -X POST http://localhost:11540/v1/classifier/load \
  -H "Content-Type: application/json" \
  -d '{"model": "intent-classifier"}'
```

### List & Delete Models

```bash
# List all saved classifiers
curl http://localhost:11540/v1/classifier/models

# Delete a model
curl -X DELETE http://localhost:11540/v1/classifier/models/intent-classifier
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

### Quick Example (LlamaFarm API - Recommended)

```bash
# 1. Train on normal data
curl -X POST http://localhost:14345/v1/ml/anomaly/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-monitor",
    "backend": "isolation_forest",
    "data": [[100, 1024], [105, 1100], [98, 980], [102, 1050]],
    "contamination": 0.1
  }'

# 2. Detect anomalies in new data
curl -X POST http://localhost:14345/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "api-monitor",
    "data": [[100, 1024], [9999, 50000], [103, 1080]]
  }'
```

---

## Time-Series Forecasting

Predict future values using Amazon's Chronos foundation model (zero-shot) or traditional statistical methods (ARIMA, Exponential Smoothing, Theta).

:::info Zero-Shot vs Traditional
- **Chronos** requires NO training - just provide historical data and get forecasts immediately
- **Traditional methods** (ARIMA, ETS) need fitting but may be more accurate for domain-specific patterns
:::

### Supported Backends

| Backend | Training Required | Best For |
|---------|------------------|----------|
| `chronos` | No (zero-shot) | General time-series, quick predictions |
| `arima` | Yes (fit) | Linear trends, seasonal patterns |
| `exponential_smoothing` | Yes (fit) | Trend + seasonality decomposition |
| `theta` | Yes (fit) | Simple, robust forecasting |

### List Available Backends

```bash
curl http://localhost:11540/v1/timeseries/backends
```

### Chronos Zero-Shot Forecasting

Chronos is a foundation model for time-series - no training required:

```bash
curl -X POST http://localhost:11540/v1/timeseries/predict \
  -H "Content-Type: application/json" \
  -d '{
    "backend": "chronos",
    "values": [50, 52, 48, 55, 60, 58, 62, 65, 63, 70],
    "horizon": 5,
    "return_ci": true
  }'
```

**Response:**
```json
{
  "backend": "chronos",
  "predictions": [72.3, 74.1, 75.8, 77.2, 78.5],
  "timestamps": null,
  "ci_lower": [68.1, 69.5, 70.2, 71.0, 71.8],
  "ci_upper": [76.5, 78.7, 81.4, 83.4, 85.2],
  "horizon": 5,
  "predict_time_ms": 156.3
}
```

### Traditional Methods (Fit + Predict)

For domain-specific data, traditional methods may work better:

```bash
# 1. Fit model on historical data
curl -X POST http://localhost:11540/v1/timeseries/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "cpu-forecast",
    "backend": "exponential_smoothing",
    "values": [50, 52, 48, 55, 60, 58, 62, 65, 63, 70, 68, 75],
    "seasonal_periods": 7
  }'

# 2. Forecast future values
curl -X POST http://localhost:11540/v1/timeseries/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "cpu-forecast",
    "horizon": 24,
    "return_ci": true
  }'
```

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/timeseries/backends` | GET | List available forecasting backends |
| `/v1/timeseries/fit` | POST | Fit traditional model on data |
| `/v1/timeseries/predict` | POST | Generate forecast |
| `/v1/timeseries/models` | GET | List fitted models |
| `/v1/timeseries/models/{id}` | DELETE | Delete a model |

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `backend` | string | Yes | `chronos`, `arima`, `exponential_smoothing`, `theta` |
| `values` | array | Yes | Historical time-series values |
| `horizon` | int | Yes | Number of periods to forecast |
| `return_ci` | bool | No | Return confidence intervals (default: false) |
| `seasonal_periods` | int | No | Seasonality period for traditional methods |
| `model_id` | string | No | Model name for traditional methods |

---

## Time-Series Anomaly Detection (ADTK)

Detect temporal anomalies using specialized detectors from the ADTK (Anomaly Detection Toolkit) library. Unlike general anomaly detection (PyOD), ADTK understands time context.

:::tip When to Use ADTK vs PyOD
| Use Case | Tool |
|----------|------|
| Point anomalies (no time context) | PyOD (`/v1/ml/anomaly/*`) |
| Level shifts (baseline changes) | ADTK (`/v1/adtk/detect`) |
| Spikes (short-term outliers) | ADTK (`/v1/adtk/detect`) |
| Threshold violations | ADTK (`/v1/adtk/detect`) |
| Seasonal anomalies | ADTK (`/v1/adtk/detect`) |
:::

### Available Detectors

```bash
curl http://localhost:11540/v1/adtk/detectors
```

| Detector | Description | Use Case |
|----------|-------------|----------|
| `level_shift` | Detects sudden baseline changes | Infrastructure migrations, config changes |
| `spike` | Detects short-term outliers | Traffic spikes, system errors |
| `threshold` | Alerts when values exceed limits | Resource monitoring, SLA violations |
| `volatility_shift` | Detects variance changes | Market data, system stability |
| `seasonal` | Detects seasonal pattern deviations | Periodic workloads, business cycles |

### Level Shift Detection

Detect sudden changes in baseline values:

```bash
curl -X POST http://localhost:11540/v1/adtk/detect \
  -H "Content-Type: application/json" \
  -d '{
    "detector": "level_shift",
    "values": [50, 52, 48, 51, 49, 80, 82, 78, 81, 79],
    "timestamps": ["2024-01-01T00:00:00", "2024-01-01T00:05:00", ...],
    "params": {"c": 6.0, "window": 3}
  }'
```

**Response:**
```json
{
  "detector": "level_shift",
  "anomalies": [
    {"timestamp": "2024-01-01T00:25:00", "value": 80, "score": 0.95}
  ],
  "anomaly_count": 1,
  "detection_time_ms": 12.5
}
```

### Spike Detection

Detect short-term outliers:

```bash
curl -X POST http://localhost:11540/v1/adtk/detect \
  -H "Content-Type: application/json" \
  -d '{
    "detector": "spike",
    "values": [50, 52, 150, 51, 49, 48, 200, 52],
    "params": {"c": 3.0}
  }'
```

### Threshold Detection

Alert when values exceed limits:

```bash
curl -X POST http://localhost:11540/v1/adtk/detect \
  -H "Content-Type: application/json" \
  -d '{
    "detector": "threshold",
    "values": [50, 60, 70, 90, 95, 75, 60],
    "params": {"high": 85}
  }'
```

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/adtk/detectors` | GET | List available detectors |
| `/v1/adtk/detect` | POST | Detect anomalies in time-series |

### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `detector` | string | Yes | Detector type (see table above) |
| `values` | array | Yes | Time-series values |
| `timestamps` | array | No | ISO timestamps (auto-generated if omitted) |
| `params` | object | No | Detector-specific parameters |

### Detector Parameters

**level_shift:**
- `c` (float): Sensitivity factor, higher = less sensitive (default: 6.0)
- `window` (int): Window size for comparison (default: 5)

**spike:**
- `c` (float): Standard deviation threshold (default: 3.0)

**threshold:**
- `high` (float): Upper threshold
- `low` (float): Lower threshold

---

## Data Drift Detection

Monitor data distribution changes using Alibi Detect. Essential for ML model monitoring to detect when production data differs from training data.

:::info Why Drift Detection Matters
When production data distributions shift from training data, model performance degrades. Drift detection helps you:
- Identify when to retrain models
- Catch data quality issues early
- Monitor feature distributions over time
:::

### Available Detectors

```bash
curl http://localhost:11540/v1/drift/detectors
```

| Detector | Type | Best For |
|----------|------|----------|
| `ks` | Univariate | Individual feature distribution shifts |
| `mmd` | Multivariate | Correlation and joint distribution changes |
| `chi2` | Categorical | Categorical feature drift |

### Workflow

```
1. Fit detector on reference data (training distribution)
   ↓
2. Monitor production data for drift
   ↓
3. Alert and retrain when drift detected
```

### Step 1: Fit Drift Detector

Train the detector on your reference (training) data distribution:

```bash
curl -X POST http://localhost:11540/v1/drift/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model": "feature-monitor",
    "detector": "ks",
    "reference_data": [
      [50.1, 100.2, 45.3],
      [52.4, 98.1, 48.7],
      [48.9, 102.5, 44.1],
      ...
    ],
    "params": {"p_val": 0.05}
  }'
```

**Response:**
```json
{
  "model": "feature-monitor",
  "detector": "ks",
  "saved_path": "~/.llamafarm/models/drift/feature-monitor_ks.joblib",
  "training_time_ms": 45.2,
  "reference_size": 500,
  "n_features": 3
}
```

### Step 2: Detect Drift

Check new data for distribution drift:

```bash
curl -X POST http://localhost:11540/v1/drift/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "feature-monitor",
    "data": [
      [70.1, 100.2, 25.3],
      [72.4, 98.1, 28.7],
      ...
    ]
  }'
```

**Response (no drift):**
```json
{
  "model": "feature-monitor",
  "detector": "ks",
  "result": {
    "is_drift": false,
    "p_value": 0.234,
    "threshold": 0.05,
    "p_values": [0.45, 0.67, 0.12]
  },
  "detection_time_ms": 8.3
}
```

**Response (drift detected):**
```json
{
  "model": "feature-monitor",
  "detector": "ks",
  "result": {
    "is_drift": true,
    "p_value": 0.001,
    "threshold": 0.05,
    "p_values": [0.0001, 0.82, 0.003]
  },
  "detection_time_ms": 8.1
}
```

### Production Monitoring Example

```python
import httpx

async def monitor_batch(batch_data: list) -> bool:
    """Check a batch of production data for drift."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:11540/v1/drift/detect",
            json={"model": "feature-monitor", "data": batch_data}
        )
        result = response.json()["result"]

        if result["is_drift"]:
            # Find which features drifted
            drifted = [
                i for i, p in enumerate(result["p_values"])
                if p < 0.05
            ]
            print(f"ALERT: Drift detected in features {drifted}")
            return True
        return False
```

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/drift/detectors` | GET | List available detector types |
| `/v1/drift/fit` | POST | Fit detector on reference data |
| `/v1/drift/detect` | POST | Check for drift in new data |
| `/v1/drift/load` | POST | Load saved detector |
| `/v1/drift/status/{model}` | GET | Get detector status |
| `/v1/drift/models` | GET | List saved models |
| `/v1/drift/models/{model}` | DELETE | Delete a model |

### Request Parameters (Fit)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | No | Model name (auto-generated if omitted) |
| `detector` | string | Yes | `ks`, `mmd`, or `chi2` |
| `reference_data` | array | Yes | Reference distribution samples |
| `params` | object | No | Detector parameters |
| `feature_names` | array | No | Names for features |

### Detector Parameters

**ks (Kolmogorov-Smirnov):**
- `p_val` (float): Significance threshold (default: 0.05)

**mmd (Maximum Mean Discrepancy):**
- `p_val` (float): Significance threshold (default: 0.05)
- Requires TensorFlow installation

**chi2 (Chi-squared):**
- `p_val` (float): Significance threshold (default: 0.05)
- Best for categorical/discrete features

---

## CatBoost Gradient Boosting

Train gradient boosting models with native categorical feature support and incremental learning capabilities.

:::info CatBoost Advantages
- **Native categorical handling**: No one-hot encoding needed
- **Incremental learning**: Update models without full retraining
- **Ordered boosting**: Reduces overfitting on small datasets
- **GPU acceleration**: When CUDA is available
:::

### Check Availability

```bash
curl http://localhost:11540/v1/catboost/info
```

**Response:**
```json
{
  "available": true,
  "gpu_available": false,
  "version": "1.2.0",
  "features": ["native_categorical", "incremental_learning", "ordered_boosting"]
}
```

### Train a Classifier

CatBoost handles categorical features natively - just pass string values:

```bash
curl -X POST http://localhost:11540/v1/catboost/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn-predictor",
    "model_type": "classifier",
    "data": [
      [25, 3, 89.99, "month-to-month", "credit"],
      [45, 24, 49.99, "two-year", "bank"],
      [32, 12, 79.99, "one-year", "electronic"],
      ...
    ],
    "labels": [1, 0, 0, ...],
    "feature_names": ["age", "tenure", "monthly_charges", "contract", "payment"],
    "cat_features": [3, 4],
    "iterations": 200,
    "depth": 4
  }'
```

**Response:**
```json
{
  "model_id": "churn-predictor",
  "model_type": "classifier",
  "iterations": 200,
  "fit_time_ms": 214.5,
  "samples_fitted": 500
}
```

### Make Predictions

```bash
curl -X POST http://localhost:11540/v1/catboost/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn-predictor",
    "data": [
      [28, 2, 99.99, "month-to-month", "credit"],
      [55, 36, 39.99, "two-year", "bank"]
    ],
    "return_proba": true
  }'
```

**Response:**
```json
{
  "model_id": "churn-predictor",
  "predictions": [
    {"prediction": 1, "probabilities": {"0": 0.35, "1": 0.65}},
    {"prediction": 0, "probabilities": {"0": 0.88, "1": 0.12}}
  ],
  "predict_time_ms": 1.2
}
```

### Incremental Learning

Update your model with new data without full retraining:

```bash
curl -X POST http://localhost:11540/v1/catboost/update \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "churn-predictor",
    "data": [
      [30, 1, 109.99, "month-to-month", "electronic"],
      ...
    ],
    "labels": [1, ...]
  }'
```

**Response:**
```json
{
  "model_id": "churn-predictor",
  "samples_added": 100,
  "trees_before": 200,
  "trees_after": 300,
  "update_time_ms": 21.3
}
```

### Feature Importance

Understand which features drive predictions:

```bash
curl http://localhost:11540/v1/catboost/churn-predictor/importance
```

**Response:**
```json
{
  "model_id": "churn-predictor",
  "importances": [
    {"feature": "monthly_charges", "importance": 30.4},
    {"feature": "age", "importance": 20.3},
    {"feature": "tenure", "importance": 19.3},
    {"feature": "contract", "importance": 16.1},
    {"feature": "payment", "importance": 14.0}
  ],
  "importance_type": "FeatureImportance"
}
```

### Train a Regressor

```bash
curl -X POST http://localhost:11540/v1/catboost/fit \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "price-predictor",
    "model_type": "regressor",
    "data": [[1500, 3, "suburban"], [2200, 4, "urban"], ...],
    "labels": [250000, 450000, ...],
    "feature_names": ["sqft", "bedrooms", "location"],
    "cat_features": [2]
  }'
```

### API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/catboost/info` | GET | Check CatBoost availability |
| `/v1/catboost/fit` | POST | Train classifier or regressor |
| `/v1/catboost/predict` | POST | Make predictions |
| `/v1/catboost/update` | POST | Incremental model update |
| `/v1/catboost/{model}/importance` | GET | Get feature importance |
| `/v1/catboost/models` | GET | List saved models |
| `/v1/catboost/models/{model}` | DELETE | Delete a model |

### Request Parameters (Fit)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | string | Yes | Unique model identifier |
| `model_type` | string | Yes | `classifier` or `regressor` |
| `data` | array | Yes | Training samples (n_samples x n_features) |
| `labels` | array | Yes | Target values |
| `feature_names` | array | No | Names for features |
| `cat_features` | array | No | Indices of categorical features |
| `iterations` | int | No | Number of boosting iterations (default: 100) |
| `depth` | int | No | Tree depth (default: 6) |
| `learning_rate` | float | No | Learning rate (default: auto) |

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

## When to Use What

| Use Case | Tool | Endpoint |
|----------|------|----------|
| Point anomalies (no time context) | PyOD | `/v1/ml/anomaly/*` |
| Time-series anomalies | ADTK | `/v1/adtk/*` |
| Predict future values | Chronos/Darts | `/v1/timeseries/*` |
| ML data quality monitoring | Alibi Detect | `/v1/drift/*` |
| Few-shot text classification | SetFit | `/v1/classifier/*` |
| Tabular data with categoricals | CatBoost | `/v1/catboost/*` |

## Next Steps

- [Anomaly Detection Guide](./anomaly-detection.md) - Complete anomaly detection documentation
- [Universal Runtime Overview](./index.md#universal-runtime) - General runtime configuration
- [API Reference](../api/index.md) - Full API documentation
- [ML Examples](https://github.com/llamafarm/llamafarm/tree/main/examples/ml) - Working code examples
