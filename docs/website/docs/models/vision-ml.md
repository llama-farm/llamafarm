---
title: Vision & ML Models
sidebar_position: 4
---

# Vision & ML Models

LlamaFarm provides a comprehensive suite of vision and ML models through the Universal Runtime. These models are accessible via the LlamaFarm API proxy, making them easy to integrate into your applications.

## Quick Start

All vision/ML endpoints are available through the LlamaFarm API at `http://localhost:8000/v1/ml/...` (when the server is running). The Universal Runtime runs on port 11540 by default.

```bash
# Start the servers
lf services start

# Or manually:
cd runtimes/universal && uv run python server.py  # Port 11540
cd server && uv run uvicorn main:app              # Port 8000
```

---

## Image Classification

### Zero-Shot Classification (CLIP)

Classify images into arbitrary categories without any training. Just provide labels and get predictions.

**Endpoint:** `POST /v1/ml/vision/classify-zero-shot`

**Use Cases:**
- Document type classification (receipt, invoice, ID card)
- Content moderation (safe/unsafe)
- Quick image categorization
- Photo vs drawing detection

**Example Request:**
```bash
# Encode image to base64
IMAGE_B64=$(base64 -i photo.jpg)

curl -X POST http://localhost:8000/v1/ml/vision/classify-zero-shot \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$IMAGE_B64"'",
    "labels": ["cat", "dog", "bird", "horse"],
    "model": "openai/clip-vit-base-patch32"
  }'
```

**Response:**
```json
{
  "object": "classification",
  "label": "cat",
  "score": 0.87,
  "all_scores": {
    "cat": 0.87,
    "dog": 0.08,
    "bird": 0.03,
    "horse": 0.02
  },
  "model": "openai/clip-vit-base-patch32"
}
```

**Batch Classification:** `POST /v1/ml/vision/classify-zero-shot/batch`
```json
{
  "images": ["<base64_image1>", "<base64_image2>"],
  "labels": ["cat", "dog", "bird"]
}
```

---

### Few-Shot Classification (Trainable)

Train custom classifiers with just 5-50 images per class. Perfect for domain-specific classification.

**Endpoints:**
- `POST /v1/ml/vision/classify/fit` - Train a classifier
- `POST /v1/ml/vision/classify/predict` - Classify an image
- `POST /v1/ml/vision/classify/predict/batch` - Batch classification
- `POST /v1/ml/vision/classify/refine` - Add more training data
- `GET /v1/ml/vision/classify/info/{id}` - Get classifier info
- `DELETE /v1/ml/vision/classify/{id}` - Delete classifier

**Use Cases:**
- Species/breed identification (golden retriever vs labrador)
- Product classification for specific catalogs
- Medical image classification
- Quality control (defect detection)
- Refining low-confidence zero-shot predictions

#### Training a Classifier

```bash
curl -X POST http://localhost:8000/v1/ml/vision/classify/fit \
  -H "Content-Type: application/json" \
  -d '{
    "classifier_id": "dog-breeds",
    "images": [
      "'"$(base64 -i golden1.jpg)"'",
      "'"$(base64 -i golden2.jpg)"'",
      "'"$(base64 -i labrador1.jpg)"'",
      "'"$(base64 -i labrador2.jpg)"'"
    ],
    "labels": ["golden_retriever", "golden_retriever", "labrador", "labrador"],
    "epochs": 100
  }'
```

**Response:**
```json
{
  "object": "few_shot_classifier",
  "classifier_id": "dog-breeds",
  "success": true,
  "num_samples": 4,
  "num_classes": 2,
  "classes": ["golden_retriever", "labrador"],
  "final_accuracy": 1.0,
  "training_time_ms": 1234.5
}
```

#### Predicting with a Trained Classifier

```bash
curl -X POST http://localhost:8000/v1/ml/vision/classify/predict \
  -H "Content-Type: application/json" \
  -d '{
    "classifier_id": "dog-breeds",
    "image": "'"$(base64 -i unknown_dog.jpg)"'"
  }'
```

**Response:**
```json
{
  "object": "classification",
  "classifier_id": "dog-breeds",
  "label": "golden_retriever",
  "score": 0.92,
  "all_scores": {
    "golden_retriever": 0.92,
    "labrador": 0.08
  }
}
```

#### Adding New Classes (Refinement)

```bash
curl -X POST http://localhost:8000/v1/ml/vision/classify/refine \
  -H "Content-Type: application/json" \
  -d '{
    "classifier_id": "dog-breeds",
    "images": [
      "'"$(base64 -i german_shepherd1.jpg)"'",
      "'"$(base64 -i german_shepherd2.jpg)"'"
    ],
    "labels": ["german_shepherd", "german_shepherd"],
    "epochs": 50
  }'
```

**Response:**
```json
{
  "object": "few_shot_classifier",
  "classifier_id": "dog-breeds",
  "success": true,
  "refined_samples": 2,
  "num_classes": 3,
  "classes": ["german_shepherd", "golden_retriever", "labrador"],
  "new_classes_added": ["german_shepherd"],
  "accuracy_on_new_data": 1.0
}
```

---

## Object Detection

### Standard Detection (YOLOS)

Detect common objects (80 COCO classes) with bounding boxes.

**Endpoint:** `POST /v1/ml/vision/detect-objects`

**Use Cases:**
- People counting
- Vehicle detection
- General object localization

```bash
curl -X POST http://localhost:8000/v1/ml/vision/detect-objects \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$(base64 -i street.jpg)"'",
    "threshold": 0.5,
    "labels": ["car", "person"]
  }'
```

**Response:**
```json
{
  "object": "object_detection",
  "objects": [
    {
      "label": "person",
      "score": 0.95,
      "box": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}
    },
    {
      "label": "car",
      "score": 0.88,
      "box": {"x1": 300, "y1": 150, "x2": 500, "y2": 350}
    }
  ],
  "count": 2,
  "image_size": {"width": 640, "height": 480}
}
```

---

### Open-Vocabulary Detection (OWL-ViT)

Detect ANY object using natural language descriptions. No training required.

**Endpoints:**
- `POST /v1/ml/vision/detect-open` - Text-conditioned detection
- `POST /v1/ml/vision/detect-open/batch` - Batch detection
- `POST /v1/ml/vision/detect-open/by-image` - Image-guided detection

**Use Cases:**
- Find specific objects ("a red fire hydrant", "a person wearing a hat")
- Species detection in wildlife photos
- Product localization
- Custom domain detection ("damaged car door", "rust spot")
- Visual search using reference images

#### Text-Conditioned Detection

```bash
curl -X POST http://localhost:8000/v1/ml/vision/detect-open \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$(base64 -i wildlife.jpg)"'",
    "queries": ["a golden retriever", "a german shepherd", "a labrador"],
    "threshold": 0.1,
    "top_k": 5
  }'
```

**Response:**
```json
{
  "object": "open_vocab_detection",
  "objects": [
    {
      "query": "a golden retriever",
      "label": "a golden retriever",
      "score": 0.85,
      "box": {"x1": 100, "y1": 50, "x2": 400, "y2": 350}
    }
  ],
  "count": 1,
  "queries": ["a golden retriever", "a german shepherd", "a labrador"],
  "image_size": {"width": 640, "height": 480}
}
```

**Tips for Better Results:**
- Use descriptive queries: `"a photo of a cat"` works better than just `"cat"`
- Lower threshold (0.05-0.1) for recall, higher (0.3-0.5) for precision
- Combine with few-shot classification for species refinement

#### Image-Guided Detection

Find objects similar to reference images:

```bash
curl -X POST http://localhost:8000/v1/ml/vision/detect-open/by-image \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$(base64 -i target.jpg)"'",
    "query_images": ["'"$(base64 -i reference_cat.jpg)"'"],
    "threshold": 0.9,
    "top_k": 5
  }'
```

**Response:**
```json
{
  "object": "image_guided_detection",
  "objects": [
    {
      "query_index": 0,
      "score": 0.95,
      "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 250}
    }
  ],
  "count": 1,
  "num_queries": 1,
  "image_size": {"width": 640, "height": 480}
}
```

---

## Background Removal (RMBG)

Remove backgrounds from images, producing PNG with transparent background.

**Endpoint:** `POST /v1/ml/vision/segment`

**Use Cases:**
- Product photography
- Portrait background removal
- E-commerce images

```bash
curl -X POST http://localhost:8000/v1/ml/vision/segment \
  -H "Content-Type: application/json" \
  -d '{
    "image": "'"$(base64 -i product.jpg)"'",
    "return_mask": false
  }'
```

**Response:**
```json
{
  "object": "background_removal",
  "image": "<base64_png_with_alpha>",
  "width": 640,
  "height": 480
}
```

---

## Document & OCR

### OCR (Text Extraction)

Extract text from images using multiple backends.

**Endpoint:** `POST /v1/ml/ocr`

**Backends:**
- `surya` - Best accuracy, transformer-based (recommended)
- `easyocr` - Good multilingual support (80+ languages)
- `paddleocr` - Fast, optimized for Asian languages
- `tesseract` - Classic OCR engine

```bash
curl -X POST http://localhost:8000/v1/ml/ocr \
  -H "Content-Type: application/json" \
  -d '{
    "model": "surya",
    "images": ["'"$(base64 -i document.jpg)"'"],
    "languages": ["en"],
    "return_boxes": true
  }'
```

### Document Extraction

Extract structured data from documents (forms, invoices, receipts).

**Endpoint:** `POST /v1/ml/documents/extract`

```bash
curl -X POST http://localhost:8000/v1/ml/documents/extract \
  -H "Content-Type: application/json" \
  -d '{
    "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
    "images": ["'"$(base64 -i receipt.jpg)"'"],
    "task": "extraction"
  }'
```

---

## Text Analysis

### Language Detection

Detect the language of text.

**Endpoint:** `POST /v1/ml/text/detect-language`

```bash
curl -X POST http://localhost:8000/v1/ml/text/detect-language \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "Bonjour le monde", "Hola mundo"]
  }'
```

### PII Detection & Redaction

Detect and redact personally identifiable information.

**Endpoints:**
- `POST /v1/ml/text/pii/detect` - Find PII
- `POST /v1/ml/text/pii/redact` - Redact PII

```bash
curl -X POST http://localhost:8000/v1/ml/text/pii/redact \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Contact John Smith at john@email.com or 555-123-4567",
    "entities": ["PERSON", "EMAIL", "PHONE"]
  }'
```

### Keyword Extraction

Extract keywords and keyphrases from text.

**Endpoint:** `POST /v1/ml/text/keywords`

---

## Time Series

### Forecasting

Forecast future values using Chronos.

**Endpoint:** `POST /v1/ml/timeseries/forecast`

```bash
curl -X POST http://localhost:8000/v1/ml/timeseries/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "values": [100, 120, 115, 130, 145, 140, 160],
    "horizon": 7,
    "model": "amazon/chronos-t5-small"
  }'
```

### Changepoint Detection

Detect significant changes in time series.

**Endpoint:** `POST /v1/ml/timeseries/changepoints`

### Drift Detection

Detect distribution drift in streaming data.

**Endpoint:** `POST /v1/ml/timeseries/drift`

---

## Anomaly Detection

Train anomaly detection models and detect outliers.

**Endpoints:**
- `POST /v1/ml/anomaly/train` - Train model
- `POST /v1/ml/anomaly/detect` - Detect anomalies

```bash
# Train on normal data
curl -X POST http://localhost:8000/v1/ml/anomaly/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "api-monitor",
    "data": [
      {"response_time": 100, "error_rate": 0.01},
      {"response_time": 105, "error_rate": 0.02}
    ]
  }'

# Detect anomalies
curl -X POST http://localhost:8000/v1/ml/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "api-monitor",
    "data": [
      {"response_time": 500, "error_rate": 0.5}
    ]
  }'
```

---

## Table Question Answering

Answer questions about tabular data.

**Endpoint:** `POST /v1/ml/tables/qa`

```bash
curl -X POST http://localhost:8000/v1/ml/tables/qa \
  -H "Content-Type: application/json" \
  -d '{
    "table": {
      "columns": ["Product", "Sales", "Region"],
      "data": [
        ["Widget A", 1000, "North"],
        ["Widget B", 1500, "South"]
      ]
    },
    "questions": ["What is the total sales?", "Which product has higher sales?"]
  }'
```

---

## Python Client Examples

### Using requests

```python
import base64
import requests

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Zero-shot classification
response = requests.post(
    "http://localhost:8000/v1/ml/vision/classify-zero-shot",
    json={
        "image": encode_image("photo.jpg"),
        "labels": ["cat", "dog", "bird"]
    }
)
result = response.json()
print(f"Predicted: {result['label']} ({result['score']:.1%})")

# Train few-shot classifier
response = requests.post(
    "http://localhost:8000/v1/ml/vision/classify/fit",
    json={
        "classifier_id": "my-classifier",
        "images": [encode_image(f"cat{i}.jpg") for i in range(5)] +
                  [encode_image(f"dog{i}.jpg") for i in range(5)],
        "labels": ["cat"] * 5 + ["dog"] * 5,
        "epochs": 100
    }
)
print(f"Trained: {response.json()['classes']}")

# Open-vocabulary detection
response = requests.post(
    "http://localhost:8000/v1/ml/vision/detect-open",
    json={
        "image": encode_image("wildlife.jpg"),
        "queries": ["a deer", "a bird", "a squirrel"],
        "threshold": 0.1
    }
)
for obj in response.json()["objects"]:
    print(f"Found: {obj['query']} at {obj['box']} ({obj['score']:.1%})")
```

### Using httpx (async)

```python
import asyncio
import base64
import httpx

async def classify_images(paths, labels):
    async with httpx.AsyncClient() as client:
        tasks = []
        for path in paths:
            with open(path, "rb") as f:
                image_b64 = base64.b64encode(f.read()).decode()
            tasks.append(
                client.post(
                    "http://localhost:8000/v1/ml/vision/classify-zero-shot",
                    json={"image": image_b64, "labels": labels}
                )
            )
        responses = await asyncio.gather(*tasks)
        return [r.json() for r in responses]

# Run async classification
results = asyncio.run(classify_images(
    ["photo1.jpg", "photo2.jpg", "photo3.jpg"],
    ["cat", "dog", "bird"]
))
```

---

## Workflow: Species Identification Pipeline

Combine multiple models for accurate species identification:

```python
import requests
import base64

def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

BASE_URL = "http://localhost:8000/v1/ml"

# Step 1: Detect animals with open-vocabulary detection
detect_response = requests.post(
    f"{BASE_URL}/vision/detect-open",
    json={
        "image": encode_image("wildlife_photo.jpg"),
        "queries": ["an animal", "a mammal", "a bird"],
        "threshold": 0.1
    }
)
detections = detect_response.json()["objects"]

# Step 2: For each detection, classify species
for det in detections:
    # Crop image to bounding box (use PIL or opencv)
    # cropped_image = crop_image(det["box"])

    # Classify with trained species classifier
    classify_response = requests.post(
        f"{BASE_URL}/vision/classify/predict",
        json={
            "classifier_id": "wildlife-species",
            "image": encode_image("cropped.jpg")  # Use cropped image
        }
    )
    species = classify_response.json()
    print(f"Detected {det['query']}: {species['label']} ({species['score']:.1%})")
```

---

## Model Summary Table

| Model | Endpoint | Training Required | Key Capability |
|-------|----------|-------------------|----------------|
| CLIP | `/vision/classify-zero-shot` | No | Zero-shot classification |
| Few-Shot Classifier | `/vision/classify/fit` | Yes (5-50 images) | Custom classification |
| YOLOS | `/vision/detect-objects` | No | COCO object detection |
| OWL-ViT | `/vision/detect-open` | No | Open-vocabulary detection |
| RMBG | `/vision/segment` | No | Background removal |
| Surya/EasyOCR | `/ocr` | No | Text extraction |
| Donut/LayoutLM | `/documents/extract` | No | Document extraction |
| XLM-RoBERTa | `/text/detect-language` | No | Language detection |
| GLiNER | `/text/pii/redact` | No | PII detection |
| Chronos | `/timeseries/forecast` | No | Forecasting |
| Autoencoder | `/anomaly/train` | Yes | Anomaly detection |
| TAPAS | `/tables/qa` | No | Table QA |
