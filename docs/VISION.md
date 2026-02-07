# LlamaFarm Vision System

A comprehensive computer vision pipeline for object detection, image classification, embeddings, and real-time streaming analysis.

## Overview

The Vision system provides:

- **Object Detection** - YOLO-based detection (v8/v11) with 80+ COCO classes
- **Image Classification** - CLIP zero-shot classification with custom labels
- **Image Embeddings** - CLIP embeddings for similarity search and RAG
- **Streaming Detection** - Real-time video analysis with cooldown and review queues
- **Training Pipeline** - Fine-tune YOLO on custom datasets
- **Storage Layer** - SQLite-based image and detection storage with retention policies

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LlamaFarm Server (:14345)                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │
│  │ /vision/detect │  │/vision/classify│  │ /vision/embed  │            │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘            │
│          │                   │                   │                      │
│          └───────────────────┴───────────────────┘                      │
│                              │                                          │
│                    ┌─────────▼─────────┐                               │
│                    │  Vision Services  │                               │
│                    └─────────┬─────────┘                               │
└──────────────────────────────┼──────────────────────────────────────────┘
                               │ HTTP
┌──────────────────────────────▼──────────────────────────────────────────┐
│                      Universal Runtime (:11540)                         │
│                                                                         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐                │
│  │ YOLO Model  │  │ CLIP Model   │  │ Streaming      │                │
│  │ (Detection) │  │ (Class/Embed)│  │ Detector       │                │
│  └──────┬──────┘  └──────┬───────┘  └───────┬────────┘                │
│         │                │                   │                         │
│  ┌──────▼────────────────▼───────────────────▼────────┐               │
│  │              Vision Base Classes                    │               │
│  │  DetectionModel | ClassificationModel | Embedding   │               │
│  └────────────────────────────────────────────────────┘               │
│                                                                         │
│  ┌────────────────────────────────────────────────────┐               │
│  │                   Storage Layer                     │               │
│  │            ImageStore | RetentionManager            │               │
│  └────────────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Start the Runtime and Server

```bash
# Terminal 1: Start Universal Runtime
cd runtimes/universal
uv run python server.py

# Terminal 2: Start LlamaFarm Server
cd server
uv run python main.py
```

### 2. Test Detection

```bash
# Detect objects in an image
curl -X POST http://localhost:14345/v1/vision/detect \
  -H "Content-Type: application/json" \
  -d '{
    "model": "yolov8n",
    "image_url": "https://ultralytics.com/images/zidane.jpg"
  }'
```

### 3. Test Classification

```bash
# Zero-shot classify an image
curl -X POST http://localhost:14345/v1/vision/classify \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clip-vit-base",
    "image_url": "https://example.com/cat.jpg",
    "classes": ["cat", "dog", "bird", "fish"]
  }'
```

### 4. Generate Embeddings

```bash
# Get image embedding
curl -X POST http://localhost:14345/v1/vision/embed \
  -H "Content-Type: application/json" \
  -d '{
    "model": "clip-vit-base",
    "images": ["base64_encoded_image_here"]
  }'
```

## API Reference

### Detection Endpoint

**POST** `/v1/vision/detect`

Detect objects in an image using YOLO.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | Yes | Model ID (yolov8n, yolov8s, etc.) |
| image | string | Yes* | Base64-encoded image |
| image_url | string | Yes* | URL to fetch image from |
| confidence_threshold | float | No | Minimum confidence (default: 0.5) |
| classes | string[] | No | Filter to specific classes |

*One of `image` or `image_url` required

**Response:**
```json
{
  "object": "detection",
  "model": "yolov8n",
  "inference_time_ms": 45.2,
  "image_size": {"width": 640, "height": 480},
  "detections": [
    {
      "class": "person",
      "confidence": 0.95,
      "box": {"x1": 100, "y1": 50, "x2": 300, "y2": 400}
    }
  ]
}
```

### Classification Endpoint

**POST** `/v1/vision/classify`

Zero-shot image classification using CLIP.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | Yes | Model ID (clip-vit-base, siglip-base) |
| image | string | Yes* | Base64-encoded image |
| image_url | string | Yes* | URL to fetch image from |
| classes | string[] | Yes | Classes to classify against |
| top_k | int | No | Number of top results (default: 5) |

**Response:**
```json
{
  "object": "classification",
  "model": "clip-vit-base",
  "inference_time_ms": 32.1,
  "class": "cat",
  "confidence": 0.89,
  "scores": {
    "cat": 0.89,
    "dog": 0.08,
    "bird": 0.03
  }
}
```

### Embedding Endpoint

**POST** `/v1/vision/embed`

Generate embeddings for images or text.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| model | string | Yes | Model ID |
| images | string[] | Yes* | Base64-encoded images |
| texts | string[] | Yes* | Text strings to embed |

*One of `images` or `texts` required

**Response:**
```json
{
  "object": "embedding",
  "model": "clip-vit-base",
  "inference_time_ms": 25.0,
  "dimensions": 512,
  "embeddings": [
    [0.023, -0.156, 0.089, ...]
  ]
}
```

### Streaming Endpoints

**POST** `/v1/vision/streaming/start`

Start a streaming detection session.

```json
{
  "model": "yolov8n",
  "target_fps": 1.0,
  "confidence_threshold": 0.7,
  "action_classes": ["person", "car"],
  "cooldown_seconds": 5.0
}
```

**POST** `/v1/vision/streaming/frame/{session_id}`

Process a single frame in an active session.

**POST** `/v1/vision/streaming/stop/{session_id}`

Stop a session and get statistics.

**GET** `/v1/vision/streaming/sessions`

List all active sessions.

### Training Endpoints

**POST** `/v1/vision/training/start`

Start a YOLO fine-tuning job.

```json
{
  "base_model": "yolov8n",
  "dataset_path": "/path/to/dataset",
  "epochs": 10,
  "batch_size": 16,
  "output_name": "my_custom_model"
}
```

**GET** `/v1/vision/training/status`

Get current training job status.

**GET** `/v1/vision/training/jobs`

List all training jobs.

## Model Variants

### YOLO Detection Models

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| yolov8n | 6MB | Fast | Good | Real-time, edge devices |
| yolov8s | 22MB | Fast | Better | General purpose |
| yolov8m | 52MB | Medium | High | Balanced |
| yolov8l | 87MB | Slow | Higher | Accuracy critical |
| yolov8x | 137MB | Slowest | Best | Maximum accuracy |
| yolov11n | 5MB | Fastest | Good | Latest architecture |
| yolov11s | 18MB | Fast | Better | Latest, general |
| yolov11m | 38MB | Medium | High | Latest, balanced |

### CLIP Classification Models

| Model | Embedding Dim | Speed | Accuracy |
|-------|---------------|-------|----------|
| clip-vit-base | 512 | Fast | Good |
| clip-vit-base-16 | 512 | Fast | Good |
| clip-vit-large | 768 | Medium | Better |
| siglip-base | 768 | Fast | Better (for classification) |
| siglip-large | 1024 | Medium | Best |

## Streaming Detection

The streaming system is designed for real-time video analysis with intelligent filtering:

### Session Lifecycle

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   START     │────▶│   PROCESS    │────▶│    STOP     │
│  session    │     │   frames     │     │  session    │
└─────────────┘     └──────────────┘     └─────────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
         ┌────────┐ ┌──────────┐ ┌─────────┐
         │   OK   │ │  ACTION  │ │ REVIEW  │
         │(ignore)│ │(trigger) │ │(queue)  │
         └────────┘ └──────────┘ └─────────┘
```

### Status Types

- **OK** - No significant detections or cooldown active
- **ACTION** - High-confidence detection, trigger automation
- **REVIEW** - Uncertain detection, queue for human review

### Cooldown Logic

Prevents alert fatigue by suppressing repeated triggers:

```python
# Example: Person detection with 5-second cooldown
config = StreamingConfig(
    action_classes=["person"],
    confidence_threshold=0.7,
    cooldown_seconds=5.0,
)

# Frame 1 (t=0): Person detected (0.9) → ACTION
# Frame 2 (t=1): Person detected (0.85) → OK (suppressed)
# Frame 3 (t=6): Person detected (0.88) → ACTION (cooldown expired)
```

## Training Pipeline

### Dataset Format

YOLO training requires datasets in this structure:

```
dataset/
├── data.yaml          # Dataset config
├── train/
│   ├── images/        # Training images
│   └── labels/        # YOLO format labels
├── val/
│   ├── images/        # Validation images
│   └── labels/        # Validation labels
└── test/              # Optional test set
    ├── images/
    └── labels/
```

**data.yaml:**
```yaml
path: /path/to/dataset
train: train/images
val: val/images
test: test/images

names:
  0: cat
  1: dog
  2: bird
```

**Label format (one .txt per image):**
```
# class_id center_x center_y width height (normalized 0-1)
0 0.5 0.5 0.4 0.6
1 0.2 0.3 0.15 0.2
```

### Incremental Training

Fine-tune from a checkpoint:

```python
# Start with pre-trained model
result = await training_service.start_training(
    base_model="yolov8n",
    dataset_path="/data/custom",
    epochs=10,
)

# Later: continue training
result = await training_service.start_training(
    base_model="/runs/train/exp/weights/last.pt",  # Previous checkpoint
    dataset_path="/data/custom_v2",
    epochs=5,
)
```

## Storage Layer

### Image Store

SQLite-based storage for images and detections:

```python
from storage.image_store import ImageStore

store = ImageStore(db_path="vision.db")
store.initialize()

# Store image
image_id = store.store_image(
    image_data=frame_bytes,
    source="front_camera",
    metadata={"timestamp": time.time()}
)

# Store detection
store.store_detection(
    image_id=image_id,
    class_name="person",
    confidence=0.95,
    box={"x1": 100, "y1": 50, "x2": 300, "y2": 400},
    model_id="yolov8n"
)

# Human label
store.store_label(
    image_id=image_id,
    class_name="employee",
    annotator="reviewer_1",
    box={"x1": 100, "y1": 50, "x2": 300, "y2": 400}
)
```

### Retention Policies

Automatic cleanup of old data:

```python
from storage.retention import RetentionManager

manager = RetentionManager(store)

# Delete images older than 7 days
manager.apply_max_age(max_hours=168)

# Keep only most recent 10,000 images
manager.apply_max_count(max_count=10000)

# Limit storage to 1GB
manager.apply_max_size(max_bytes=1024*1024*1024)

# Run all policies
result = manager.run_all(
    max_hours=168,
    max_count=10000,
    max_bytes=1024*1024*1024,
)
```

## Integration Examples

### Security Camera Pipeline

```python
import asyncio
from llamafarm import VisionClient

async def security_monitor():
    client = VisionClient("http://localhost:14345")
    
    # Start streaming session
    session = await client.start_stream(
        model="yolov8n",
        action_classes=["person"],
        cooldown_seconds=30.0,
        confidence_threshold=0.8,
    )
    
    async for frame in camera.stream():
        result = await client.process_frame(session.id, frame)
        
        if result.status == "action":
            await send_alert(f"Person detected: {result.detections}")
        elif result.status == "review":
            await queue_for_review(result.image_id)
```

### Image Search with RAG

```python
# Build index from images
embeddings = []
for image_path in image_paths:
    with open(image_path, "rb") as f:
        result = await client.embed(images=[f.read()])
        embeddings.append(result.embeddings[0])

# Save to vector store
vector_store.add(embeddings, image_paths)

# Search by text
query_embedding = await client.embed(texts=["a red sports car"])
similar = vector_store.search(query_embedding, k=10)
```

### Active Learning Loop

```python
# 1. Run detection on unlabeled images
for image in unlabeled_images:
    result = await client.detect(image, model="custom_model")
    
    for det in result.detections:
        if det.confidence < 0.7:
            # Queue uncertain detections for review
            await review_queue.add(image, det)

# 2. Human reviews and labels
labels = await review_queue.get_labeled()

# 3. Add to training dataset
dataset.add_samples(labels)

# 4. Retrain model
await client.train(
    base_model="custom_model",
    dataset_path=dataset.path,
    epochs=5,
)
```

## Performance Tips

### Batch Processing

Process multiple images in one call for better throughput:

```python
# Instead of:
for img in images:
    await client.embed(images=[img])

# Do:
results = await client.embed(images=images)  # All at once
```

### Model Selection

- Use **yolov8n** for real-time (<30ms inference)
- Use **yolov8m** for accuracy/speed balance
- Use **siglip-base** for classification (better than CLIP)

### Device Selection

```python
# Automatic device selection (recommended)
model = YOLOModel("yolov8n", device="auto")

# Force specific device
model = YOLOModel("yolov8n", device="mps")  # Apple Silicon
model = YOLOModel("yolov8n", device="cuda")  # NVIDIA GPU
model = YOLOModel("yolov8n", device="cpu")   # CPU only
```

## Troubleshooting

### Common Issues

**"Model not found"**
- First detection call downloads the model (~6MB for yolov8n)
- Check internet connection and disk space

**"CUDA out of memory"**
- Reduce batch size
- Use smaller model (yolov8n instead of yolov8x)
- Process images one at a time

**"Slow inference"**
- Check device is using GPU: `model.get_model_info()["device"]`
- Install CUDA/MPS support: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

**"Classification always same class"**
- Ensure classes are semantically meaningful
- Try more specific prompts: "a photo of a {}" template
- Use siglip model for better accuracy

### Logs

Enable debug logging:

```python
import logging
logging.getLogger("models.yolo_model").setLevel(logging.DEBUG)
logging.getLogger("models.clip_model").setLevel(logging.DEBUG)
```

## Next Steps

- See [demos/](../demos/) for runnable examples
- Check [examples/vision/](../examples/vision/) for integration patterns
- Read [Plan.md](../Plan.md) for full implementation details
