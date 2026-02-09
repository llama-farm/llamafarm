# Vision Demos

Interactive demos for the LlamaFarm Vision system.

## Prerequisites

1. Start the Universal Runtime:
   ```bash
   cd runtimes/universal
   uv run python server.py
   ```

2. Start the LlamaFarm Server:
   ```bash
   cd server
   uv run python main.py
   ```

## Demos

### 1. Basic Detection Demo
Detects objects in sample images.

```bash
cd demos/vision
python demo_detection.py
```

### 2. Classification Demo
Zero-shot image classification with custom labels.

```bash
python demo_classification.py
```

### 3. Embedding & Similarity Demo
Generate embeddings and compute image similarity.

```bash
python demo_embeddings.py
```

### 4. Streaming Detection Demo
Simulates real-time video analysis with cooldown.

```bash
python demo_streaming.py
```

### 5. Training Demo
Fine-tune YOLO on a small custom dataset.

```bash
python demo_training.py
```

### 6. Full Pipeline Demo
End-to-end: detect → classify → store → train.

```bash
python demo_full_pipeline.py
```

## Sample Images

The demos automatically download sample images from:
- Ultralytics test images (COCO objects)
- Public domain images

Or provide your own:
```bash
python demo_detection.py --image /path/to/your/image.jpg
```
