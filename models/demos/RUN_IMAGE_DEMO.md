# 🚀 LlamaFarm Image Recognition - Step-by-Step Demo

This guide will walk you through using the image recognition CLI commands with UV.

## Prerequisites

Make sure you have UV installed and dependencies synced:
```bash
# Sync dependencies (already done)
uv sync
```

## Step-by-Step CLI Commands

### 1. Check System Information

First, let's verify your hardware setup:

```bash
# Check detected hardware
uv run python cli.py image info --device

# See available models
uv run python cli.py image info --models

# See available strategies
uv run python cli.py image info --strategies
```

### 2. Setup and Verification

Verify the image recognition system is working:

```bash
# Quick setup test
uv run python cli.py image setup --test

# Download default models (optional)
uv run python cli.py image setup --download-models
```

### 3. Download Sample Images

Get some test images to work with:

```bash
# Download street scene samples
uv run python cli.py image download-sample --output-dir ./sample_images --type street

# Download all sample types
uv run python cli.py image download-sample --output-dir ./sample_images --type all
```

### 4. Object Detection

Run object detection on an image:

```bash
# Basic detection
uv run python cli.py image detect ./sample_images/street_scene.jpg

# Detection with options
uv run python cli.py image detect ./sample_images/street_scene.jpg \
  --confidence 0.3 \
  --measure-time \
  --output-format summary

# Detection with visualization
uv run python cli.py image detect ./sample_images/street_scene.jpg \
  --visualize \
  --output-path ./detection_result.jpg

# Detect from URL
uv run python cli.py image detect https://ultralytics.com/images/bus.jpg \
  --confidence 0.5
```

### 5. Different Output Formats

Get results in various formats:

```bash
# JSON output (for programmatic use)
uv run python cli.py image detect ./sample_images/street_scene.jpg \
  --output-format json > detections.json

# CSV output (for spreadsheets)
uv run python cli.py image detect ./sample_images/street_scene.jpg \
  --output-format csv > detections.csv

# Text output (human readable)
uv run python cli.py image detect ./sample_images/street_scene.jpg \
  --output-format text

# Summary output (default)
uv run python cli.py image detect ./sample_images/street_scene.jpg \
  --output-format summary
```

### 6. Batch Processing

Process multiple images at once:

```bash
# Process all images in a directory
uv run python cli.py image batch-detect ./sample_images \
  --output-dir ./batch_results \
  --summary

# Batch with visualizations
uv run python cli.py image batch-detect ./sample_images \
  --output-dir ./batch_results_viz \
  --visualize \
  --summary

# Parallel processing (faster)
uv run python cli.py image batch-detect ./sample_images \
  --output-dir ./batch_parallel \
  --parallel \
  --batch-size 4
```

### 7. Performance Benchmark

Test performance on your hardware:

```bash
# Quick benchmark
uv run python cli.py image benchmark --runs 5

# Compare different models
uv run python cli.py image benchmark \
  --runs 10 \
  --models yolov8n yolov8s yolov8m

# Benchmark with specific image
uv run python cli.py image benchmark \
  --image ./sample_images/street_scene.jpg \
  --runs 20
```

### 8. List Available Models

See all available models with sizes:

```bash
# All models
uv run python cli.py image list-models --show-size

# Detection models only
uv run python cli.py image list-models --type detection --show-size

# Classification models
uv run python cli.py image list-models --type classification

# Segmentation models
uv run python cli.py image list-models --type segmentation
```

### 9. Image Classification

Classify images (requires classification model):

```bash
# Classify an image
uv run python cli.py image classify ./sample_images/street_scene.jpg \
  --top-k 5 \
  --output-format summary
```

### 10. Interactive Demo

Run the built-in interactive demo:

```bash
# Quick demo
uv run python cli.py image demo --type quick

# Full demo
uv run python cli.py image demo --type all

# Interactive mode
uv run python cli.py image demo --type quick --interactive
```

## Advanced Usage

### Training a Custom Detector (Few-Shot Learning)

```bash
# Create a dataset from images
uv run python cli.py image create-dataset ./my_images \
  --output-dir ./my_dataset \
  --format yolo

# Train a detector
uv run python cli.py image train ./my_dataset \
  --base-model yolov8n \
  --epochs 10 \
  --output-dir ./my_detector
```

### Model Export

Export models to different formats:

```bash
# Export to ONNX
uv run python cli.py image export yolov8n.pt \
  --format onnx \
  --output-path model.onnx

# Export to CoreML (for iOS)
uv run python cli.py image export yolov8n.pt \
  --format coreml \
  --output-path model.mlmodel

# Export to TensorFlow Lite
uv run python cli.py image export yolov8n.pt \
  --format tflite \
  --output-path model.tflite
```

## Python Script Examples

### Using CLI Commands in Python

```python
import subprocess

# Run detection
result = subprocess.run([
    "uv", "run", "python", "cli.py", "image", "detect",
    "image.jpg", "--output-format", "json"
], capture_output=True, text=True)

import json
detections = json.loads(result.stdout)
```

### Direct Python Usage

```python
from components.factory import ImageRecognizerFactory

# Create recognizer
config = {"type": "yolo", "config": {"version": "yolov8n"}}
recognizer = ImageRecognizerFactory.create(config)

# Detect objects
detections = recognizer.detect("image.jpg", confidence=0.5)

# Print results
for det in detections:
    print(f"{det.label}: {det.confidence:.2%} at {det.bbox}")
```

## Troubleshooting

### If you get import errors:
```bash
# Ensure dependencies are installed
uv sync

# Check installation
uv run python -c "import ultralytics; print('OK')"
```

### If detection is slow:
- Use smaller models (yolov8n instead of yolov8x)
- Reduce image size
- Enable GPU acceleration if available

### To see all available options:
```bash
# Main help
uv run python cli.py image --help

# Command-specific help
uv run python cli.py image detect --help
uv run python cli.py image benchmark --help
```

## Next Steps

1. Try detecting objects in your own images
2. Experiment with different confidence thresholds
3. Train a custom detector for your specific use case
4. Integrate image recognition into your workflows

## Performance Tips

- **Apple Silicon (M1/M2/M3)**: Automatically uses MPS acceleration
- **NVIDIA GPUs**: Automatically uses CUDA if available
- **CPU**: Use smaller models (yolov8n) for better performance

Enjoy using LlamaFarm Image Recognition! 🎉