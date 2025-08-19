# Vision CLI Commands

The LlamaFarm Models framework includes comprehensive image recognition capabilities through the Vision CLI. This document details all available commands for object detection, classification, segmentation, and more.

## Table of Contents
- [Quick Start](#quick-start)
- [System Commands](#system-commands)
- [Detection Commands](#detection-commands)
- [Classification Commands](#classification-commands)
- [Batch Processing](#batch-processing)
- [Performance Commands](#performance-commands)
- [Training Commands](#training-commands)
- [Export Commands](#export-commands)
- [Demo Commands](#demo-commands)

## Quick Start

### Basic Object Detection
```bash
# Detect objects in an image
uv run python cli.py image detect path/to/image.jpg

# With confidence threshold
uv run python cli.py image detect image.jpg --confidence 0.5

# With visualization
uv run python cli.py image detect image.jpg --visualize --output-path result.jpg
```

### Check Hardware
```bash
# See what hardware acceleration is available
uv run python cli.py image info --device
```

## System Commands

### `image info`
Display system information and available models.

```bash
# Show hardware information
uv run python cli.py image info --device

# List available models
uv run python cli.py image info --models

# Show available strategies
uv run python cli.py image info --strategies

# Show all information
uv run python cli.py image info --all
```

**Output:**
- Hardware detection (MPS for Apple Silicon, CUDA for NVIDIA, CPU)
- Available YOLO models (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
- Model categories (detection, classification, segmentation)

### `image setup`
Verify and set up the image recognition system.

```bash
# Test setup
uv run python cli.py image setup --test

# Download default models
uv run python cli.py image setup --download-models

# Full setup with verification
uv run python cli.py image setup --test --download-models
```

## Detection Commands

### `image detect`
Detect objects in images using YOLO models.

```bash
# Basic detection
uv run python cli.py image detect image.jpg

# With options
uv run python cli.py image detect image.jpg \
  --model yolov8s \
  --confidence 0.3 \
  --iou-threshold 0.5 \
  --measure-time \
  --output-format summary

# With visualization
uv run python cli.py image detect image.jpg \
  --visualize \
  --output-path annotated.jpg

# From URL
uv run python cli.py image detect https://example.com/image.jpg

# Multiple output formats
uv run python cli.py image detect image.jpg --output-format json
uv run python cli.py image detect image.jpg --output-format csv
uv run python cli.py image detect image.jpg --output-format text
uv run python cli.py image detect image.jpg --output-format summary
```

**Parameters:**
- `--model`: YOLO model to use (default: yolov8n)
- `--confidence`: Confidence threshold (0.0-1.0, default: 0.5)
- `--iou-threshold`: IoU threshold for NMS (default: 0.45)
- `--measure-time`: Show performance metrics
- `--output-format`: Output format (json, csv, text, summary)
- `--visualize`: Create annotated image
- `--output-path`: Path for visualization output

### `image segment`
Perform instance segmentation on images.

```bash
# Basic segmentation
uv run python cli.py image segment image.jpg

# Save segmentation mask
uv run python cli.py image segment image.jpg \
  --output-mask mask.png \
  --visualize
```

## Classification Commands

### `image classify`
Classify images into categories.

```bash
# Basic classification
uv run python cli.py image classify image.jpg

# Top-k predictions
uv run python cli.py image classify image.jpg \
  --top-k 10 \
  --output-format summary

# With confidence scores
uv run python cli.py image classify image.jpg \
  --top-k 5 \
  --show-confidence
```

**Parameters:**
- `--model`: Classification model (default: yolov8n-cls)
- `--top-k`: Number of top predictions (default: 5)
- `--show-confidence`: Display confidence scores

## Batch Processing

### `image batch-detect`
Process multiple images in a directory.

```bash
# Process all images in directory
uv run python cli.py image batch-detect ./images

# With output directory
uv run python cli.py image batch-detect ./images \
  --output-dir ./results \
  --summary

# With visualizations
uv run python cli.py image batch-detect ./images \
  --output-dir ./results \
  --visualize \
  --summary

# Parallel processing
uv run python cli.py image batch-detect ./images \
  --output-dir ./results \
  --parallel \
  --batch-size 4
```

**Parameters:**
- `--output-dir`: Directory for results
- `--visualize`: Create annotated images
- `--summary`: Show summary statistics
- `--parallel`: Enable parallel processing
- `--batch-size`: Batch size for parallel processing

### `image batch-classify`
Classify multiple images in a directory.

```bash
# Classify all images
uv run python cli.py image batch-classify ./images \
  --output-dir ./classifications \
  --top-k 3
```

## Performance Commands

### `image benchmark`
Benchmark detection performance on your hardware.

```bash
# Quick benchmark
uv run python cli.py image benchmark --runs 5

# Compare models
uv run python cli.py image benchmark \
  --runs 10 \
  --models yolov8n yolov8s yolov8m

# With specific image
uv run python cli.py image benchmark \
  --image test.jpg \
  --runs 20

# Full benchmark suite
uv run python cli.py image benchmark \
  --runs 50 \
  --models all \
  --save-results benchmark.json
```

**Parameters:**
- `--runs`: Number of benchmark runs
- `--models`: Models to benchmark (space-separated or "all")
- `--image`: Specific image to use
- `--save-results`: Save results to file

### `image profile`
Profile memory and compute usage.

```bash
# Profile model
uv run python cli.py image profile --model yolov8n

# Detailed profiling
uv run python cli.py image profile \
  --model yolov8m \
  --detailed \
  --save-report profile.json
```

## Training Commands

### `image train`
Train a custom detector using few-shot learning.

```bash
# Basic training
uv run python cli.py image train ./dataset \
  --base-model yolov8n \
  --epochs 10 \
  --output-dir ./my_detector

# With validation
uv run python cli.py image train ./dataset \
  --base-model yolov8s \
  --epochs 20 \
  --validation-split 0.2 \
  --output-dir ./detector

# Advanced options
uv run python cli.py image train ./dataset \
  --base-model yolov8m \
  --epochs 30 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --augmentation \
  --early-stopping \
  --output-dir ./best_detector
```

**Parameters:**
- `--base-model`: Base YOLO model to fine-tune
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--augmentation`: Enable data augmentation
- `--early-stopping`: Enable early stopping
- `--validation-split`: Validation data percentage

### `image create-dataset`
Create a dataset from images for training.

```bash
# Create YOLO format dataset
uv run python cli.py image create-dataset ./raw_images \
  --output-dir ./dataset \
  --format yolo

# With automatic labeling
uv run python cli.py image create-dataset ./raw_images \
  --output-dir ./dataset \
  --format yolo \
  --auto-label \
  --base-model yolov8n
```

## Export Commands

### `image export`
Export models to different formats.

```bash
# Export to ONNX
uv run python cli.py image export model.pt \
  --format onnx \
  --output-path model.onnx

# Export to CoreML (iOS)
uv run python cli.py image export model.pt \
  --format coreml \
  --output-path model.mlmodel

# Export to TensorFlow Lite
uv run python cli.py image export model.pt \
  --format tflite \
  --output-path model.tflite

# Export to TensorRT
uv run python cli.py image export model.pt \
  --format engine \
  --output-path model.engine \
  --device cuda
```

**Supported Formats:**
- `onnx`: ONNX format for cross-platform deployment
- `coreml`: Apple CoreML for iOS/macOS
- `tflite`: TensorFlow Lite for mobile/edge
- `engine`: TensorRT for NVIDIA GPUs
- `torchscript`: PyTorch TorchScript
- `paddle`: PaddlePaddle format
- `openvino`: Intel OpenVINO

### `image optimize`
Optimize models for deployment.

```bash
# Quantize model
uv run python cli.py image optimize model.pt \
  --quantize int8 \
  --output-path model_int8.pt

# Prune model
uv run python cli.py image optimize model.pt \
  --prune 0.3 \
  --output-path model_pruned.pt
```

## Demo Commands

### `image demo`
Run interactive demonstrations.

```bash
# Quick demo
uv run python cli.py image demo --type quick

# Full demo
uv run python cli.py image demo --type all

# Interactive mode
uv run python cli.py image demo --type quick --interactive

# Specific demo
uv run python cli.py image demo --type detection
uv run python cli.py image demo --type classification
uv run python cli.py image demo --type segmentation
```

### `image download-sample`
Download sample images for testing.

```bash
# Download street scenes
uv run python cli.py image download-sample \
  --output-dir ./samples \
  --type street

# Download all sample types
uv run python cli.py image download-sample \
  --output-dir ./samples \
  --type all

# Specific categories
uv run python cli.py image download-sample \
  --output-dir ./samples \
  --type people animals vehicles
```

## Model Management

### `image list-models`
List all available models with details.

```bash
# All models
uv run python cli.py image list-models --show-size

# By type
uv run python cli.py image list-models --type detection
uv run python cli.py image list-models --type classification
uv run python cli.py image list-models --type segmentation

# With performance info
uv run python cli.py image list-models \
  --show-size \
  --show-performance
```

### `image download-model`
Download specific models.

```bash
# Download specific model
uv run python cli.py image download-model yolov8m

# Download multiple models
uv run python cli.py image download-model yolov8n yolov8s yolov8m

# Download all models of a type
uv run python cli.py image download-model --type detection
```

## Advanced Usage

### Using Custom Models
```bash
# Use custom trained model
uv run python cli.py image detect image.jpg \
  --model ./my_models/custom_detector.pt

# Use model from hub
uv run python cli.py image detect image.jpg \
  --model ultralytics/yolov8x
```

### Pipeline Processing
```bash
# Detect then classify
uv run python cli.py image detect image.jpg --output-format json | \
uv run python cli.py image classify --from-regions

# Chain multiple operations
uv run python cli.py image detect image.jpg | \
uv run python cli.py image segment --from-detections | \
uv run python cli.py image export --format coreml
```

### Integration with Other Tools
```bash
# Save to database
uv run python cli.py image detect image.jpg --output-format json | \
  python save_to_db.py

# Stream processing
watch -n 1 'uv run python cli.py image detect camera.jpg --output-format summary'

# Batch processing with GNU parallel
find ./images -name "*.jpg" | \
  parallel -j 4 'uv run python cli.py image detect {} --output-path {.}_detected.jpg'
```

## Hardware Optimization

### Apple Silicon (M1/M2/M3)
The system automatically detects and uses MPS (Metal Performance Shaders) acceleration:
- Expected performance: 20-30+ FPS with yolov8n
- Optimized for unified memory architecture
- Best with batch_size=1 for real-time processing

### NVIDIA GPUs
Automatically uses CUDA when available:
- Expected performance: 30-50+ FPS with yolov8n
- Supports TensorRT optimization
- Best with larger batch sizes (8-32)

### CPU Optimization
Falls back to CPU with optimizations:
- Uses all available cores
- ONNX Runtime for better performance
- Expected: 5-15 FPS with yolov8n

## Error Handling

Common issues and solutions:

### Import Errors
```bash
# Ensure dependencies are installed
uv sync

# Verify installation
uv run python -c "import ultralytics; print('OK')"
```

### Performance Issues
- Use smaller models (yolov8n instead of yolov8x)
- Reduce image size with `--max-size`
- Enable GPU acceleration if available
- Use batch processing for multiple images

### Memory Issues
- Use `--low-memory` mode
- Process images in smaller batches
- Use model quantization
- Clear cache with `--clear-cache`

## Environment Variables

Control behavior with environment variables:

```bash
# Force CPU usage
FORCE_CPU=1 uv run python cli.py image detect image.jpg

# Set cache directory
YOLO_CACHE_DIR=/tmp/yolo uv run python cli.py image detect image.jpg

# Enable debug logging
DEBUG=1 uv run python cli.py image detect image.jpg

# Set number of threads
OMP_NUM_THREADS=4 uv run python cli.py image detect image.jpg
```

## Examples

### Real-time Camera Processing
```bash
# Process webcam feed
uv run python cli.py image detect --camera 0 --realtime

# IP camera
uv run python cli.py image detect --stream rtsp://camera.local:554
```

### Production Pipeline
```bash
#!/bin/bash
# Production image processing pipeline

INPUT_DIR="./incoming"
OUTPUT_DIR="./processed"
ARCHIVE_DIR="./archive"

# Process new images
for img in "$INPUT_DIR"/*.jpg; do
  basename=$(basename "$img")
  
  # Detect objects
  uv run python cli.py image detect "$img" \
    --confidence 0.6 \
    --output-format json > "$OUTPUT_DIR/${basename%.jpg}_detections.json"
  
  # Create visualization
  uv run python cli.py image detect "$img" \
    --visualize \
    --output-path "$OUTPUT_DIR/${basename%.jpg}_annotated.jpg"
  
  # Move to archive
  mv "$img" "$ARCHIVE_DIR/"
done
```

### Custom Integration
```python
# Python integration example
import subprocess
import json

def detect_objects(image_path):
    """Detect objects using CLI."""
    result = subprocess.run([
        "uv", "run", "python", "cli.py", "image", "detect",
        image_path, "--output-format", "json"
    ], capture_output=True, text=True)
    
    return json.loads(result.stdout)

# Use in application
detections = detect_objects("image.jpg")
for obj in detections:
    print(f"Found {obj['label']} with {obj['confidence']:.2%} confidence")
```

## Performance Tips

1. **Model Selection**
   - Start with yolov8n for speed
   - Use yolov8s/m for balanced performance
   - Use yolov8l/x only when accuracy is critical

2. **Batch Processing**
   - Process multiple images together
   - Use `--parallel` for independent images
   - Adjust `--batch-size` based on memory

3. **Optimization**
   - Enable GPU acceleration when available
   - Use model quantization for edge devices
   - Cache models with `--cache-models`

4. **Image Preprocessing**
   - Resize large images with `--max-size`
   - Use consistent image formats (JPEG/PNG)
   - Preprocess in batches when possible

## See Also

- [Demo Script](demos/demo_4_vision_simple.py) - Interactive demonstration
- [Component Documentation](components/image_recognizers/README.md) - Technical details
- [Training Guide](docs/training.md) - Custom model training
- [Deployment Guide](docs/deployment.md) - Production deployment

## Support

For issues or questions:
- Check hardware with `uv run python cli.py image info --device`
- Run setup test with `uv run python cli.py image setup --test`
- Enable debug mode with `DEBUG=1` environment variable
- See full help with `uv run python cli.py image --help`