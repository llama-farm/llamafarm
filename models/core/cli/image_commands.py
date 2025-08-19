"""Enhanced Image Recognition CLI for LlamaFarm Models."""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse
import requests
from io import BytesIO

# Import components
from components.factory import ImageRecognizerFactory, ImageTrainerFactory
from components.base import TrainingExample, Detection

logger = logging.getLogger(__name__)


def add_image_commands(subparsers):
    """Add comprehensive image recognition commands to the CLI."""
    
    # Main image command
    image_parser = subparsers.add_parser(
        "image",
        help="Image recognition and processing commands"
    )
    
    image_subparsers = image_parser.add_subparsers(
        dest="image_command",
        help="Image recognition commands"
    )
    
    # --- SETUP & INFO COMMANDS ---
    
    # Setup command
    setup_parser = image_subparsers.add_parser(
        "setup",
        help="Setup and verify image recognition components"
    )
    setup_parser.add_argument(
        "--download-models",
        action="store_true",
        help="Download default YOLO models"
    )
    setup_parser.add_argument(
        "--test",
        action="store_true",
        help="Run a quick test with sample image"
    )
    setup_parser.set_defaults(func=setup_command)
    
    # Info command
    info_parser = image_subparsers.add_parser(
        "info",
        help="Show information about image recognition system"
    )
    info_parser.add_argument(
        "--device",
        action="store_true",
        help="Show detected hardware device"
    )
    info_parser.add_argument(
        "--models",
        action="store_true",
        help="List available models"
    )
    info_parser.add_argument(
        "--strategies",
        action="store_true",
        help="List available strategies"
    )
    info_parser.set_defaults(func=info_command)
    
    # Download sample command
    download_parser = image_subparsers.add_parser(
        "download-sample",
        help="Download sample images for testing"
    )
    download_parser.add_argument(
        "--output-dir",
        default="./sample_images",
        help="Directory to save samples"
    )
    download_parser.add_argument(
        "--type",
        choices=["street", "document", "faces", "animals", "all"],
        default="all",
        help="Type of samples to download"
    )
    download_parser.set_defaults(func=download_sample_command)
    
    # --- DETECTION COMMANDS ---
    
    # Detect command (enhanced)
    detect_parser = image_subparsers.add_parser(
        "detect",
        help="Run object detection on an image"
    )
    detect_parser.add_argument(
        "image_path",
        help="Path to image file or URL"
    )
    detect_parser.add_argument(
        "--strategy",
        default="default_yolo",
        help="Strategy to use"
    )
    detect_parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold"
    )
    detect_parser.add_argument(
        "--output-format",
        choices=["json", "text", "csv", "summary"],
        default="summary",
        help="Output format"
    )
    detect_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization"
    )
    detect_parser.add_argument(
        "--output-path",
        help="Path to save output"
    )
    detect_parser.add_argument(
        "--measure-time",
        action="store_true",
        help="Show timing information"
    )
    detect_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed processing information"
    )
    detect_parser.set_defaults(func=detect_command)
    
    # Classify command
    classify_parser = image_subparsers.add_parser(
        "classify",
        help="Classify an image"
    )
    classify_parser.add_argument(
        "image_path",
        help="Path to image file"
    )
    classify_parser.add_argument(
        "--strategy",
        default="default_yolo_cls",
        help="Strategy to use"
    )
    classify_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top predictions"
    )
    classify_parser.add_argument(
        "--output-format",
        choices=["json", "text", "summary"],
        default="summary",
        help="Output format"
    )
    classify_parser.set_defaults(func=classify_command)
    
    # Batch detect command
    batch_parser = image_subparsers.add_parser(
        "batch-detect",
        help="Process multiple images"
    )
    batch_parser.add_argument(
        "input_dir",
        help="Directory containing images"
    )
    batch_parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save results"
    )
    batch_parser.add_argument(
        "--strategy",
        default="default_yolo",
        help="Strategy to use"
    )
    batch_parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process in parallel"
    )
    batch_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    batch_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualizations"
    )
    batch_parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics"
    )
    batch_parser.set_defaults(func=batch_detect_command)
    
    # --- TRAINING COMMANDS ---
    
    # Train detector command
    train_parser = image_subparsers.add_parser(
        "train",
        help="Train a detector with few-shot learning"
    )
    train_parser.add_argument(
        "examples_dir",
        help="Directory with training examples"
    )
    train_parser.add_argument(
        "--base-model",
        default="yolov8n",
        help="Base model to fine-tune"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs"
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size"
    )
    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--output-dir",
        default="./fine_tuned_detector",
        help="Output directory"
    )
    train_parser.add_argument(
        "--augmentation-factor",
        type=int,
        default=3,
        help="Augmentation factor"
    )
    train_parser.set_defaults(func=train_detector_command)
    
    # Create dataset command
    dataset_parser = image_subparsers.add_parser(
        "create-dataset",
        help="Create a training dataset from images"
    )
    dataset_parser.add_argument(
        "images_dir",
        help="Directory with images"
    )
    dataset_parser.add_argument(
        "--output-dir",
        default="./training_dataset",
        help="Output directory"
    )
    dataset_parser.add_argument(
        "--format",
        choices=["yolo", "coco", "custom"],
        default="yolo",
        help="Dataset format"
    )
    dataset_parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.8,
        help="Train/val split ratio"
    )
    dataset_parser.set_defaults(func=create_dataset_command)
    
    # --- BENCHMARK COMMANDS ---
    
    # Benchmark command
    benchmark_parser = image_subparsers.add_parser(
        "benchmark",
        help="Benchmark performance on your hardware"
    )
    benchmark_parser.add_argument(
        "--image",
        help="Image to use (or will download sample)"
    )
    benchmark_parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs"
    )
    benchmark_parser.add_argument(
        "--models",
        nargs="+",
        default=["yolov8n"],
        help="Models to benchmark"
    )
    benchmark_parser.add_argument(
        "--compare-devices",
        action="store_true",
        help="Compare CPU vs GPU performance"
    )
    benchmark_parser.set_defaults(func=benchmark_command)
    
    # --- MODEL MANAGEMENT ---
    
    # List models command
    list_models_parser = image_subparsers.add_parser(
        "list-models",
        help="List available models"
    )
    list_models_parser.add_argument(
        "--type",
        choices=["detection", "classification", "segmentation", "all"],
        default="all",
        help="Model type"
    )
    list_models_parser.add_argument(
        "--show-size",
        action="store_true",
        help="Show model sizes"
    )
    list_models_parser.set_defaults(func=list_models_command)
    
    # Export model command
    export_parser = image_subparsers.add_parser(
        "export",
        help="Export model to different format"
    )
    export_parser.add_argument(
        "model_path",
        help="Path to model"
    )
    export_parser.add_argument(
        "--format",
        choices=["onnx", "torchscript", "coreml", "tflite", "engine"],
        default="onnx",
        help="Export format"
    )
    export_parser.add_argument(
        "--output-path",
        required=True,
        help="Output path"
    )
    export_parser.set_defaults(func=export_model_command)
    
    # Note: No demo command - use real commands with --verbose for detailed output
    
    return image_parser


# --- COMMAND IMPLEMENTATIONS ---

def setup_command(args):
    """Setup and verify image recognition components."""
    print("🔧 Setting up image recognition system...")
    
    # Check dependencies
    print("\n📦 Checking dependencies...")
    try:
        import ultralytics
        print("✅ ultralytics installed")
    except ImportError:
        print("❌ ultralytics not installed. Run: uv pip install ultralytics")
        return 1
    
    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        if torch.backends.mps.is_available():
            device = "MPS (Apple Silicon)"
        print(f"✅ PyTorch installed (device: {device})")
    except ImportError:
        print("❌ PyTorch not installed. Run: uv sync")
        return 1
    
    # Download models if requested
    if args.download_models:
        print("\n📥 Downloading default models...")
        from ultralytics import YOLO
        for model in ["yolov8n", "yolov8s"]:
            print(f"  Downloading {model}...")
            YOLO(f"{model}.pt")
        print("✅ Models downloaded")
    
    # Run test if requested
    if args.test:
        print("\n🧪 Running quick test...")
        config = {"type": "yolo", "config": {"version": "yolov8n"}}
        recognizer = ImageRecognizerFactory.create(config)
        print(f"✅ Recognizer created with device: {recognizer.device}")
    
    print("\n✅ Setup complete!")
    return 0


def info_command(args):
    """Show system information."""
    import platform
    
    if args.device or (not args.models and not args.strategies):
        print("🖥️  Hardware Information:")
        print(f"  System: {platform.system()}")
        print(f"  Processor: {platform.processor()}")
        
        try:
            import torch
            if torch.cuda.is_available():
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  CUDA: {torch.version.cuda}")
            elif torch.backends.mps.is_available():
                print("  GPU: Apple Silicon (MPS)")
            else:
                print("  GPU: Not available (CPU only)")
        except ImportError:
            print("  GPU: Unknown (PyTorch not installed)")
    
    if args.models:
        print("\n📦 Available Models:")
        models = {
            "Detection": ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            "Classification": ["yolov8n-cls", "yolov8s-cls", "yolov8m-cls"],
            "Segmentation": ["yolov8n-seg", "yolov8s-seg", "yolov8m-seg"]
        }
        for category, model_list in models.items():
            print(f"\n  {category}:")
            for model in model_list:
                print(f"    - {model}")
    
    if args.strategies:
        print("\n📋 Available Strategies:")
        strategies = [
            "default_yolo - Fast object detection",
            "high_accuracy_detection - More accurate detection",
            "mobile_detection - Optimized for mobile",
            "realtime_detection - Optimized for video",
            "apple_silicon_optimized - M1/M2/M3 optimized",
            "cpu_optimized - CPU optimized"
        ]
        for strategy in strategies:
            print(f"  - {strategy}")
    
    return 0


def download_sample_command(args):
    """Download sample images for testing."""
    from PIL import Image
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = {
        "street": [
            ("https://ultralytics.com/images/bus.jpg", "street_scene.jpg"),
            ("https://ultralytics.com/images/zidane.jpg", "people.jpg")
        ],
        "animals": [
            ("https://images.unsplash.com/photo-1611003228941-98852ba62227?w=640", "dog.jpg"),
            ("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg", "animals_farm.jpg")
        ],
        "airplane": [
            ("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg", "airplane_sky.jpg")
        ],
        "tools": [
            ("https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg", "tools_workshop.jpg")
        ]
    }
    
    if args.type == "all":
        download_list = [(url, name) for urls in samples.values() for url, name in urls]
    else:
        download_list = samples.get(args.type, [])
    
    print(f"📥 Downloading {len(download_list)} sample images...")
    
    for url, filename in download_list:
        try:
            print(f"  Downloading {filename}...")
            response = requests.get(url)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img.save(output_dir / filename)
                print(f"  ✅ Saved {filename}")
            else:
                print(f"  ❌ Failed to download {filename}")
        except Exception as e:
            print(f"  ❌ Error downloading {filename}: {e}")
    
    print(f"\n✅ Samples saved to {output_dir}")
    return 0


def detect_command(args):
    """Enhanced object detection command."""
    from PIL import Image
    
    # Verbose output
    if args.verbose:
        print("🔍 Starting Object Detection")
        print("=" * 60)
        print(f"  Image: {args.image_path}")
        print(f"  Model: yolov8n")
        print(f"  Confidence: {args.confidence}")
        print(f"  Output Format: {args.output_format}")
        print("=" * 60)
    
    # Handle URL input
    image_path = args.image_path
    if image_path.startswith(("http://", "https://")):
        if args.verbose:
            print(f"📥 Downloading image from URL: {image_path}")
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
        temp_path = Path("/tmp/downloaded_image.jpg")
        img.save(temp_path)
        image_path = temp_path
        if args.verbose:
            print(f"  ✅ Saved to temporary file: {temp_path}")
    
    # Create recognizer
    config = {
        "type": "yolo",
        "config": {
            "version": "yolov8n",
            "confidence_threshold": args.confidence
        }
    }
    
    if args.verbose:
        print("\n📦 Loading YOLO Model")
        print(f"  Model: yolov8n")
        print(f"  Device: Detecting best available...")
    
    recognizer = ImageRecognizerFactory.create(config)
    
    if args.verbose:
        print(f"  ✅ Model loaded on device: {recognizer.device}")
        print(f"\n🎯 Running Detection")
        print(f"  Processing: {Path(image_path).name}")
    
    # Measure time if requested
    if args.measure_time or args.verbose:
        start_time = time.time()
    
    # Run detection
    detections = recognizer.detect(image_path, confidence=args.confidence)
    
    if args.measure_time or args.verbose:
        elapsed = time.time() - start_time
        if args.verbose:
            print(f"  ✅ Detection complete in {elapsed:.3f}s")
            print(f"  ⚡ Performance: {1/elapsed:.1f} FPS")
            print(f"\n📊 Detection Results")
            print("-" * 40)
        else:
            print(f"⏱️  Detection time: {elapsed:.3f}s")
    
    # Format output
    if args.output_format == "json":
        output = json.dumps([d.dict() for d in detections], indent=2)
        print(output)
    elif args.output_format == "text":
        for d in detections:
            print(f"{d.label}: {d.confidence:.2f} at {d.bbox}")
    elif args.output_format == "csv":
        print("label,confidence,x1,y1,x2,y2")
        for d in detections:
            print(f"{d.label},{d.confidence:.3f},{','.join(map(str, d.bbox))}")
    else:  # summary
        print(f"\n🎯 Detected {len(detections)} objects:")
        object_counts = {}
        for d in detections:
            object_counts[d.label] = object_counts.get(d.label, 0) + 1
        
        for label, count in object_counts.items():
            print(f"  - {count} {label}(s)")
        
        if args.verbose and detections:
            # Show detailed info for each detection
            print(f"\n📋 Detailed Detections:")
            for i, d in enumerate(detections, 1):
                print(f"\n  Detection #{i}:")
                print(f"    Label: {d.label}")
                print(f"    Confidence: {d.confidence:.1%}")
                print(f"    Bounding Box: [{d.bbox[0]:.1f}, {d.bbox[1]:.1f}, {d.bbox[2]:.1f}, {d.bbox[3]:.1f}]")
                if hasattr(d, 'metadata') and d.metadata:
                    print(f"    Area: {d.metadata.get('area', 0):.1f} pixels²")
        
        if args.measure_time or args.verbose:
            print(f"\n⚡ Performance: {1/elapsed:.1f} FPS")
    
    # Save visualization if requested
    if args.visualize:
        viz_path = Path(args.output_path) if args.output_path else Path(image_path).with_suffix(".viz.jpg")
        recognizer.visualize(image_path, detections, viz_path)
        print(f"🖼️  Visualization saved to {viz_path}")
    
    # Save results if output path specified
    if args.output_path and not args.visualize:
        with open(args.output_path, 'w') as f:
            json.dump([d.dict() for d in detections], f, indent=2)
        print(f"💾 Results saved to {args.output_path}")
    
    return 0


def benchmark_command(args):
    """Benchmark performance."""
    from PIL import Image
    
    print("⚡ Performance Benchmark")
    print("=" * 50)
    
    # Get or download test image
    if args.image:
        image_path = Path(args.image)
    else:
        # Download sample
        print("📥 Downloading test image...")
        response = requests.get("https://ultralytics.com/images/bus.jpg")
        img = Image.open(BytesIO(response.content))
        image_path = Path("/tmp/benchmark_image.jpg")
        img.save(image_path)
    
    results = {}
    
    for model_name in args.models:
        print(f"\n📊 Benchmarking {model_name}...")
        
        config = {
            "type": "yolo",
            "config": {"version": model_name}
        }
        
        recognizer = ImageRecognizerFactory.create(config)
        print(f"  Device: {recognizer.device}")
        
        # Warmup
        recognizer.detect(image_path)
        
        # Benchmark runs
        times = []
        for i in range(args.runs):
            start = time.time()
            detections = recognizer.detect(image_path)
            elapsed = time.time() - start
            times.append(elapsed)
            print(f"  Run {i+1}: {elapsed:.3f}s ({len(detections)} objects)")
        
        avg_time = sum(times) / len(times)
        fps = 1 / avg_time
        
        results[model_name] = {
            "avg_time": avg_time,
            "fps": fps,
            "device": recognizer.device
        }
        
        print(f"  Average: {avg_time:.3f}s")
        print(f"  FPS: {fps:.1f}")
    
    # Summary
    print("\n📈 Benchmark Summary:")
    print("-" * 50)
    for model, stats in results.items():
        print(f"{model:15} | {stats['avg_time']:.3f}s | {stats['fps']:.1f} FPS | {stats['device']}")
    
    return 0




# Keep other command implementations from original file...
def classify_command(args):
    """Classify an image."""
    config = {
        "type": "yolo",
        "config": {"version": "yolov8n-cls"}
    }
    
    recognizer = ImageRecognizerFactory.create(config)
    classifications = recognizer.classify(args.image_path, top_k=args.top_k)
    
    if args.output_format == "json":
        print(json.dumps([c.dict() for c in classifications], indent=2))
    elif args.output_format == "text":
        for i, c in enumerate(classifications, 1):
            print(f"{i}. {c.label}: {c.confidence:.3f}")
    else:  # summary
        print(f"\n🏷️  Top {args.top_k} classifications:")
        for i, c in enumerate(classifications, 1):
            bar = "█" * int(c.confidence * 20)
            print(f"  {i}. {c.label:20} {bar} {c.confidence:.1%}")
    
    return 0


def batch_detect_command(args):
    """Process multiple images in batch."""
    import concurrent.futures
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_paths = [p for p in input_dir.iterdir() if p.suffix.lower() in image_extensions]
    
    if not image_paths:
        print(f"❌ No images found in {input_dir}")
        return 1
    
    print(f"📦 Processing {len(image_paths)} images...")
    
    # Create recognizer
    config = {"type": "yolo", "config": {"version": "yolov8n"}}
    recognizer = ImageRecognizerFactory.create(config)
    
    # Process images
    total_objects = 0
    start_time = time.time()
    
    for i, img_path in enumerate(image_paths, 1):
        detections = recognizer.detect(img_path)
        total_objects += len(detections)
        
        # Save results
        result_path = output_dir / f"{img_path.stem}_results.json"
        with open(result_path, 'w') as f:
            json.dump([d.dict() for d in detections], f, indent=2)
        
        # Save visualization if requested
        if args.visualize:
            viz_path = output_dir / f"{img_path.stem}_viz.jpg"
            recognizer.visualize(img_path, detections, viz_path)
        
        print(f"  [{i}/{len(image_paths)}] {img_path.name}: {len(detections)} objects")
    
    elapsed = time.time() - start_time
    
    if args.summary:
        print(f"\n📊 Summary:")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Total objects: {total_objects}")
        print(f"  Avg objects/image: {total_objects/len(image_paths):.1f}")
        print(f"  Total time: {elapsed:.1f}s")
        print(f"  Avg time/image: {elapsed/len(image_paths):.2f}s")
        print(f"  Throughput: {len(image_paths)/elapsed:.1f} images/sec")
    
    print(f"\n✅ Results saved to {output_dir}")
    return 0


def train_detector_command(args):
    """Train a detector with few-shot learning."""
    # Implementation from original file
    print("🎓 Training detector...")
    print(f"  Examples: {args.examples_dir}")
    print(f"  Base model: {args.base_model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Output: {args.output_dir}")
    
    # Create trainer
    config = {
        "type": "yolo",
        "config": {
            "dataset_path": "./yolo_dataset",
            "augmentation_factor": args.augmentation_factor
        }
    }
    
    # Note: Trainer implementation would go here
    print("⚠️  Training not yet implemented in factory")
    return 0


def create_dataset_command(args):
    """Create a training dataset."""
    print(f"📁 Creating dataset from {args.images_dir}")
    print(f"  Format: {args.format}")
    print(f"  Split: {args.split_ratio:.0%} train / {1-args.split_ratio:.0%} val")
    
    # Implementation would go here
    print("⚠️  Dataset creation not yet implemented")
    return 0


def list_models_command(args):
    """List available models."""
    models = {
        "detection": {
            "yolov8n": "6.2MB - Nano (fastest)",
            "yolov8s": "21.5MB - Small",
            "yolov8m": "49.7MB - Medium",
            "yolov8l": "83.7MB - Large",
            "yolov8x": "130.5MB - Extra Large (most accurate)"
        },
        "classification": {
            "yolov8n-cls": "5.0MB - Nano classifier",
            "yolov8s-cls": "20.0MB - Small classifier",
            "yolov8m-cls": "45.0MB - Medium classifier"
        },
        "segmentation": {
            "yolov8n-seg": "6.7MB - Nano segmentation",
            "yolov8s-seg": "23.4MB - Small segmentation",
            "yolov8m-seg": "52.0MB - Medium segmentation"
        }
    }
    
    if args.type == "all":
        for category, model_dict in models.items():
            print(f"\n{category.capitalize()} Models:")
            for model, desc in model_dict.items():
                if args.show_size:
                    print(f"  {model:15} - {desc}")
                else:
                    print(f"  {model}")
    else:
        print(f"\n{args.type.capitalize()} Models:")
        for model, desc in models.get(args.type, {}).items():
            if args.show_size:
                print(f"  {model:15} - {desc}")
            else:
                print(f"  {model}")
    
    return 0


def export_model_command(args):
    """Export model to different format."""
    print(f"📤 Exporting {args.model_path} to {args.format}")
    
    config = {"type": "yolo", "config": {}}
    recognizer = ImageRecognizerFactory.create(config)
    recognizer.load_model(args.model_path)
    
    output_path = Path(args.output_path)
    recognizer.export_model(output_path, format=args.format)
    
    print(f"✅ Model exported to {output_path}")
    return 0


def main():
    """Standalone CLI entry point."""
    parser = argparse.ArgumentParser(description="LlamaFarm Image Recognition CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    add_image_commands(subparsers)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())