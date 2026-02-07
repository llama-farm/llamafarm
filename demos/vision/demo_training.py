#!/usr/bin/env python3
"""
Demo: YOLO Fine-Tuning Pipeline

Demonstrates:
1. Creating a small training dataset
2. Starting a fine-tuning job
3. Monitoring training progress
4. Testing the fine-tuned model

Note: This demo creates a minimal dataset for demonstration.
Real training requires more data (100+ images per class).
"""

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np

RUNTIME_URL = "http://localhost:11540"

# Sample images for mini-dataset
SAMPLE_IMAGES = [
    {"url": "https://ultralytics.com/images/zidane.jpg", "class": "person"},
    {"url": "https://ultralytics.com/images/bus.jpg", "class": "vehicle"},
]


def download_image(url: str) -> bytes:
    """Download image from URL."""
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.content


def detect(image_b64: str, model: str = "yolov8n") -> dict:
    """Run detection to get bounding boxes."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/detect",
        json={"model": model, "image": image_b64, "confidence_threshold": 0.3},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def start_training(
    dataset_path: str,
    base_model: str = "yolov8n",
    epochs: int = 3,
    batch_size: int = 4,
) -> dict:
    """Start a training job."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/training/start",
        json={
            "base_model": base_model,
            "dataset_path": dataset_path,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def get_training_status() -> dict:
    """Get current training status."""
    response = httpx.get(
        f"{RUNTIME_URL}/v1/vision/training/status",
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def list_training_jobs() -> dict:
    """List all training jobs."""
    response = httpx.get(
        f"{RUNTIME_URL}/v1/vision/training/jobs",
        timeout=30.0,
    )
    response.raise_for_status()
    return response.json()


def create_mini_dataset(output_dir: str) -> str:
    """Create a minimal YOLO-format dataset.
    
    Creates:
    output_dir/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    """
    print("\n📦 Creating mini training dataset...")
    
    output_path = Path(output_dir)
    train_images = output_path / "train" / "images"
    train_labels = output_path / "train" / "labels"
    val_images = output_path / "val" / "images"
    val_labels = output_path / "val" / "labels"
    
    # Create directories
    for d in [train_images, train_labels, val_images, val_labels]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Download images and create labels
    for i, sample in enumerate(SAMPLE_IMAGES):
        print(f"  Processing: {sample['url']}")
        
        # Download image
        image_data = download_image(sample["url"])
        
        # Detect objects to get bounding boxes
        image_b64 = base64.b64encode(image_data).decode()
        result = detect(image_b64)
        
        # Save to train (first) and val (duplicate for demo)
        for subset, img_dir, lbl_dir in [
            ("train", train_images, train_labels),
            ("val", val_images, val_labels),
        ]:
            img_path = img_dir / f"sample_{i}.jpg"
            lbl_path = lbl_dir / f"sample_{i}.txt"
            
            # Save image
            with open(img_path, "wb") as f:
                f.write(image_data)
            
            # Create YOLO-format labels
            # Format: class_id center_x center_y width height (normalized)
            img_size = result.get("image_size", {"width": 640, "height": 480})
            w, h = img_size["width"], img_size["height"]
            
            labels = []
            for det in result.get("detections", []):
                # Map class to our simple schema (0=person, 1=vehicle)
                cls_id = 0 if det["class"] in ["person"] else 1
                
                box = det["box"]
                cx = ((box["x1"] + box["x2"]) / 2) / w
                cy = ((box["y1"] + box["y2"]) / 2) / h
                bw = (box["x2"] - box["x1"]) / w
                bh = (box["y2"] - box["y1"]) / h
                
                labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            
            with open(lbl_path, "w") as f:
                f.write("\n".join(labels))
    
    # Create data.yaml
    data_yaml = f"""
# Mini dataset for training demo
path: {output_path.absolute()}
train: train/images
val: val/images

nc: 2
names:
  0: person
  1: vehicle
"""
    
    yaml_path = output_path / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(data_yaml.strip())
    
    print(f"  ✅ Dataset created at: {output_path}")
    print(f"  Train images: {len(list(train_images.glob('*.jpg')))}")
    print(f"  Val images: {len(list(val_images.glob('*.jpg')))}")
    
    return str(yaml_path)


def run_demo(epochs: int = 3, skip_training: bool = False):
    """Run the training demo."""
    print("🎓 LlamaFarm Vision - Training Pipeline Demo")
    print("=" * 60)
    
    # Create temporary directory for dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        # Step 1: Create dataset
        data_yaml_path = create_mini_dataset(tmpdir)
        
        print(f"\n📄 Dataset config: {data_yaml_path}")
        with open(data_yaml_path) as f:
            print(f"{'─' * 40}")
            print(f"{f.read()}")
            print(f"{'─' * 40}")
        
        if skip_training:
            print("\n⏭️  Skipping training (--skip-training flag)")
            print("  In a real scenario, training would:")
            print("  1. Load the yolov8n base model")
            print("  2. Fine-tune on the custom dataset")
            print("  3. Save weights to runs/train/exp/weights/")
            return
        
        # Step 2: Start training
        print("\n🚀 Step 2: Starting Training")
        print("-" * 40)
        
        try:
            result = start_training(
                dataset_path=data_yaml_path,
                base_model="yolov8n",
                epochs=epochs,
                batch_size=2,
            )
            
            if "job_id" in result:
                job_id = result["job_id"]
                print(f"  Job ID: {job_id}")
                print(f"  Status: {result.get('status', 'unknown')}")
                
                # Step 3: Monitor training
                print("\n⏳ Step 3: Monitoring Training")
                print("-" * 40)
                
                for _ in range(30):  # Poll for up to 30 seconds
                    status = get_training_status()
                    active = status.get("active_jobs", [])
                    
                    if not active:
                        print("  Training completed!")
                        break
                    
                    for job in active:
                        progress = job.get("progress", 0)
                        epoch = job.get("current_epoch", 0)
                        print(f"\r  Epoch {epoch}/{epochs} - {progress:.1%}", end="")
                    
                    time.sleep(1)
                
                print()
            else:
                print(f"  Result: {result}")
                
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 501:
                print("\n⚠️  Training endpoint not fully implemented")
                print("  This demo shows the expected workflow:")
                print("  1. Create YOLO-format dataset")
                print("  2. Call /v1/vision/training/start")
                print("  3. Monitor with /v1/vision/training/status")
                print("  4. Get results with /v1/vision/training/jobs")
            else:
                raise
        
        # Step 4: Show what would happen
        print("\n📊 Training Results (Example)")
        print("-" * 40)
        print("""
  After training completes, you would see:
  
  ┌────────────────────────────────────────────┐
  │  Training Complete!                        │
  │                                            │
  │  Metrics:                                  │
  │    • mAP50: 0.85                          │
  │    • mAP50-95: 0.62                       │
  │    • Precision: 0.89                      │
  │    • Recall: 0.78                         │
  │                                            │
  │  Model saved to:                          │
  │    runs/train/exp/weights/best.pt         │
  │    runs/train/exp/weights/last.pt         │
  └────────────────────────────────────────────┘
  
  You can then use the fine-tuned model:
  
    curl -X POST http://localhost:11540/v1/vision/detect \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "runs/train/exp/weights/best.pt",
        "image": "..."
      }'
""")
    
    print("\n✨ Training demo complete!")
    print("\n💡 For real training:")
    print("   • Use 100+ images per class")
    print("   • Train for 50-100 epochs")
    print("   • Use data augmentation")
    print("   • Monitor val metrics to prevent overfitting")


def main():
    parser = argparse.ArgumentParser(description="YOLO Training Demo")
    parser.add_argument("--epochs", "-e", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip actual training (just create dataset)")
    args = parser.parse_args()
    
    try:
        run_demo(args.epochs, args.skip_training)
    except httpx.ConnectError:
        print("❌ Could not connect to runtime!")
        print("   Make sure the runtime is running:")
        print("   cd runtimes/universal && uv run python server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
