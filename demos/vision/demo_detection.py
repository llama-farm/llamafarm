#!/usr/bin/env python3
"""
Demo: Object Detection with YOLO

Demonstrates:
- Loading images from URL or file
- Running YOLO detection
- Visualizing results with bounding boxes
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx

# LlamaFarm API server (proxies to Universal Runtime)
API_URL = os.environ.get("LLAMAFARM_URL", "http://localhost:14345")
RUNTIME_URL = API_URL  # backwards compat
SERVER_URL = API_URL

# Sample images for demo
SAMPLE_IMAGES = [
    {
        "name": "Zidane Soccer",
        "url": "https://ultralytics.com/images/zidane.jpg",
        "expected": ["person"],
    },
    {
        "name": "Bus Scene",
        "url": "https://ultralytics.com/images/bus.jpg",
        "expected": ["person", "bus"],
    },
]


def download_image(url: str) -> bytes:
    """Download image from URL."""
    print(f"  Downloading: {url}")
    response = httpx.get(url, timeout=30.0)
    response.raise_for_status()
    return response.content


def detect_via_runtime(image_b64: str, model: str = "yolov8n") -> dict:
    """Call runtime detection endpoint directly."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/detect",
        json={
            "model": model,
            "image": image_b64,
            "confidence_threshold": 0.5,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def detect_via_server(image_b64: str, model: str = "yolov8n") -> dict:
    """Call server detection endpoint (proxies to runtime)."""
    response = httpx.post(
        f"{SERVER_URL}/v1/vision/detect",
        json={
            "model": model,
            "image": image_b64,
            "confidence_threshold": 0.5,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def print_detections(result: dict, name: str):
    """Pretty-print detection results."""
    print(f"\n{'='*60}")
    print(f"🖼️  Image: {name}")
    print(f"{'='*60}")
    
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Inference time: {result.get('inference_time_ms', 0):.1f}ms")
    
    detections = result.get("detections", [])
    print(f"Detections: {len(detections)}")
    
    if detections:
        print("\n📦 Detected Objects:")
        for i, det in enumerate(detections, 1):
            cls = det.get("class", "unknown")
            conf = det.get("confidence", 0)
            box = det.get("box", {})
            
            print(f"  {i}. {cls}")
            print(f"     Confidence: {conf:.1%}")
            print(f"     Box: ({box.get('x1', 0):.0f}, {box.get('y1', 0):.0f}) → "
                  f"({box.get('x2', 0):.0f}, {box.get('y2', 0):.0f})")
    else:
        print("\n  (No objects detected)")


def run_demo(image_path: str | None = None, model: str = "yolov8n", use_server: bool = False):
    """Run the detection demo."""
    detect_fn = detect_via_server if use_server else detect_via_runtime
    print("🔍 LlamaFarm Vision - Object Detection Demo")
    print("=" * 60)

    if image_path:
        # Use provided image
        print(f"\nUsing custom image: {image_path}")
        with open(image_path, "rb") as f:
            image_data = f.read()
        image_b64 = base64.b64encode(image_data).decode()

        print(f"\nDetecting with {model}...")
        result = detect_fn(image_b64, model)
        print_detections(result, Path(image_path).name)
    else:
        # Use sample images
        print(f"\nUsing {len(SAMPLE_IMAGES)} sample images...")

        for sample in SAMPLE_IMAGES:
            try:
                image_data = download_image(sample["url"])
                image_b64 = base64.b64encode(image_data).decode()

                print(f"\nDetecting with {model}...")
                result = detect_fn(image_b64, model)
                print_detections(result, sample["name"])
                
                # Verify expected classes
                detected_classes = {d["class"] for d in result.get("detections", [])}
                expected = set(sample["expected"])
                if expected & detected_classes:
                    print(f"✅ Found expected: {expected & detected_classes}")
                else:
                    print(f"⚠️  Expected {expected} but got {detected_classes}")
                    
            except Exception as e:
                print(f"❌ Error processing {sample['name']}: {e}")
    
    print("\n" + "=" * 60)
    print("✨ Detection demo complete!")


def main():
    parser = argparse.ArgumentParser(description="Vision Detection Demo")
    parser.add_argument("--image", "-i", help="Path to image file")
    parser.add_argument("--model", "-m", default="yolov8n", 
                       help="YOLO model variant (yolov8n, yolov8s, etc.)")
    parser.add_argument("--server", action="store_true",
                       help="Use server endpoint instead of runtime")
    args = parser.parse_args()
    
    try:
        run_demo(args.image, args.model, args.server)
    except httpx.ConnectError:
        print("❌ Could not connect to server!")
        print("   Make sure the runtime is running:")
        print("   cd runtimes/universal && uv run python server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
