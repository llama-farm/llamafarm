#!/usr/bin/env python3
"""
Test Vision APIs with Real Images

Usage:
    # Test detection on a single image
    python test_vision_real.py detect path/to/image.jpg
    
    # Test classification
    python test_vision_real.py classify path/to/image.jpg "cat,dog,bird,car,person"
    
    # Test OCR
    python test_vision_real.py ocr path/to/document.png
    
    # Test streaming cascade (interactive)
    python test_vision_real.py stream
    
    # Test with a URL
    python test_vision_real.py detect https://example.com/image.jpg
"""

import argparse
import base64
import json
import sys
from pathlib import Path
from urllib.request import urlopen, Request

import requests

BASE_URL = "http://localhost:14345"
RUNTIME_URL = "http://localhost:11540"


def image_to_base64(image_path: str) -> str:
    """Convert image file or URL to base64 data URI."""
    if image_path.startswith("http://") or image_path.startswith("https://"):
        # Download from URL with User-Agent
        print(f"📥 Downloading from {image_path}...")
        req = Request(image_path, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
        response = urlopen(req)
        image_bytes = response.read()
        content_type = response.headers.get("Content-Type", "image/jpeg")
    else:
        # Read from file
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_bytes = path.read_bytes()
        
        # Detect content type from extension
        ext = path.suffix.lower()
        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        content_type = content_types.get(ext, "image/jpeg")
    
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{content_type};base64,{b64}"


def test_detection(image_path: str, model: str = "yolov8n", confidence: float = 0.5):
    """Test object detection on an image."""
    print(f"\n🎯 Testing Detection")
    print(f"   Image: {image_path}")
    print(f"   Model: {model}")
    print(f"   Confidence threshold: {confidence}")
    print("-" * 50)
    
    image_b64 = image_to_base64(image_path)
    
    # Use multipart form
    response = requests.post(
        f"{BASE_URL}/v1/vision/detect",
        data={
            "image": image_b64,
            "model": model,
            "confidence_threshold": confidence,
        },
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    
    print(f"\n✅ Detection complete ({result.get('inference_time_ms', 0):.1f}ms)")
    print(f"   Model: {result.get('model', 'unknown')}")
    
    detections = result.get("detections", [])
    if not detections:
        print("\n   No objects detected")
    else:
        print(f"\n   Found {len(detections)} object(s):")
        for i, det in enumerate(detections, 1):
            box = det.get("box", {})
            print(f"   {i}. {det['class_name']:15s} {det['confidence']*100:5.1f}%  "
                  f"[{box.get('x1',0):.0f},{box.get('y1',0):.0f} → {box.get('x2',0):.0f},{box.get('y2',0):.0f}]")


def test_classification(image_path: str, classes: str, model: str = "clip-vit-base"):
    """Test zero-shot image classification."""
    print(f"\n🏷️  Testing Classification")
    print(f"   Image: {image_path}")
    print(f"   Classes: {classes}")
    print(f"   Model: {model}")
    print("-" * 50)
    
    image_b64 = image_to_base64(image_path)
    
    response = requests.post(
        f"{BASE_URL}/v1/vision/classify",
        data={
            "image": image_b64,
            "model": model,
            "classes": classes,
            "top_k": 5,
        },
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    
    print(f"\n✅ Classification complete ({result.get('inference_time_ms', 0):.1f}ms)")
    print(f"   Top prediction: {result.get('class_name')} ({result.get('confidence', 0)*100:.1f}%)")
    
    all_scores = result.get("all_scores", {})
    if all_scores:
        print("\n   All scores:")
        for cls, score in sorted(all_scores.items(), key=lambda x: -x[1]):
            bar = "█" * int(score * 30)
            print(f"   {cls:15s} {score*100:5.1f}% {bar}")


def test_ocr(image_path: str, model: str = "surya"):
    """Test OCR text extraction."""
    print(f"\n📄 Testing OCR")
    print(f"   Image: {image_path}")
    print(f"   Model: {model}")
    print("-" * 50)
    
    image_b64 = image_to_base64(image_path)
    
    response = requests.post(
        f"{BASE_URL}/v1/vision/ocr",
        data={
            "images": json.dumps([image_b64]),
            "model": model,
            "languages": "en",
        },
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
        return
    
    result = response.json()
    
    print(f"\n✅ OCR complete")
    print(f"   Model: {result.get('model', 'unknown')}")
    
    for item in result.get("data", []):
        print(f"\n   Page {item.get('index', 0)} (confidence: {item.get('confidence', 0)*100:.1f}%):")
        print("   " + "-" * 40)
        text = item.get("text", "")
        # Indent text
        for line in text.split("\n")[:20]:  # First 20 lines
            print(f"   {line}")
        if text.count("\n") > 20:
            print(f"   ... ({text.count(chr(10)) - 20} more lines)")


def test_streaming(primary_model: str = "yolov8n", secondary_model: str = "yolov8m"):
    """Interactive streaming cascade test."""
    print(f"\n🎬 Streaming Cascade Test")
    print(f"   Primary model: {primary_model}")
    print(f"   Secondary model: {secondary_model}")
    print("-" * 50)
    
    # Start session
    response = requests.post(
        f"{RUNTIME_URL}/v1/vision/stream/start",
        json={
            "model": primary_model,
            "config": {
                "confidence_threshold": 0.7,
                "escalation_threshold": 0.5,
                "cooldown_seconds": 1.0,
                "cascade": {
                    "secondary_model_id": secondary_model,
                    "feedback_to_primary": True,
                    "save_uncertain_images": True,
                },
            },
        },
    )
    
    if response.status_code != 200:
        print(f"❌ Failed to start session: {response.status_code}")
        print(response.text)
        return
    
    session = response.json()
    session_id = session["session_id"]
    print(f"\n✅ Session started: {session_id}")
    
    print("\n📸 Enter image paths to process (or 'quit' to stop):")
    print("   Tip: You can also paste URLs")
    
    try:
        while True:
            image_path = input("\n> ").strip()
            
            if image_path.lower() in ("quit", "exit", "q"):
                break
            
            if not image_path:
                continue
            
            try:
                image_b64 = image_to_base64(image_path)
            except Exception as e:
                print(f"❌ Error loading image: {e}")
                continue
            
            # Process frame
            response = requests.post(
                f"{RUNTIME_URL}/v1/vision/stream/frame",
                json={
                    "session_id": session_id,
                    "image": image_b64,
                },
            )
            
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code}")
                continue
            
            result = response.json()
            
            status_emoji = {
                "ok": "⚪",
                "action": "🟢",
                "review": "🟡",
            }.get(result["status"], "❓")
            
            print(f"\n{status_emoji} Status: {result['status'].upper()}")
            
            if result.get("escalated_to"):
                print(f"   ⬆️  Escalated to: {result['escalated_to']}")
            
            if result.get("added_to_replay"):
                print(f"   📚 Added to replay buffer!")
            
            if result.get("detections"):
                print(f"   Detections:")
                for det in result["detections"]:
                    print(f"      • {det['class_name']}: {det['confidence']*100:.1f}%")
            
            if result.get("image_id"):
                print(f"   📋 Queued for review: {result['image_id']}")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted")
    
    # Stop session
    response = requests.post(
        f"{RUNTIME_URL}/v1/vision/stream/stop",
        json={"session_id": session_id},
    )
    
    if response.status_code == 200:
        stats = response.json()
        print(f"\n📊 Session Statistics:")
        print(f"   Frames processed: {stats.get('frames_processed', 0)}")
        print(f"   Actions triggered: {stats.get('actions_triggered', 0)}")
        print(f"   Duration: {stats.get('duration_seconds', 0):.1f}s")
    
    # Get replay buffer stats
    response = requests.get(f"{RUNTIME_URL}/v1/vision/replay-buffer")
    if response.status_code == 200:
        buffer = response.json()
        print(f"\n📚 Replay Buffer:")
        print(f"   Size: {buffer['stats']['size']}")
        print(f"   Training eligible: {buffer['training_eligible']}")


def main():
    parser = argparse.ArgumentParser(description="Test Vision APIs with real images")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Detection
    detect_parser = subparsers.add_parser("detect", help="Test object detection")
    detect_parser.add_argument("image", help="Image path or URL")
    detect_parser.add_argument("--model", default="yolov8n", help="Model to use")
    detect_parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold")
    
    # Classification
    classify_parser = subparsers.add_parser("classify", help="Test image classification")
    classify_parser.add_argument("image", help="Image path or URL")
    classify_parser.add_argument("classes", help="Comma-separated class names")
    classify_parser.add_argument("--model", default="clip-vit-base", help="Model to use")
    
    # OCR
    ocr_parser = subparsers.add_parser("ocr", help="Test OCR text extraction")
    ocr_parser.add_argument("image", help="Image path or URL")
    ocr_parser.add_argument("--model", default="surya", help="OCR backend")
    
    # Streaming
    stream_parser = subparsers.add_parser("stream", help="Interactive streaming cascade test")
    stream_parser.add_argument("--primary", default="yolov8n", help="Primary model")
    stream_parser.add_argument("--secondary", default="yolov8m", help="Secondary model")
    
    args = parser.parse_args()
    
    if args.command == "detect":
        test_detection(args.image, args.model, args.confidence)
    elif args.command == "classify":
        test_classification(args.image, args.classes, args.model)
    elif args.command == "ocr":
        test_ocr(args.image, args.model)
    elif args.command == "stream":
        test_streaming(args.primary, args.secondary)
    else:
        parser.print_help()
        print("\n" + "=" * 50)
        print("Quick Examples:")
        print("=" * 50)
        print("""
# Detection (finds objects)
python test_vision_real.py detect ~/Desktop/photo.jpg
python test_vision_real.py detect https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg

# Classification (what is this?)
python test_vision_real.py classify ~/Desktop/photo.jpg "cat,dog,bird,person,car"

# OCR (extract text)
python test_vision_real.py ocr ~/Desktop/document.png

# Streaming cascade (interactive)
python test_vision_real.py stream
""")


if __name__ == "__main__":
    main()
