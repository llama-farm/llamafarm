#!/usr/bin/env python3
"""
Demo: Zero-Shot Image Classification with CLIP

Demonstrates:
- Zero-shot classification with custom labels
- Confidence scores for all classes
- Using different CLIP model variants
"""

import argparse
import base64
import os
import sys

import httpx

API_URL = os.environ.get("LLAMAFARM_URL", "http://localhost:14345")
RUNTIME_URL = API_URL

# Sample classification scenarios
SCENARIOS = [
    {
        "name": "Animal Classification",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "classes": ["cat", "dog", "bird", "fish", "rabbit"],
        "expected": "cat",
    },
    {
        "name": "Vehicle Classification",
        "url": "https://ultralytics.com/images/bus.jpg",
        "classes": ["car", "bus", "truck", "motorcycle", "bicycle"],
        "expected": "bus",
    },
    {
        "name": "Scene Classification",
        "url": "https://ultralytics.com/images/zidane.jpg",
        "classes": [
            "a soccer game",
            "a tennis match",
            "a basketball game",
            "a golf course",
            "a swimming pool",
        ],
        "expected": "a soccer game",
    },
    {
        "name": "Fine-Grained Classification",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
        "classes": [
            "a tabby cat",
            "a siamese cat",
            "a persian cat",
            "a black cat",
            "an orange cat",
        ],
        "expected": "a tabby cat",
    },
]


def download_image(url: str) -> bytes:
    """Download image from URL."""
    print(f"  Downloading image...")
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def classify_image(image_b64: str, classes: list[str], model: str = "clip-vit-base") -> dict:
    """Classify image using CLIP."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/classify",
        json={
            "model": model,
            "image": image_b64,
            "classes": classes,
            "top_k": len(classes),
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def print_classification(result: dict, scenario: dict):
    """Pretty-print classification results."""
    print(f"\n{'='*60}")
    print(f"🏷️  Scenario: {scenario['name']}")
    print(f"{'='*60}")
    
    print(f"Model: {result.get('model', 'unknown')}")
    print(f"Inference time: {result.get('inference_time_ms', 0):.1f}ms")
    
    predicted = result.get("class", "unknown")
    confidence = result.get("confidence", 0)
    
    print(f"\n🎯 Prediction: {predicted}")
    print(f"   Confidence: {confidence:.1%}")
    
    print(f"\n📊 All Scores:")
    scores = result.get("scores", {})
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    for i, (cls, score) in enumerate(sorted_scores, 1):
        bar = "█" * int(score * 30) + "░" * (30 - int(score * 30))
        marker = " ✓" if cls == predicted else ""
        print(f"  {i}. [{bar}] {score:5.1%} {cls}{marker}")
    
    # Check expected
    expected = scenario.get("expected", "")
    if predicted == expected:
        print(f"\n✅ Correct! Matched expected: {expected}")
    else:
        print(f"\n⚠️  Expected '{expected}' but got '{predicted}'")


def run_demo(model: str = "clip-vit-base"):
    """Run the classification demo."""
    print("🏷️  LlamaFarm Vision - Zero-Shot Classification Demo")
    print("=" * 60)
    print(f"Using model: {model}")
    print(f"Running {len(SCENARIOS)} classification scenarios...\n")
    
    correct = 0
    total = len(SCENARIOS)
    
    for scenario in SCENARIOS:
        try:
            image_data = download_image(scenario["url"])
            image_b64 = base64.b64encode(image_data).decode()
            
            result = classify_image(image_b64, scenario["classes"], model)
            print_classification(result, scenario)
            
            if result.get("class") == scenario.get("expected"):
                correct += 1
                
        except Exception as e:
            print(f"❌ Error in {scenario['name']}: {e}")
    
    print("\n" + "=" * 60)
    print(f"📈 Results: {correct}/{total} scenarios classified correctly")
    print("✨ Classification demo complete!")


def main():
    parser = argparse.ArgumentParser(description="Vision Classification Demo")
    parser.add_argument("--model", "-m", default="clip-vit-base",
                       help="CLIP model variant (clip-vit-base, siglip-base, etc.)")
    parser.add_argument("--image", "-i", help="Custom image path")
    parser.add_argument("--classes", "-c", nargs="+", 
                       help="Custom classes (requires --image)")
    args = parser.parse_args()
    
    if args.image and args.classes:
        # Custom classification
        print("🏷️  Custom Classification")
        with open(args.image, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode()
        
        result = classify_image(image_b64, args.classes, args.model)
        scenario = {"name": "Custom", "classes": args.classes, "expected": ""}
        print_classification(result, scenario)
    else:
        # Run all scenarios
        try:
            run_demo(args.model)
        except httpx.ConnectError:
            print("❌ Could not connect to runtime!")
            print("   Make sure the runtime is running:")
            print("   cd runtimes/universal && uv run python server.py")
            sys.exit(1)


if __name__ == "__main__":
    main()
