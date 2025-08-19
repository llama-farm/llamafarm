#!/usr/bin/env python3
"""Quick test of image detection with a real image."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from components.factory import ImageRecognizerFactory
import requests
from PIL import Image
from io import BytesIO

# Download a sample image
print("Downloading sample image...")
url = "https://ultralytics.com/images/bus.jpg"
response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to download image: {response.status_code}")
    sys.exit(1)
img = Image.open(BytesIO(response.content))

# Save it
img_path = Path("demos/static_samples/images/sample_street.jpg")
img_path.parent.mkdir(parents=True, exist_ok=True)
img.save(img_path)
print(f"Saved sample image to {img_path}")

# Create recognizer
print("\nCreating YOLO recognizer...")
config = {
    "type": "yolo",
    "config": {
        "version": "yolov8n",
        "confidence_threshold": 0.5
    }
}

recognizer = ImageRecognizerFactory.create(config)
print(f"Recognizer created with device: {recognizer.device}")

# Run detection
print("\nRunning object detection...")
detections = recognizer.detect(img_path, confidence=0.3)

# Display results
print(f"\n🎯 Found {len(detections)} objects:")
for det in detections:
    print(f"   - {det.label}: {det.confidence:.2%} at bbox {[int(x) for x in det.bbox]}")

# Save visualization
viz_path = img_path.with_suffix(".detection_viz.jpg")
recognizer.visualize(img_path, detections, viz_path)
print(f"\n🖼️  Visualization saved to: {viz_path}")