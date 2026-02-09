#!/usr/bin/env python3
"""
Step 1: Detection, Classification, and Segmentation

Demonstrates the core vision endpoints:
- Object detection with bounding boxes
- Image classification with CLIP embeddings
- Segmentation masks
- OCR text extraction

Prerequisites:
    - Universal Runtime running: nx start universal-runtime

Run:
    cd runtimes/universal
    uv run python ../../examples/vision/01_detect_and_classify.py
"""

import asyncio
import base64
import json
import os
from io import BytesIO
from pathlib import Path

import httpx

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def get_runtime_url():
    """Get Universal Runtime URL from environment or default."""
    if url := os.environ.get("RUNTIME_URL"):
        return url.rstrip("/")
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("RUNTIME_URL="):
                    return line.split("=", 1)[1].strip().strip('"\'').rstrip("/")
    return "http://localhost:14345"


RUNTIME_URL = get_runtime_url()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_test_image(width: int = 640, height: int = 480) -> bytes:
    """Generate a simple test image (gradient with shapes)."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed -- generating a minimal JPEG header")
        # Minimal valid JPEG
        return b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"

    img = Image.new("RGB", (width, height), (135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(img)

    # Ground
    draw.rectangle([(0, height // 2), (width, height)], fill=(34, 139, 34))

    # "Person" - rectangle + circle
    px, py = width // 4, height // 3
    draw.ellipse([(px - 15, py - 40), (px + 15, py)], fill=(255, 220, 185))
    draw.rectangle([(px - 20, py), (px + 20, py + 60)], fill=(0, 0, 200))
    draw.rectangle([(px - 5, py + 60), (px + 5, py + 90)], fill=(0, 0, 180))

    # "Car" - rectangles
    cx, cy = 3 * width // 4, height // 2 + 20
    draw.rectangle([(cx - 50, cy - 20), (cx + 50, cy + 20)], fill=(200, 0, 0))
    draw.rectangle([(cx - 35, cy - 40), (cx + 35, cy - 20)], fill=(180, 0, 0))
    draw.ellipse([(cx - 40, cy + 10), (cx - 20, cy + 30)], fill=(30, 30, 30))
    draw.ellipse([(cx + 20, cy + 10), (cx + 40, cy + 30)], fill=(30, 30, 30))

    # Text label (for OCR)
    try:
        draw.text((10, 10), "LLAMAFARM VISION TEST", fill=(255, 255, 255))
    except Exception:
        pass

    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode()


def print_header(title: str):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)
    print()


def print_detections(detections: list[dict]):
    if not detections:
        print("  (no detections)")
        return
    for det in detections:
        box = det.get("box") or det
        conf = det.get("confidence", 0)
        cls = det.get("class_name", det.get("class", "?"))
        coords = ""
        if "x1" in box:
            coords = f"  [{box['x1']:.0f},{box['y1']:.0f} -> {box['x2']:.0f},{box['y2']:.0f}]"
        print(f"  {cls:12s}  conf={conf:.3f}{coords}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    print_header("LlamaFarm Vision - Detection & Classification Demo")

    image_bytes = generate_test_image()
    image_b64 = b64(image_bytes)
    print(f"Generated test image: {len(image_bytes):,} bytes ({len(image_b64):,} base64 chars)")

    async with httpx.AsyncClient(base_url=RUNTIME_URL, timeout=60.0) as client:

        # ------------------------------------------------------------------
        # 1. Health check
        # ------------------------------------------------------------------
        print("\n--- Health Check ---")
        try:
            resp = await client.get("/health")
            health = resp.json()
            print(f"  Status: {health.get('status', 'unknown')}")
        except Exception as e:
            print(f"  Cannot reach runtime at {RUNTIME_URL}: {e}")
            print("  Start the runtime: nx start universal-runtime")
            return

        # ------------------------------------------------------------------
        # 2. Object Detection
        # ------------------------------------------------------------------
        print_header("1. Object Detection")
        print("POST /v1/vision/detect")
        print()

        resp = await client.post("/v1/vision/detect", json={
            "image": image_b64,
            "model": "yolov8n",
            "confidence_threshold": 0.25,
        })

        if resp.status_code == 200:
            data = resp.json()
            detections = data.get("detections", data.get("boxes", []))
            print(f"  Model: {data.get('model', 'yolov8n')}")
            print(f"  Inference: {data.get('inference_time_ms', 0):.1f}ms")
            print(f"  Detections: {len(detections)}")
            print_detections(detections)
        else:
            print(f"  Response {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 3. Classification
        # ------------------------------------------------------------------
        print_header("2. Image Classification")
        print("POST /v1/vision/classify")
        print()

        resp = await client.post("/v1/vision/classify", json={
            "image": image_b64,
            "labels": ["outdoor scene", "indoor", "vehicle", "person", "animal"],
        })

        if resp.status_code == 200:
            data = resp.json()
            print("  Classification results:")
            for item in sorted(
                data.get("scores", data.get("results", [])),
                key=lambda x: x.get("score", x.get("confidence", 0)),
                reverse=True,
            ):
                label = item.get("label", item.get("class", "?"))
                score = item.get("score", item.get("confidence", 0))
                bar = "#" * int(score * 30)
                print(f"    {label:20s} {score:.3f} {bar}")
        else:
            print(f"  Response {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 4. Segmentation
        # ------------------------------------------------------------------
        print_header("3. Segmentation")
        print("POST /v1/vision/segment")
        print()

        resp = await client.post("/v1/vision/segment", json={
            "image": image_b64,
            "model": "yolov8n-seg",
            "confidence_threshold": 0.25,
        })

        if resp.status_code == 200:
            data = resp.json()
            segments = data.get("segments", data.get("detections", []))
            print(f"  Segments found: {len(segments)}")
            for seg in segments[:5]:
                cls = seg.get("class_name", "?")
                conf = seg.get("confidence", 0)
                has_mask = bool(seg.get("mask") or seg.get("mask_polygon"))
                print(f"    {cls:12s} conf={conf:.3f}  mask={'yes' if has_mask else 'no'}")
        else:
            print(f"  Response {resp.status_code}: {resp.text[:200]}")

        # ------------------------------------------------------------------
        # 5. OCR
        # ------------------------------------------------------------------
        print_header("4. OCR (Text Extraction)")
        print("POST /v1/vision/ocr")
        print()

        resp = await client.post("/v1/vision/ocr", json={
            "image": image_b64,
        })

        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            print(f"  Extracted text: \"{text[:100]}\"")
            blocks = data.get("blocks", [])
            if blocks:
                print(f"  Text blocks: {len(blocks)}")
        else:
            print(f"  Response {resp.status_code}: {resp.text[:200]}")

    # Done
    print_header("Done!")
    print("All core vision endpoints demonstrated.")
    print("Run 02_streaming_cascade.py next for the streaming pipeline.")


if __name__ == "__main__":
    asyncio.run(main())
