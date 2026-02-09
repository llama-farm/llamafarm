#!/usr/bin/env python3
"""
Demo: Full Vision Pipeline

Demonstrates the complete end-to-end workflow:
1. Detection: Find objects in images
2. Classification: Classify detected regions
3. Embedding: Generate vector representations
4. Storage: Persist images and detections
5. Similarity: Find similar images
6. Review Queue: Simulate human review

This shows how the components work together.
"""

import argparse
import base64
import json
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import httpx
import numpy as np

API_URL = os.environ.get("LLAMAFARM_URL", "http://localhost:14345")
RUNTIME_URL = API_URL

# Sample images
IMAGES = [
    {
        "name": "Soccer Match",
        "url": "https://ultralytics.com/images/zidane.jpg",
    },
    {
        "name": "Street Scene",
        "url": "https://ultralytics.com/images/bus.jpg",
    },
    {
        "name": "Cat Portrait",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    },
]


def download_image(url: str) -> bytes:
    """Download image from URL."""
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def detect(image_b64: str, model: str = "yolov8n") -> dict:
    """Run object detection."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/detect",
        json={"model": model, "image": image_b64, "confidence_threshold": 0.5},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def classify(image_b64: str, classes: list, model: str = "clip-vit-base") -> dict:
    """Run classification."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/classify",
        json={"model": model, "image": image_b64, "classes": classes},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def embed(images_b64: list, model: str = "clip-vit-base") -> dict:
    """Generate embeddings."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/embed",
        json={"model": model, "images": images_b64},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def cosine_similarity(a, b):
    """Compute cosine similarity."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SimpleImageStore:
    """Simple in-memory image store for demo."""
    
    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()
        self.images = {}
        self.embeddings = {}
    
    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS images (
                id TEXT PRIMARY KEY,
                name TEXT,
                source TEXT,
                created_at REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT,
                class_name TEXT,
                confidence REAL,
                box TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT,
                class_name TEXT,
                annotator TEXT,
                FOREIGN KEY (image_id) REFERENCES images(id)
            )
        """)
        self.conn.commit()
    
    def store_image(self, image_id: str, name: str, image_data: bytes, embedding: list):
        self.conn.execute(
            "INSERT INTO images (id, name, source, created_at) VALUES (?, ?, ?, ?)",
            (image_id, name, "demo", time.time())
        )
        self.conn.commit()
        self.images[image_id] = image_data
        self.embeddings[image_id] = embedding
        return image_id
    
    def store_detection(self, image_id: str, class_name: str, confidence: float, box: dict):
        self.conn.execute(
            "INSERT INTO detections (image_id, class_name, confidence, box) VALUES (?, ?, ?, ?)",
            (image_id, class_name, confidence, json.dumps(box))
        )
        self.conn.commit()
    
    def add_label(self, image_id: str, class_name: str, annotator: str = "demo"):
        self.conn.execute(
            "INSERT INTO labels (image_id, class_name, annotator) VALUES (?, ?, ?)",
            (image_id, class_name, annotator)
        )
        self.conn.commit()
    
    def get_stats(self) -> dict:
        images = self.conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        detections = self.conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        labels = self.conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        return {"images": images, "detections": detections, "labels": labels}
    
    def find_similar(self, query_id: str, k: int = 3) -> list:
        """Find most similar images to query."""
        query_emb = self.embeddings.get(query_id)
        if not query_emb:
            return []
        
        similarities = []
        for img_id, emb in self.embeddings.items():
            if img_id != query_id:
                sim = cosine_similarity(query_emb, emb)
                name = self.conn.execute(
                    "SELECT name FROM images WHERE id = ?", (img_id,)
                ).fetchone()[0]
                similarities.append((img_id, name, sim))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:k]


def run_pipeline():
    """Run the full pipeline demo."""
    print("🚀 LlamaFarm Vision - Full Pipeline Demo")
    print("=" * 70)
    
    store = SimpleImageStore()
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 1: Load and Store Images
    # ═══════════════════════════════════════════════════════════════
    print("\n📥 Step 1: Loading Images")
    print("-" * 50)
    
    images_data = {}
    for i, img_info in enumerate(IMAGES):
        print(f"  [{i+1}/{len(IMAGES)}] Downloading: {img_info['name']}")
        data = download_image(img_info["url"])
        images_data[img_info["name"]] = data
    
    print(f"  ✅ Loaded {len(images_data)} images")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: Generate Embeddings for All Images
    # ═══════════════════════════════════════════════════════════════
    print("\n🔢 Step 2: Generating Embeddings")
    print("-" * 50)
    
    images_b64 = [base64.b64encode(d).decode() for d in images_data.values()]
    embed_result = embed(images_b64)
    embeddings = embed_result["embeddings"]
    
    print(f"  ✅ Generated {len(embeddings)} embeddings")
    print(f"  Dimensions: {embed_result['dimensions']}")
    
    # Store images with embeddings
    for (name, data), emb in zip(images_data.items(), embeddings):
        img_id = f"img_{name.lower().replace(' ', '_')}"
        store.store_image(img_id, name, data, emb)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: Run Detection on Each Image
    # ═══════════════════════════════════════════════════════════════
    print("\n🔍 Step 3: Object Detection")
    print("-" * 50)
    
    total_detections = 0
    for (name, data), b64 in zip(images_data.items(), images_b64):
        result = detect(b64)
        detections = result.get("detections", [])
        
        print(f"\n  📸 {name}:")
        if detections:
            for det in detections:
                cls = det["class"]
                conf = det["confidence"]
                print(f"      • {cls}: {conf:.1%}")
                
                # Store detection
                img_id = f"img_{name.lower().replace(' ', '_')}"
                store.store_detection(img_id, cls, conf, det["box"])
                total_detections += 1
        else:
            print("      (no objects detected)")
    
    print(f"\n  ✅ Total detections: {total_detections}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: Classify Images
    # ═══════════════════════════════════════════════════════════════
    print("\n🏷️  Step 4: Scene Classification")
    print("-" * 50)
    
    scene_classes = [
        "sports event",
        "street scene",
        "animal portrait",
        "indoor scene",
        "landscape",
    ]
    
    for (name, _), b64 in zip(images_data.items(), images_b64):
        result = classify(b64, scene_classes)
        predicted = result["class"]
        confidence = result["confidence"]
        
        print(f"  📸 {name}: {predicted} ({confidence:.1%})")
        
        # Add label
        img_id = f"img_{name.lower().replace(' ', '_')}"
        store.add_label(img_id, predicted, "clip_classifier")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 5: Similarity Search
    # ═══════════════════════════════════════════════════════════════
    print("\n🔗 Step 5: Similarity Search")
    print("-" * 50)
    
    # Find images similar to each query
    for img_id in ["img_soccer_match", "img_cat_portrait"]:
        name = img_id.replace("img_", "").replace("_", " ").title()
        similar = store.find_similar(img_id, k=2)
        
        print(f"\n  Similar to '{name}':")
        for _, sim_name, score in similar:
            print(f"      → {sim_name}: {score:.3f}")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 6: Simulated Review Queue
    # ═══════════════════════════════════════════════════════════════
    print("\n👤 Step 6: Human Review Simulation")
    print("-" * 50)
    
    # Simulate human adding labels
    human_labels = [
        ("img_soccer_match", "Zinedine Zidane"),
        ("img_cat_portrait", "tabby cat"),
        ("img_street_scene", "London bus"),
    ]
    
    for img_id, label in human_labels:
        store.add_label(img_id, label, "human_reviewer")
        print(f"  ✏️  Added label '{label}' to {img_id}")
    
    # ═══════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("📊 Pipeline Summary")
    print("=" * 70)
    
    stats = store.get_stats()
    print(f"  Images stored: {stats['images']}")
    print(f"  Detections recorded: {stats['detections']}")
    print(f"  Labels added: {stats['labels']}")
    
    print("\n🔄 Pipeline Flow:")
    print("""
    ┌─────────────┐    ┌───────────────┐    ┌─────────────────┐
    │   Load      │ → │   Detect      │ → │   Classify      │
    │   Images    │    │   Objects     │    │   Scenes        │
    └─────────────┘    └───────────────┘    └─────────────────┘
           │                  │                     │
           ▼                  ▼                     ▼
    ┌─────────────┐    ┌───────────────┐    ┌─────────────────┐
    │   Generate  │ → │   Store       │ → │   Similarity    │
    │   Embeddings│    │   Results     │    │   Search        │
    └─────────────┘    └───────────────┘    └─────────────────┘
           │                                        │
           └────────────────────────────────────────┘
                              ▼
                    ┌─────────────────┐
                    │   Human Review  │
                    │   & Labeling    │
                    └─────────────────┘
    """)
    
    print("✨ Full pipeline demo complete!")


def main():
    parser = argparse.ArgumentParser(description="Full Vision Pipeline Demo")
    args = parser.parse_args()
    
    try:
        run_pipeline()
    except httpx.ConnectError:
        print("❌ Could not connect to runtime!")
        print("   Make sure the runtime is running:")
        print("   cd runtimes/universal && uv run python server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
