#!/usr/bin/env python3
"""
Demo: Image Embeddings with CLIP

Demonstrates:
- Generating image embeddings
- Generating text embeddings
- Computing cross-modal similarity (image ↔ text)
- Image-to-image similarity search
"""

import argparse
import base64
import sys
from typing import List

import httpx
import numpy as np

RUNTIME_URL = "http://localhost:11540"

# Sample images for similarity demo
SAMPLE_IMAGES = [
    {
        "name": "Cat",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    },
    {
        "name": "Soccer",
        "url": "https://ultralytics.com/images/zidane.jpg",
    },
    {
        "name": "Bus",
        "url": "https://ultralytics.com/images/bus.jpg",
    },
]

# Text queries for cross-modal search
TEXT_QUERIES = [
    "a photo of a cat",
    "people playing soccer",
    "a bus on the street",
    "a dog running",
    "a beautiful sunset",
]


def download_image(url: str) -> bytes:
    """Download image from URL."""
    response = httpx.get(url, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    return response.content


def embed_images(images_b64: List[str], model: str = "clip-vit-base") -> dict:
    """Generate embeddings for images."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/embed",
        json={
            "model": model,
            "images": images_b64,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def embed_texts(texts: List[str], model: str = "clip-vit-base") -> dict:
    """Generate embeddings for texts."""
    response = httpx.post(
        f"{RUNTIME_URL}/v1/vision/embed",
        json={
            "model": model,
            "texts": texts,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_demo(model: str = "clip-vit-base"):
    """Run the embeddings demo."""
    print("🔢 LlamaFarm Vision - Embeddings Demo")
    print("=" * 60)
    print(f"Using model: {model}")
    
    # === Part 1: Generate Image Embeddings ===
    print("\n📸 Part 1: Image Embeddings")
    print("-" * 40)
    
    images_b64 = []
    image_names = []
    
    for sample in SAMPLE_IMAGES:
        print(f"  Downloading {sample['name']}...")
        image_data = download_image(sample["url"])
        images_b64.append(base64.b64encode(image_data).decode())
        image_names.append(sample["name"])
    
    print("\n  Generating embeddings...")
    img_result = embed_images(images_b64, model)
    img_embeddings = img_result["embeddings"]
    
    print(f"  ✅ Generated {len(img_embeddings)} image embeddings")
    print(f"  Dimensions: {img_result['dimensions']}")
    print(f"  Inference time: {img_result['inference_time_ms']:.1f}ms")
    
    # === Part 2: Generate Text Embeddings ===
    print("\n📝 Part 2: Text Embeddings")
    print("-" * 40)
    
    print(f"  Embedding {len(TEXT_QUERIES)} text queries...")
    txt_result = embed_texts(TEXT_QUERIES, model)
    txt_embeddings = txt_result["embeddings"]
    
    print(f"  ✅ Generated {len(txt_embeddings)} text embeddings")
    print(f"  Inference time: {txt_result['inference_time_ms']:.1f}ms")
    
    # === Part 3: Cross-Modal Similarity ===
    print("\n🔗 Part 3: Cross-Modal Similarity (Image ↔ Text)")
    print("-" * 40)
    
    # Build similarity matrix
    print("\n  Similarity Matrix (Image → Text):\n")
    
    # Header
    header = "           " + " ".join(f"{name:>10}" for name in image_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    
    for i, query in enumerate(TEXT_QUERIES):
        row = f"  {query[:25]:25}"
        for j, _ in enumerate(image_names):
            sim = cosine_similarity(txt_embeddings[i], img_embeddings[j])
            row += f" {sim:>10.3f}"
        print(row)
    
    # === Part 4: Image Search by Text ===
    print("\n\n🔍 Part 4: Search Images by Text Query")
    print("-" * 40)
    
    for query in TEXT_QUERIES[:3]:
        query_emb = txt_embeddings[TEXT_QUERIES.index(query)]
        
        similarities = [
            (name, cosine_similarity(query_emb, img_emb))
            for name, img_emb in zip(image_names, img_embeddings)
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n  Query: \"{query}\"")
        print(f"  Results:")
        for rank, (name, sim) in enumerate(similarities, 1):
            bar = "█" * int(sim * 20) + "░" * (20 - int(sim * 20))
            print(f"    {rank}. [{bar}] {sim:.3f} {name}")
    
    # === Part 5: Image-to-Image Similarity ===
    print("\n\n🖼️  Part 5: Image-to-Image Similarity")
    print("-" * 40)
    
    print("\n  Pairwise similarity:")
    for i, name_i in enumerate(image_names):
        for j, name_j in enumerate(image_names):
            if j > i:
                sim = cosine_similarity(img_embeddings[i], img_embeddings[j])
                print(f"    {name_i} ↔ {name_j}: {sim:.3f}")
    
    print("\n" + "=" * 60)
    print("✨ Embeddings demo complete!")
    print("\n💡 Use cases for embeddings:")
    print("   • Image search (find similar images)")
    print("   • Image-text matching (caption retrieval)")
    print("   • Deduplication (find near-duplicates)")
    print("   • RAG (retrieve relevant images for context)")


def main():
    parser = argparse.ArgumentParser(description="Vision Embeddings Demo")
    parser.add_argument("--model", "-m", default="clip-vit-base",
                       help="CLIP model variant")
    args = parser.parse_args()
    
    try:
        run_demo(args.model)
    except httpx.ConnectError:
        print("❌ Could not connect to runtime!")
        print("   Make sure the runtime is running:")
        print("   cd runtimes/universal && uv run python server.py")
        sys.exit(1)


if __name__ == "__main__":
    main()
