#!/usr/bin/env python3
"""Test RAG-lite API endpoints."""

import requests

BASE_URL = "http://127.0.0.1:8000/v1/rag-lite"


def test_stats():
    """Test stats endpoint."""
    print("\n" + "=" * 80)
    print("Testing GET /stats")
    print("=" * 80)

    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")


def test_ingest():
    """Test ingest endpoint."""
    print("\n" + "=" * 80)
    print("Testing POST /ingest")
    print("=" * 80)

    file_path = "../examples/rag_pipeline/sample_files/news_articles/ai_breakthrough.html"

    with open(file_path, "rb") as f:
        files = {"file": ("ai_breakthrough.html", f, "text/html")}
        response = requests.post(f"{BASE_URL}/ingest", files=files)

    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result.get('success')}")
    print(f"File: {result.get('file')}")
    print(f"Chunks: {result.get('chunks')}")
    print(f"Characters: {result.get('characters')}")
    if not result.get('success'):
        print(f"Error: {result.get('error')}")


def test_search():
    """Test search endpoint."""
    print("\n" + "=" * 80)
    print("Testing POST /search")
    print("=" * 80)

    payload = {
        "query": "machine learning",
        "top_k": 3
    }

    response = requests.post(f"{BASE_URL}/search", json=payload)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Success: {result.get('success')}")
    print(f"Query: {result.get('query')}")
    print(f"Count: {result.get('count')}")

    print("\nTop results:")
    for i, doc in enumerate(result.get('results', [])[:3], 1):
        print(f"\n{i}. Score: {doc['score']:.4f}")
        print(f"   Content preview: {doc['content'][:100]}...")
        print(f"   Source: {doc['metadata'].get('source', 'unknown')}")


if __name__ == "__main__":
    test_stats()
    test_ingest()
    test_stats()  # Check updated count
    test_search()
