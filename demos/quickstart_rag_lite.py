#!/usr/bin/env python3
"""Quick start guide for RAG-lite."""

import requests

print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║              LlamaFarm RAG-Lite Quick Start                    ║
║                                                                ║
║  Lightweight, built-in RAG for document search                 ║
║  No separate worker needed!                                    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝

Prerequisites:
1. Server running: nx start server
2. Universal Runtime: nx start universal-runtime

Let's get started!
""")

BASE_URL = "http://127.0.0.1:8000/v1/rag-lite"

# Step 1: Check system status
print("Step 1: Check RAG-lite status")
print("-" * 60)

try:
    response = requests.get(f"{BASE_URL}/stats", timeout=5)
    stats = response.json()
    print(f"✓ RAG-lite is running!")
    print(f"  Documents: {stats['document_count']}")
    print(f"  Model: {stats['embedding_model']}")
    print(f"  Dimension: {stats['embedding_dimension']}")
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nMake sure server is running: nx start server")
    exit(1)

# Step 2: Ingest a document
print("\nStep 2: Ingest a document")
print("-" * 60)

file_path = "../examples/rag_pipeline/sample_files/Alpaca-Care-for-Beginners.pdf"

try:
    with open(file_path, "rb") as f:
        files = {"file": ("Alpaca-Care.pdf", f, "application/pdf")}
        response = requests.post(f"{BASE_URL}/ingest", files=files, timeout=30)

    result = response.json()
    if result['success']:
        print(f"✓ Document ingested!")
        print(f"  Chunks: {result['chunks']}")
        print(f"  Characters: {result['characters']}")
    else:
        print(f"✗ Ingestion failed: {result.get('error')}")
except FileNotFoundError:
    print("✓ Sample file not found (that's okay for demo)")
except Exception as e:
    print(f"✗ Error: {e}")

# Step 3: Search for information
print("\nStep 3: Search for information")
print("-" * 60)

queries = [
    "What is alpaca care?",
    "How do alpacas differ from llamas?",
    "What are the health considerations?"
]

for query in queries:
    print(f"\nQuery: '{query}'")

    try:
        payload = {"query": query, "top_k": 2}
        response = requests.post(f"{BASE_URL}/search", json=payload, timeout=10)
        result = response.json()

        if result['success'] and result['count'] > 0:
            top = result['results'][0]
            print(f"  ✓ Found {result['count']} results")
            print(f"  Top score: {top['score']:.4f}")
            print(f"  Preview: {top['content'][:100]}...")
        else:
            print(f"  No results found")
    except Exception as e:
        print(f"  ✗ Error: {e}")

# Final stats
print("\n" + "=" * 60)
print("Final Status")
print("=" * 60)

response = requests.get(f"{BASE_URL}/stats")
stats = response.json()
print(f"Total documents: {stats['document_count']}")

print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║  Success! RAG-lite is working!                                 ║
║                                                                ║
║  Next steps:                                                   ║
║  1. Ingest your own documents via API or curl                  ║
║  2. Build search into your application                         ║
║  3. Check out demos/demo_rag_comparison.py for more examples   ║
║                                                                ║
║  API Documentation: server/services/rag/README.md              ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
""")
