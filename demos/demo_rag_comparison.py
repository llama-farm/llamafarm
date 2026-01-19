#!/usr/bin/env python3
"""Demo comparing RAG-lite (built-in) vs Advanced RAG."""

import requests

print("=" * 80)
print("RAG System Comparison Demo")
print("=" * 80)

# Test RAG-lite (built-in, lightweight)
print("\n" + "=" * 80)
print("RAG-LITE (Built-in Lightweight RAG)")
print("=" * 80)

rag_lite_url = "http://127.0.0.1:8000/v1/rag-lite"

print("\n1. Stats:")
response = requests.get(f"{rag_lite_url}/stats")
stats = response.json()
print(f"   Documents: {stats['document_count']}")
print(f"   Model: {stats['embedding_model']}")
print(f"   Dimension: {stats['embedding_dimension']}")
print(f"   Collection: {stats['collection']}")

print("\n2. Search Test:")
payload = {"query": "artificial intelligence breakthrough", "top_k": 2}
response = requests.post(f"{rag_lite_url}/search", json=payload)
result = response.json()
print(f"   Query: '{result['query']}'")
print(f"   Results: {result['count']}")
if result['count'] > 0:
    print(f"   Top score: {result['results'][0]['score']:.4f}")
    print(f"   Preview: {result['results'][0]['content'][:100]}...")

# Test Advanced RAG (if available)
print("\n" + "=" * 80)
print("ADVANCED RAG (Celery-based, full features)")
print("=" * 80)

health_url = "http://127.0.0.1:8000/health"
response = requests.get(health_url)
health = response.json()

rag_component = [c for c in health['components'] if c['name'] == 'rag-service']
if rag_component:
    rag = rag_component[0]
    print(f"\n1. Status: {rag['status']}")
    print(f"   Message: {rag['message']}")
    if 'details' in rag:
        print(f"   Worker ID: {rag['details'].get('worker_id', 'unknown')}")
        print(f"   Memory: {rag['details'].get('metrics', {}).get('memory_mb', 0):.1f} MB")
else:
    print("\n   Not configured or not running")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
RAG-LITE (Built-in):
  ✓ Always available (no separate worker needed)
  ✓ Lightweight (MarkItDown, ChromaDB embedded)
  ✓ Perfect for 90% of users
  ✓ HTTP-based embedders (Universal Runtime/Ollama)
  ✓ Simple chunking and retrieval

ADVANCED RAG (Optional):
  ✓ Power users (10%)
  ✓ Advanced parsers (Docling, etc.)
  ✓ Multiple vector stores
  ✓ Celery distributed processing
  ✓ Advanced retrieval strategies
  ✓ Requires: nx start rag
""")

print("=" * 80)
print("Demo Complete!")
print("=" * 80)
