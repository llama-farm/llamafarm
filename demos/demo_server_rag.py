#!/usr/bin/env python
"""Demo server RAG service."""

import asyncio
import sys
from pathlib import Path

# Add server to path
server_path = Path(__file__).parent.parent / "server"
sys.path.insert(0, str(server_path))

from services.rag.rag_service import LightweightRAGService


async def main():
    """Demo server RAG service."""
    print("=" * 80)
    print("Server RAG Service Demo")
    print("=" * 80)

    # Initialize service
    print("\n1. Initializing RAG service...")
    rag = LightweightRAGService()
    print("   ✓ Service initialized")

    # Show stats
    stats = rag.get_stats()
    print(f"\n2. Initial stats:")
    print(f"   Documents: {stats['document_count']}")
    print(f"   Model: {stats['embedding_model']}")
    print(f"   Dimension: {stats['embedding_dimension']}")

    # Ingest a document
    examples_dir = Path(__file__).parent.parent / "examples" / "rag_pipeline" / "sample_files"
    pdf_file = examples_dir / "Alpaca-Care-for-Beginners.pdf"

    if pdf_file.exists():
        print(f"\n3. Ingesting document: {pdf_file.name}")
        result = await rag.ingest_file(
            str(pdf_file),
            metadata={"type": "pdf", "topic": "alpaca care"},
        )

        if result["success"]:
            print(f"   ✓ Success!")
            print(f"   Chunks: {result['chunks']}")
            print(f"   Characters: {result['characters']}")
        else:
            print(f"   ❌ Failed: {result['error']}")

    # Show updated stats
    stats = rag.get_stats()
    print(f"\n4. Updated stats:")
    print(f"   Documents: {stats['document_count']}")

    # Search
    queries = [
        "What is alpaca care?",
        "How to care for alpacas?",
        "Alpaca health",
    ]

    for query in queries:
        print(f"\n5. Searching: '{query}'")
        result = await rag.search(query, top_k=3)

        if result["success"]:
            print(f"   ✓ Found {result['count']} results")
            for i, res in enumerate(result["results"][:2], 1):
                print(f"\n   {i}. Score: {res['score']:.4f}")
                preview = res['content'][:100].replace("\n", " ")
                print(f"      Preview: {preview}...")
        else:
            print(f"   ❌ Failed: {result['error']}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
