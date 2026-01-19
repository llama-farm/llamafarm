#!/usr/bin/env python
"""Demo script for shared library embedders."""

import sys
from pathlib import Path

# Add common to path
common_path = Path(__file__).parent.parent / "common"
sys.path.insert(0, str(common_path))

from llamafarm_rag_common.embedders import OllamaEmbedder, UniversalEmbedder


def main():
    """Demo embedders from shared library."""
    print("=" * 80)
    print("Shared Library Embedders Demo")
    print("=" * 80)

    test_texts = [
        "Hello world",
        "Machine learning is fascinating",
        "The quick brown fox jumps over the lazy dog",
    ]

    # Test UniversalEmbedder
    print("\n1. Testing UniversalEmbedder...")
    try:
        universal = UniversalEmbedder(
            config={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "api_base": "http://127.0.0.1:11540",
            }
        )
        print(f"   Model: {universal.model}")
        print(f"   API Base: {universal.api_base}")
        print(f"   Embedding Dimension: {universal.get_embedding_dimension()}")

        embeddings = universal.embed(test_texts)
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   First embedding dim: {len(embeddings[0])}")
        print(f"   Sample values: {embeddings[0][:5]}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    # Test OllamaEmbedder
    print("\n2. Testing OllamaEmbedder...")
    try:
        ollama = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "api_base": "http://localhost:11434",
            }
        )
        print(f"   Model: {ollama.model}")
        print(f"   API Base: {ollama.api_base}")
        print(f"   Embedding Dimension: {ollama.get_embedding_dimension()}")

        embeddings = ollama.embed(["Hello from Ollama"])
        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   First embedding dim: {len(embeddings[0])}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
