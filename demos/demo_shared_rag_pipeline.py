#!/usr/bin/env python
"""Demo full RAG pipeline using shared library."""

import sys
import tempfile
from pathlib import Path

# Add common to path
common_path = Path(__file__).parent.parent / "common"
sys.path.insert(0, str(common_path))

from llamafarm_rag_common import (
    BasicSimilarityStrategy,
    ChromaStore,
    Document,
    UniversalEmbedder,
)


def main():
    """Demo complete RAG pipeline with shared library."""
    print("=" * 80)
    print("Shared Library RAG Pipeline Demo")
    print("=" * 80)

    # Create temporary directory for ChromaDB
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n1. Setting up ChromaDB in: {tmpdir}")

        # Initialize ChromaStore
        store = ChromaStore(
            config={
                "collection_name": "demo_docs",
                "embedding_dimension": 384,
                "distance_metric": "cosine",
            },
            project_dir=Path(tmpdir),
        )
        print(f"   ✓ ChromaStore initialized")
        print(f"   Collection: {store.collection_name}")
        print(f"   Initial count: {store.get_count()}")

        # Initialize embedder
        print("\n2. Setting up UniversalEmbedder...")
        embedder = UniversalEmbedder(
            config={
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "api_base": "http://127.0.0.1:11540",
            }
        )
        print(f"   ✓ Embedder ready (dim={embedder.get_embedding_dimension()})")

        # Create sample documents
        print("\n3. Creating sample documents...")
        texts = [
            "Python is a high-level programming language.",
            "Machine learning is a subset of artificial intelligence.",
            "ChromaDB is a vector database for AI applications.",
            "LlamaFarm is a platform for running local AI models.",
            "The quick brown fox jumps over the lazy dog.",
        ]

        documents = []
        for i, text in enumerate(texts):
            doc = Document(
                id=f"doc_{i}",
                content=text,
                metadata={"source": "demo", "index": i},
            )
            documents.append(doc)

        print(f"   ✓ Created {len(documents)} documents")

        # Generate embeddings
        print("\n4. Generating embeddings...")
        embeddings = embedder.embed([doc.content for doc in documents])
        for doc, embedding in zip(documents, embeddings):
            doc.embeddings = embedding

        print(f"   ✓ Generated {len(embeddings)} embeddings")
        print(f"   Embedding dimension: {len(embeddings[0])}")

        # Add to vector store
        print("\n5. Adding documents to ChromaDB...")
        store.add_documents(documents)
        print(f"   ✓ Added documents")
        print(f"   Total count: {store.get_count()}")

        # Test search
        print("\n6. Testing search...")
        query = "What is machine learning?"
        print(f"   Query: '{query}'")

        # Generate query embedding
        query_embedding = embedder.embed([query])[0]
        print(f"   ✓ Query embedded")

        # Search using retrieval strategy
        retriever = BasicSimilarityStrategy(
            config={
                "similarity_threshold": 0.0,
                "max_results": 3,
            }
        )

        results = retriever.retrieve(
            query_embedding=query_embedding,
            vector_store=store,
            top_k=3,
        )

        print(f"\n   Results ({len(results.documents)} documents):")
        for i, (doc, score) in enumerate(
            zip(results.documents, results.scores), 1
        ):
            print(f"\n   {i}. Score: {score:.4f}")
            print(f"      Content: {doc.content}")
            print(f"      Metadata: {doc.metadata}")

        # Test deletion
        print("\n7. Testing deletion...")
        store.delete(["doc_4"])
        print(f"   ✓ Deleted 1 document")
        print(f"   Remaining count: {store.get_count()}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
