#!/usr/bin/env python3
"""Example usage of the RAG system."""

import logging
import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.base import Pipeline
from parsers.csv_parser import CustomerSupportCSVParser
from embedders.ollama_embedder import OllamaEmbedder
from stores.chroma_store import ChromaStore


def main():
    """Example RAG pipeline usage."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Create components
    parser = CustomerSupportCSVParser()
    embedder = OllamaEmbedder(
        config={"model": "nomic-embed-text", "batch_size": 8}  # Smaller batch for demo
    )
    vector_store = ChromaStore(
        config={
            "collection_name": "example_tickets",
            "persist_directory": "./example_chroma_db",
        }
    )

    # Create pipeline
    pipeline = Pipeline("Example RAG Pipeline")
    pipeline.add_component(parser)
    pipeline.add_component(embedder)
    pipeline.add_component(vector_store)

    # Process the sample CSV
    csv_file = "filtered-english-incident-tickets.csv"
    if not Path(csv_file).exists():
        print(f"Sample CSV file not found: {csv_file}")
        return

    print("Processing customer support tickets...")

    try:
        result = pipeline.run(source=csv_file)

        print(f"\nPipeline completed!")
        print(f"  Documents processed: {len(result.documents)}")
        print(f"  Errors: {len(result.errors)}")

        if result.errors:
            print("\nFirst few errors:")
            for error in result.errors[:3]:
                print(f"  - {error}")

        # Example search
        print("\nPerforming example search...")
        search_results = vector_store.search(
            query="data breach security issue", top_k=3
        )

        print(f"Found {len(search_results)} similar tickets:")
        for i, doc in enumerate(search_results, 1):
            print(f"\n{i}. Ticket {doc.id}")
            print(f"   Content: {doc.content[:150]}...")
            if "priority" in doc.metadata:
                print(f"   Priority: {doc.metadata['priority']}")
            if "tags" in doc.metadata:
                print(f"   Tags: {doc.metadata['tags']}")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
