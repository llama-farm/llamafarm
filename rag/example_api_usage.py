#!/usr/bin/env python3
"""Example usage of the internal search API."""

from api import SearchAPI, search
from pprint import pprint


def example_basic_search():
    """Example of basic search functionality."""
    print("=== Basic Search Example ===\n")
    
    # Method 1: Using the convenience function
    results = search("password reset", top_k=3)
    
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Score: {result.score:.3f}")
        print(f"  Content: {result.content[:200]}...")
        print(f"  Source: {result.source}")
        print()


def example_api_with_filters():
    """Example using the API class with filters."""
    print("\n=== Advanced Search with Filters ===\n")
    
    # Initialize API with specific config
    api = SearchAPI(config_path="config_examples/basic_config.json")
    
    # Search with minimum score filter
    results = api.search(
        query="security breach",
        top_k=5,
        min_score=-500.0  # Only high-confidence results
    )
    
    print(f"Found {len(results)} results with high confidence scores\n")
    
    # Search with metadata filter
    results_filtered = api.search(
        query="login issues",
        top_k=10,
        metadata_filter={"priority": "high"}  # Only high priority tickets
    )
    
    print(f"Found {len(results_filtered)} high-priority results about login issues\n")


def example_raw_documents():
    """Example getting raw Document objects."""
    print("\n=== Raw Documents Example ===\n")
    
    api = SearchAPI()
    
    # Get raw Document objects instead of SearchResult
    documents = api.search(
        query="network connectivity",
        top_k=3,
        return_raw_documents=True
    )
    
    for doc in documents:
        print(f"Document ID: {doc.id}")
        print(f"Metadata keys: {list(doc.metadata.keys())}")
        print(f"Has embeddings: {doc.embeddings is not None}")
        print()


def example_collection_info():
    """Example getting collection information."""
    print("\n=== Collection Info Example ===\n")
    
    api = SearchAPI()
    info = api.get_collection_info()
    
    print("Vector Store Information:")
    pprint(info)


def example_multiple_configs():
    """Example using different configurations."""
    print("\n=== Multiple Configurations Example ===\n")
    
    # Search CSV data
    csv_api = SearchAPI(config_path="config_examples/basic_config.json")
    csv_results = csv_api.search("customer complaint", top_k=2)
    print(f"CSV search found {len(csv_results)} results")
    
    # Search PDF data (if ingested)
    try:
        pdf_api = SearchAPI(config_path="config_examples/pdf_config.json")
        pdf_results = pdf_api.search("methodology", top_k=2)
        print(f"PDF search found {len(pdf_results)} results")
    except Exception as e:
        print(f"PDF search skipped: {e}")


def example_batch_search():
    """Example performing multiple searches."""
    print("\n=== Batch Search Example ===\n")
    
    api = SearchAPI()
    
    queries = [
        "password reset",
        "login problems",
        "security issue",
        "network error"
    ]
    
    all_results = {}
    for query in queries:
        results = api.search(query, top_k=2)
        all_results[query] = [r.to_dict() for r in results]
        print(f"Query '{query}': {len(results)} results")
    
    # Process results as needed
    print(f"\nTotal unique documents: {len(set(r['id'] for results in all_results.values() for r in results))}")


def example_search_with_context():
    """Example using search with context."""
    print("\n=== Search with Context Example ===\n")
    
    api = SearchAPI()
    
    # Get results with surrounding context
    results_with_context = api.search_with_context(
        query="data breach",
        top_k=2,
        context_size=1
    )
    
    for i, result in enumerate(results_with_context, 1):
        print(f"Result {i}:")
        print(f"  Main result score: {result['main']['score']:.3f}")
        print(f"  Context documents before: {len(result['context_before'])}")
        print(f"  Context documents after: {len(result['context_after'])}")
        print()


if __name__ == "__main__":
    print("🦙 RAG Search API Examples 🦙\n")
    
    # Run examples
    try:
        example_basic_search()
        example_api_with_filters()
        example_raw_documents()
        example_collection_info()
        example_multiple_configs()
        example_batch_search()
        example_search_with_context()
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("Make sure you have ingested some data first!")
        print("Run: uv run python cli.py ingest samples/small_sample.csv")