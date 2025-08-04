#!/usr/bin/env python3
"""Example usage of retrieval strategies with the search API."""

import sys
from pathlib import Path

# Add parent directory to path so we can import modules
sys.path.insert(0, str(Path(__file__).parent.parent))
from api import SearchAPI
from retrieval.factory import RetrievalStrategyFactory, create_retrieval_strategy_from_config
from pprint import pprint


def example_basic_retrieval():
    """Example using basic retrieval strategy."""
    print("=== Basic Retrieval Strategy ===\n")
    
    # Use config with basic retrieval strategy
    api = SearchAPI(config_path="config_examples/basic_with_retrieval_config.yaml")
    
    results = api.search("password reset issues", top_k=3)
    
    print(f"Found {len(results)} results using BasicStrategy:")
    for i, result in enumerate(results, 1):
        print(f"  {i}. Score: {result.score:.3f}")
        print(f"     Content: {result.content[:100]}...")
        print()


def example_advanced_retrieval():
    """Example using advanced hybrid retrieval strategy."""
    print("\n=== Advanced Hybrid Retrieval Strategy ===\n")
    
    try:
        # Use config with hybrid retrieval strategy
        api = SearchAPI(config_path="config_examples/advanced_retrieval_config.yaml")
        
        results = api.search("network connectivity problems", top_k=5)
        
        print(f"Found {len(results)} results using HybridStrategy:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     ID: {result.id}")
            print(f"     Content: {result.content[:150]}...")
            if result.metadata:
                print(f"     Metadata: {dict(list(result.metadata.items())[:3])}")
            print()
            
        # Show collection info with strategy details
        info = api.get_collection_info()
        print("Collection Info:")
        print(f"  Strategy: {info['retrieval_strategy']['name']}")
        print(f"  Type: {info['retrieval_strategy']['type']}")
        print()
            
    except Exception as e:
        print(f"Advanced retrieval example failed: {e}")
        print("Make sure you have ingested data with the advanced config first!")


def example_metadata_filtering():
    """Example using metadata filtering strategy."""
    print("\n=== Metadata Filtering Strategy ===\n")
    
    try:
        # Use PDF config with metadata filtering
        api = SearchAPI(config_path="config_examples/pdf_with_retrieval_config.yaml")
        
        # Search with additional metadata filters
        results = api.search(
            query="methodology chapter",
            top_k=3,
            metadata_filter={"page_number": [1, 2, 3]}  # Only first 3 pages
        )
        
        print(f"Found {len(results)} results with metadata filtering:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     Page: {result.metadata.get('page_number', 'N/A')}")
            print(f"     Content: {result.content[:100]}...")
            print()
            
    except Exception as e:
        print(f"Metadata filtering example failed: {e}")
        print("Make sure you have ingested PDF data first!")


def example_multi_query_strategy():
    """Example using multi-query strategy."""
    print("\n=== Multi-Query Strategy ===\n")
    
    try:
        # Use multi-query config
        api = SearchAPI(config_path="config_examples/multi_query_retrieval_config.yaml")
        
        results = api.search("login authentication error", top_k=4)
        
        print(f"Found {len(results)} results using MultiQueryStrategy:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     Content: {result.content[:120]}...")
            print()
            
    except Exception as e:
        print(f"Multi-query example failed: {e}")
        print("Make sure you have ingested data with the multi-query config first!")


def example_programmatic_strategy_creation():
    """Example creating strategies programmatically."""
    print("\n=== Programmatic Strategy Creation ===\n")
    
    # List available strategies
    print("Available retrieval strategies:")
    strategies = RetrievalStrategyFactory.list_strategies()
    for name, class_name in strategies.items():
        print(f"  {name}: {class_name}")
    print()
    
    # Create strategy directly
    basic_strategy = RetrievalStrategyFactory.create(
        strategy_type="basic",
        config={"distance_metric": "cosine"}
    )
    print(f"Created strategy: {basic_strategy.name}")
    
    # Create hybrid strategy with custom configuration
    hybrid_config = {
        "type": "ChromaHybridStrategy",
        "config": {
            "strategies": [
                {
                    "type": "ChromaBasicStrategy",
                    "weight": 0.7,
                    "config": {"distance_metric": "cosine"}
                },
                {
                    "type": "ChromaRerankedStrategy",
                    "weight": 0.3,
                    "config": {
                        "initial_k": 15,
                        "rerank_factors": {"metadata_boost": 0.3}
                    }
                }
            ]
        }
    }
    
    hybrid_strategy = create_retrieval_strategy_from_config(hybrid_config)
    print(f"Created hybrid strategy: {hybrid_strategy.name}")
    print()


def example_strategy_comparison():
    """Compare different strategies on the same query."""
    print("\n=== Strategy Comparison ===\n")
    
    query = "security vulnerability report"
    
    configs = [
        ("Basic", "config_examples/basic_with_retrieval_config.yaml"),
        ("Advanced", "config_examples/advanced_retrieval_config.yaml"),
        ("Multi-Query", "config_examples/multi_query_retrieval_config.yaml")
    ]
    
    for strategy_name, config_path in configs:
        try:
            api = SearchAPI(config_path=config_path)
            results = api.search(query, top_k=2)
            
            print(f"{strategy_name} Strategy:")
            print(f"  Results: {len(results)}")
            if results:
                print(f"  Top score: {results[0].score:.3f}")
                print(f"  Content preview: {results[0].content[:80]}...")
            print()
            
        except Exception as e:
            print(f"{strategy_name} Strategy: Failed ({e})")
            print()


def example_collection_analysis():
    """Analyze collection info with different strategies."""
    print("\n=== Collection Analysis ===\n")
    
    configs = [
        "config_examples/basic_with_retrieval_config.yaml",
        "config_examples/advanced_retrieval_config.yaml"
    ]
    
    for config_path in configs:
        try:
            api = SearchAPI(config_path=config_path)
            info = api.get_collection_info()
            
            print(f"Config: {config_path.split('/')[-1]}")
            print(f"Collection: {info.get('collection_name', 'N/A')}")
            print(f"Document count: {info.get('document_count', 'N/A')}")
            print(f"Strategy: {info['retrieval_strategy']['name']}")
            print(f"Strategy type: {info['retrieval_strategy']['type']}")
            print()
            
        except Exception as e:
            print(f"Analysis failed for {config_path}: {e}")
            print()


if __name__ == "__main__":
    print("🦙 RAG Retrieval Strategies Examples 🦙\n")
    
    # Run examples
    try:
        example_basic_retrieval()
        example_advanced_retrieval()
        example_metadata_filtering()
        example_multi_query_strategy()
        example_programmatic_strategy_creation()
        example_strategy_comparison()
        example_collection_analysis()
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\n💡 Getting Started:")
        print("1. Make sure you have ingested some data first:")
        print("   uv run python cli.py --config config_examples/basic_with_retrieval_config.yaml ingest samples/small_sample.csv")
        print("2. Try different retrieval strategies by using different config files")
        print("3. Compare results across strategies to see the differences!")