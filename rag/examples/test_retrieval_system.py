#!/usr/bin/env python3
"""Test script for retrieval strategies system."""

import json
import sys
from pathlib import Path
from api import SearchAPI, search
from retrieval.factory import RetrievalStrategyFactory, create_retrieval_strategy_from_config


def test_basic_setup():
    """Test if the retrieval system is properly set up."""
    print("🔧 Testing Basic Setup...")
    
    # Test factory
    strategies = RetrievalStrategyFactory.list_strategies()
    print(f"✅ Found {len(strategies)} available strategies:")
    for name, class_name in list(strategies.items())[:5]:  # Show first 5
        print(f"   - {name}: {class_name}")
    
    # Test strategy creation
    try:
        basic_strategy = RetrievalStrategyFactory.create("basic")
        print(f"✅ Successfully created basic strategy: {basic_strategy.name}")
    except Exception as e:
        print(f"❌ Failed to create basic strategy: {e}")
        return False
    
    # Test config file existence
    config_files = [
        "config_examples/basic_with_retrieval_config.json",
        "config_examples/advanced_retrieval_config.json"
    ]
    
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✅ Config file exists: {config_file}")
        else:
            print(f"❌ Missing config file: {config_file}")
    
    return True


def test_sample_data_ingestion():
    """Test ingesting sample data with retrieval config."""
    print("\n📊 Testing Sample Data Ingestion...")
    
    # Check if we have sample data
    sample_file = "samples/small_sample.csv"
    if not Path(sample_file).exists():
        print(f"❌ Sample file not found: {sample_file}")
        return False
    
    print(f"✅ Sample file exists: {sample_file}")
    
    # Try to ingest data using CLI (we'll simulate this by checking if we can create the API)
    try:
        api = SearchAPI(config_path="config_examples/basic_with_retrieval_config.json")
        print("✅ Successfully initialized SearchAPI with retrieval config")
        
        # Get collection info
        info = api.get_collection_info()
        doc_count = info.get('document_count', 0)
        print(f"📈 Collection has {doc_count} documents")
        
        if doc_count == 0:
            print("⚠️  No documents in collection. You may need to ingest data first.")
            print("   Run: uv run python cli.py --config config_examples/basic_with_retrieval_config.json ingest samples/small_sample.csv")
            return False
        
        strategy_info = info.get('retrieval_strategy', {})
        print(f"🔍 Active retrieval strategy: {strategy_info.get('name', 'Unknown')}")
        print(f"   Type: {strategy_info.get('type', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize API: {e}")
        return False


def test_search_functionality():
    """Test search functionality with different strategies."""
    print("\n🔍 Testing Search Functionality...")
    
    test_queries = [
        "password reset",
        "login issues", 
        "network connectivity",
        "security vulnerability"
    ]
    
    configs_to_test = [
        ("Basic Strategy", "config_examples/basic_with_retrieval_config.json"),
        ("Advanced Strategy", "config_examples/advanced_retrieval_config.json")
    ]
    
    results_summary = {}
    
    for strategy_name, config_path in configs_to_test:
        print(f"\n  Testing {strategy_name}...")
        
        if not Path(config_path).exists():
            print(f"   ❌ Config file not found: {config_path}")
            continue
            
        try:
            api = SearchAPI(config_path=config_path)
            strategy_results = []
            
            for query in test_queries:
                try:
                    results = api.search(query, top_k=3)
                    strategy_results.append({
                        'query': query,
                        'num_results': len(results),
                        'top_score': results[0].score if results else 0,
                        'success': True
                    })
                    print(f"   ✅ Query '{query}': {len(results)} results, top score: {results[0].score:.3f}" if results else f"   ⚠️  Query '{query}': No results")
                    
                except Exception as e:
                    strategy_results.append({
                        'query': query,
                        'success': False,
                        'error': str(e)
                    })
                    print(f"   ❌ Query '{query}' failed: {e}")
            
            results_summary[strategy_name] = strategy_results
            
        except Exception as e:
            print(f"   ❌ Failed to initialize {strategy_name}: {e}")
            results_summary[strategy_name] = {'error': str(e)}
    
    return results_summary


def test_strategy_comparison():
    """Compare results from different strategies."""
    print("\n⚖️  Testing Strategy Comparison...")
    
    test_query = "password reset issues"
    
    configs = [
        ("Basic", "config_examples/basic_with_retrieval_config.json"),
        ("Advanced", "config_examples/advanced_retrieval_config.json")
    ]
    
    comparison_results = {}
    
    for strategy_name, config_path in configs:
        if not Path(config_path).exists():
            print(f"   ⚠️  Skipping {strategy_name}: config not found")
            continue
            
        try:
            api = SearchAPI(config_path=config_path)
            results = api.search(test_query, top_k=3)
            
            comparison_results[strategy_name] = {
                'num_results': len(results),
                'scores': [r.score for r in results],
                'doc_ids': [r.id for r in results],
                'top_content_preview': results[0].content[:100] + "..." if results else "No results"
            }
            
            print(f"   {strategy_name} Strategy:")
            print(f"     Results: {len(results)}")
            if results:
                print(f"     Top score: {results[0].score:.3f}")
                print(f"     Preview: {results[0].content[:60]}...")
            
        except Exception as e:
            print(f"   ❌ {strategy_name} strategy failed: {e}")
            comparison_results[strategy_name] = {'error': str(e)}
    
    return comparison_results


def test_programmatic_usage():
    """Test creating and using strategies programmatically."""
    print("\n🔧 Testing Programmatic Strategy Usage...")
    
    try:
        # Test creating hybrid strategy from config
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
                        "type": "ChromaMetadataFilterStrategy",
                        "weight": 0.3,
                        "config": {"distance_metric": "cosine"}
                    }
                ]
            }
        }
        
        strategy = create_retrieval_strategy_from_config(hybrid_config)
        print(f"✅ Created hybrid strategy: {strategy.name}")
        print(f"   Type: {type(strategy).__name__}")
        print(f"   Sub-strategies: {len(strategy.strategies) if hasattr(strategy, 'strategies') else 'N/A'}")
        
        # Test strategy compatibility
        compatible = RetrievalStrategyFactory.get_strategies_for_vector_store("ChromaStore")
        print(f"✅ Found {len(compatible)} ChromaDB-compatible strategies")
        
        return True
        
    except Exception as e:
        print(f"❌ Programmatic usage test failed: {e}")
        return False


def test_convenience_search_function():
    """Test the convenience search function."""
    print("\n🎯 Testing Convenience Search Function...")
    
    try:
        # Test simple search function
        results = search(
            query="authentication error",
            config_path="config_examples/basic_with_retrieval_config.json",
            top_k=2
        )
        
        print(f"✅ Convenience search returned {len(results)} results")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result.score:.3f}")
            print(f"      Content: {result.content[:80]}...")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Convenience search failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("🦙 RAG Retrieval System Comprehensive Test 🦙")
    print("=" * 50)
    
    test_results = {}
    
    # Run all tests
    test_results['basic_setup'] = test_basic_setup()
    test_results['data_ingestion'] = test_sample_data_ingestion()
    test_results['search_functionality'] = test_search_functionality()
    test_results['strategy_comparison'] = test_strategy_comparison()
    test_results['programmatic_usage'] = test_programmatic_usage()
    test_results['convenience_function'] = test_convenience_search_function()
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for result in test_results.values() if result is True or (isinstance(result, dict) and 'error' not in result))
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result is True or (isinstance(result, dict) and 'error' not in result) else "❌ FAIL"
        print(f"{status} {test_name.replace('_', ' ').title()}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed < total:
        print("\n💡 Troubleshooting Tips:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Ensure the embedding model is available: ollama pull nomic-embed-text")
        print("3. Ingest sample data first: uv run python cli.py --config config_examples/basic_with_retrieval_config.json ingest samples/small_sample.csv")
        print("4. Check that all config files exist in config_examples/")
    
    return passed == total


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)