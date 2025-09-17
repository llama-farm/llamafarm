#!/usr/bin/env python3
"""Debug script to test retrieval strategies"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from core.schema_handler import SchemaHandler
import tempfile
import shutil

def test_simple_retrieval():
    """Test a simple ingestion and retrieval flow"""
    
    # Create a test file
    test_dir = tempfile.mkdtemp(prefix="test_retrieval_")
    test_file = Path(test_dir) / "test.txt"
    test_file.write_text("This is a test document about artificial intelligence and machine learning.")
    
    print(f"Created test file: {test_file}")
    
    # Use default config
    config_path = Path(__file__).parent.parent / "config" / "templates" / "default.yaml"
    
    # Create schema handler
    handler = SchemaHandler(str(config_path))
    
    # List available strategies
    strategies = handler.list_strategies()
    print(f"Available strategies: {strategies}")
    
    # Test text_processing strategy
    strategy_name = "text_processing_main_database"
    print(f"\nTesting strategy: {strategy_name}")
    
    try:
        # Get the processor
        processor = handler.get_data_processor(strategy_name)
        
        # Process the test file
        print(f"Processing file: {test_file}")
        results = processor.process(str(test_file))
        print(f"Processing results: {len(results.get('documents', []))} documents")
        
        # Now test retrieval
        print("\nTesting retrieval...")
        retrieval_system = handler.get_retrieval_system(strategy_name)
        
        # Search for content
        query = "artificial intelligence"
        print(f"Searching for: '{query}'")
        search_results = retrieval_system.search(query, top_k=3)
        
        print(f"Found {len(search_results)} results")
        for i, result in enumerate(search_results, 1):
            print(f"  Result {i}: {result.content[:50]}...")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        shutil.rmtree(test_dir)
        print(f"\nCleaned up test directory")

if __name__ == "__main__":
    test_simple_retrieval()