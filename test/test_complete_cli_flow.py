#!/usr/bin/env python3
"""
Complete CLI Flow Test
Tests the entire dataset + RAG pipeline through the CLI
"""

import subprocess
import json
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and return its output."""
    print(f"\n{'='*60}")
    print(f"📋 {description}")
    print(f"💻 Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ SUCCESS")
        if result.stdout.strip():
            print(f"Output:\n{result.stdout}")
    else:
        print(f"❌ FAILED (exit code: {result.returncode})")
        if result.stderr:
            print(f"Error:\n{result.stderr}")
    
    return result.returncode == 0, result.stdout

def main():
    print("\n" + "="*80)
    print("COMPLETE CLI FLOW TEST")
    print("="*80)
    
    # Track results
    results = []
    
    # 1. Check version (simpler than health check)
    success, output = run_command(
        "./lf version",
        "Check CLI version"
    )
    results.append(("CLI version check", success))
    
    # 2. List existing datasets
    success, output = run_command(
        "./lf datasets list",
        "List existing datasets"
    )
    results.append(("List datasets", success))
    
    # 3. Create a new test dataset (remove first if it exists)
    run_command(
        "./lf datasets remove cli-test-dataset 2>/dev/null",
        "Remove existing test dataset (if any)"
    )
    success, output = run_command(
        "./lf datasets add cli-test-dataset -s universal_processor -b main_database",
        "Create new dataset via CLI"
    )
    results.append(("Create dataset", success))
    
    # 4. Add a single file
    success, output = run_command(
        "./lf datasets ingest cli-test-dataset examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt",
        "Ingest single file"
    )
    results.append(("Ingest single file", success))
    
    # 5. Add multiple files
    success, output = run_command(
        "./lf datasets ingest cli-test-dataset examples/rag_pipeline/sample_files/code/*.py",
        "Ingest multiple Python files"
    )
    results.append(("Ingest multiple files", success))
    
    # 6. List datasets again to see file count
    success, output = run_command(
        "./lf datasets list",
        "List datasets with updated file count"
    )
    results.append(("List updated datasets", success))
    
    # 7. Verify documents in ChromaDB
    print(f"\n{'='*60}")
    print("📋 Verify documents in ChromaDB")
    print(f"{'='*60}")
    
    try:
        import sys
        import os
        sys.path.insert(0, '/Users/robthelen/llamafarm-1')
        os.chdir('/Users/robthelen/llamafarm-1/server')
        
        from rag.core.ingest_handler import IngestHandler
        
        handler = IngestHandler(
            config_path='/Users/robthelen/.llamafarm/projects/default/llamafarm-1/llamafarm.yaml',
            data_processing_strategy='universal_processor',
            database='main_database'
        )
        
        collection = handler.vector_store.collection
        all_docs = collection.get()
        doc_count = len(all_docs['ids'])
        
        print(f"✅ Found {doc_count} documents in ChromaDB")
        
        # Test retrieval
        query = "transformer attention mechanism"
        embeddings = handler.embedder.embed([query])
        if embeddings:
            query_embedding = embeddings[0]
            results_search = collection.query(
                query_embeddings=[query_embedding],
                n_results=3
            )
            
            if results_search and results_search['ids'] and results_search['ids'][0]:
                print(f"✅ Search for '{query}' returned {len(results_search['ids'][0])} results")
                results.append(("ChromaDB verification", True))
            else:
                print(f"❌ Search returned no results")
                results.append(("ChromaDB verification", False))
        else:
            print(f"❌ Failed to generate embeddings")
            results.append(("ChromaDB verification", False))
            
    except Exception as e:
        print(f"❌ ChromaDB verification failed: {e}")
        results.append(("ChromaDB verification", False))
    
    # 8. Remove test dataset
    # Need to change back to the original directory first
    os.chdir('/Users/robthelen/llamafarm-1')
    success, output = run_command(
        "./lf datasets remove cli-test-dataset",
        "Remove test dataset"
    )
    results.append(("Remove dataset", success))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! The CLI integration is working perfectly!")
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Please review the output above.")
    
    return 0 if passed == total else 1

if __name__ == '__main__':
    exit(main())