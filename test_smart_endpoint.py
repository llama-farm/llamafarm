#!/usr/bin/env python3
"""
Test script for the smart ingest endpoint
"""

import requests
import json
import os
from pathlib import Path

# Server configuration
SERVER_URL = "http://localhost:8000"
NAMESPACE = "default"
PROJECT = "llamafarm-1"
DATASET = "test-smart-ingest"

def test_path_detection():
    """Test the smart endpoint's path detection capabilities"""
    
    # Test paths
    test_cases = [
        {
            "paths": ["examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt"],
            "description": "Single file",
            "expected_method": "files"
        },
        {
            "paths": ["examples/rag_pipeline/sample_files/research_papers/"],
            "description": "Directory",
            "expected_method": "paths"  
        },
        {
            "paths": ["examples/rag_pipeline/sample_files/research_papers/*.txt"],
            "description": "Glob pattern",
            "expected_method": "paths"
        },
        {
            "paths": [
                "examples/rag_pipeline/sample_files/research_papers/transformer_architecture.txt",
                "examples/rag_pipeline/sample_files/code/"
            ],
            "description": "Mixed (file and directory)",
            "expected_method": "mixed"
        }
    ]
    
    print("Testing Smart Ingest Endpoint Path Detection")
    print("=" * 50)
    
    for test in test_cases:
        print(f"\n✨ Test: {test['description']}")
        print(f"   Paths: {test['paths']}")
        
        # Prepare request based on path type
        url = f"{SERVER_URL}/v1/projects/{NAMESPACE}/{PROJECT}/datasets/{DATASET}/ingest"
        
        # For this test, we'll just send paths to check detection
        data = {
            "paths": json.dumps(test["paths"]),
            "recursive": "false",
            "parallel": "true"
        }
        
        print(f"   URL: {url}")
        print(f"   Expected: {test['expected_method']} method")
        
        # Note: This is a dry-run test to check compilation
        # In a real test, we'd send the request to the server
        print(f"   ✅ Test case ready")

if __name__ == "__main__":
    test_path_detection()
    print("\n✨ All test cases prepared successfully!")
    print("\nNote: This is a dry-run test. To run actual tests:")
    print("  1. Start the LlamaFarm server")
    print("  2. Create a test dataset") 
    print("  3. Run this script again with server running")