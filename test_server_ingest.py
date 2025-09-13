#!/usr/bin/env python3
"""
Test server ingestion directly
"""
import sys
import os
sys.path.insert(0, '/Users/robthelen/llamafarm-1')

from rag.core.ingest_handler import IngestHandler

def test_ingestion():
    # Configuration
    project_dir = '/Users/robthelen/.llamafarm/projects/default/llamafarm-1'
    config_path = f'{project_dir}/llamafarm.yaml'
    
    print(f"Using config: {config_path}")
    print(f"Strategy: universal_processor")
    print(f"Database: main_database")
    
    # Initialize handler
    handler = IngestHandler(
        config_path=config_path,
        data_processing_strategy='universal_processor',
        database='main_database'
    )
    
    # Test file
    test_file = '/Users/robthelen/llamafarm-1/examples/rag_verification/sample_files/research_papers/transformer_architecture.txt'
    
    with open(test_file, 'rb') as f:
        file_data = f.read()
    
    metadata = {
        'filename': 'transformer_architecture.txt',
        'filepath': test_file
    }
    
    # Ingest
    result = handler.ingest_file(file_data, metadata)
    
    if result['status'] == 'success':
        print(f"✅ SUCCESS: Ingested {result['document_count']} chunks")
        print(f"   Parsers: {result['parsers_used']}")
        print(f"   Extractors: {result['extractors_applied']}")
        return 0
    else:
        print(f"❌ FAILED: {result.get('message', 'Unknown error')}")
        return 1

if __name__ == '__main__':
    exit(test_ingestion())