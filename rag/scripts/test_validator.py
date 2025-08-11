#!/usr/bin/env python3
"""
Test script for the Smart Schema Validator
Demonstrates validation against actual RAG schemas
"""

import os
import sys
import yaml
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.smart_schema_validator import SmartSchemaValidator

def create_test_configs():
    """Create test configuration files"""
    
    # Valid configuration
    valid_config = """
strategies:
  - name: simple_rag
    description: "Basic RAG pipeline for general documents"
    components:
      parser:
        type: PDFParser
        config:
          combine_pages: true
          extract_metadata: true
          page_separator: "\\n\\n--- Page Break ---\\n\\n"
          min_text_length: 10
      
      embedder:
        type: OllamaEmbedder
        config:
          model: nomic-embed-text
          dimension: 768
          base_url: http://localhost:11434
          batch_size: 16
      
      vector_store:
        type: ChromaStore
        config:
          collection_name: documents
          persist_directory: ./data/chroma_db
          distance_function: cosine
      
      retrieval_strategy:
        type: BasicSimilarityStrategy
        config:
          top_k: 10
          distance_metric: cosine
      
      extractors:
        - type: KeywordExtractor
          config:
            algorithm: rake
            max_keywords: 10
            min_keyword_length: 3
        
        - type: EntityExtractor
          config:
            model: en_core_web_sm
            entity_types: [PERSON, ORG, GPE]
"""
    
    # Invalid configuration with various errors
    invalid_config = """
strategies:
  - name: "Invalid Strategy"  # Space in name
    # Missing description
    components:
      parser:
        type: PDFParserr  # Typo
        config:
          combine_page: true  # Wrong field name
          extract_metadata: "yes"  # Wrong type
      
      embedder:
        type: OllamaEmbeder  # Typo
        config:
          model: invalid-model  # Invalid model
          dimension: 10000  # Out of range
      
      vector_store:
        type: ChromaDB  # Should be ChromaStore
        config:
          collection: docs  # Wrong field name
      
      # Missing retrieval_strategy
      
      extractors:
        - type: KeywordExtract  # Wrong name
          config:
            algorythm: rake  # Typo
            max_keywords: 1000  # Out of range
"""
    
    # Configuration with warnings
    warning_config = """
strategies:
  - name: warning_example
    description: "Configuration with warnings"
    components:
      parser:
        type: PlainTextParser
        config:
          chunk_size: 1000
          chunk_overlap: 200
          unknown_field: true  # Unknown field
      
      embedder:
        type: OllamaEmbedder
        config:
          model: nomic-embed-text
          dimension: 768
          extra_param: 123  # Unknown parameter
      
      vector_store:
        type: ChromaStore
        config:
          collection_name: docs123
      
      retrieval_strategy:
        type: BasicSimilarityStrategy
        config:
          top_k: 5
"""
    
    return {
        "valid": valid_config,
        "invalid": invalid_config,
        "warning": warning_config
    }

def main():
    """Run validation tests"""
    
    print("=" * 60)
    print("Smart Schema Validator Test Suite")
    print("=" * 60)
    print()
    
    # Create validator
    validator = SmartSchemaValidator(verbose=False)
    
    # Get test configurations
    configs = create_test_configs()
    
    # Create temporary files and validate
    for config_name, config_content in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Testing: {config_name.upper()} configuration")
        print("=" * 60)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_content)
            temp_file = f.name
        
        try:
            # Generate and print report
            report = validator.generate_report(temp_file)
            print(report)
            
            # Get validation result
            is_valid, issues = validator.validate_file(temp_file)
            
            # Summary
            error_count = len([i for i in issues if i.level == 'ERROR'])
            warning_count = len([i for i in issues if i.level == 'WARNING'])
            
            print("\n" + "=" * 60)
            print(f"Configuration: {config_name}")
            print(f"Valid: {'✅ YES' if is_valid else '❌ NO'}")
            print(f"Errors: {error_count}, Warnings: {warning_count}")
            print("=" * 60)
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file)
    
    # Test improvement suggestions
    print("\n" + "=" * 60)
    print("Testing Improvement Suggestions")
    print("=" * 60)
    
    test_config = yaml.safe_load(configs["valid"])
    suggestions = validator.suggest_improvements(test_config)
    
    if suggestions:
        print("\nSuggestions for valid configuration:")
        for suggestion in suggestions:
            print(suggestion)
    else:
        print("\nNo improvement suggestions for the valid configuration.")
    
    print("\n" + "=" * 60)
    print("Validation Test Suite Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()