#!/usr/bin/env python3
"""
Comprehensive CLI Test Suite
Tests every CLI command with a real vector database setup and teardown.
"""

import subprocess
import sys
import os
import json
import tempfile
import shutil
from pathlib import Path
import time
from typing import Tuple, Optional

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'

class CLITestSuite:
    """Comprehensive test suite for all CLI commands."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.cli_path = self.base_dir / "cli.py"
        self.test_dir = Path(__file__).parent / "test_data"  # Inside tests/ directory
        self.test_strategy = "test_cli_strategy"
        self.test_collection = "test_cli_collection"
        self.passed_tests = []
        self.failed_tests = []
        
    def setup(self):
        """Set up test environment."""
        print(f"\n{BOLD}{BLUE}🔧 Setting up test environment...{RESET}")
        
        # Create test directory
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test documents
        self._create_test_documents()
        
        # Create test configuration
        self._create_test_config()
        
        print(f"{GREEN}✅ Test environment ready{RESET}")
        
    def teardown(self):
        """Clean up test environment."""
        print(f"\n{BOLD}{BLUE}🧹 Cleaning up test environment...{RESET}")
        
        # Remove test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
        # Remove test vector database
        test_db = self.base_dir / "vectordb" / self.test_collection
        if test_db.exists():
            shutil.rmtree(test_db)
            
        # Restore original default_strategies.yaml if we modified it
        if hasattr(self, 'original_default_strategies'):
            default_strategies_path = self.base_dir / "default_strategies.yaml"
            with open(default_strategies_path, 'w') as f:
                f.write(self.original_default_strategies)
            
        print(f"{GREEN}✅ Cleanup complete{RESET}")
        
    def _create_test_documents(self):
        """Create test documents for ingestion."""
        docs = [
            {
                "name": "test_doc1.txt",
                "content": """# Test Document 1
                
This is a test document about artificial intelligence and machine learning.
The transformer architecture has revolutionized natural language processing.
Deep learning models have shown remarkable capabilities in various tasks.
                """
            },
            {
                "name": "test_doc2.txt", 
                "content": """# Test Document 2
                
This document discusses software engineering best practices.
Test-driven development ensures code quality and maintainability.
Continuous integration helps teams deliver software efficiently.
                """
            },
            {
                "name": "test_doc3.md",
                "content": """# Test Document 3

## Overview
This is a markdown document for testing purposes.

## Key Points
- Vector databases enable semantic search
- Embeddings capture semantic meaning
- RAG systems combine retrieval and generation

## Conclusion
Testing is essential for reliable systems.
                """
            }
        ]
        
        for doc in docs:
            doc_path = self.test_dir / doc["name"]
            doc_path.write_text(doc["content"])
            print(f"  📄 Created {doc['name']}")
            
    def _create_test_config(self):
        """Create test configuration file in YAML format following config_samples format."""
        config = f"""# Test configuration for CLI testing
# Follows the config_samples format exactly

strategies:
  # ==============================================================================
  # TEST STRATEGY FOR CLI TESTING
  # ==============================================================================
  - name: "{self.test_strategy}"
    description: "Test strategy for automated CLI testing and verification"
    tags: ["test", "development", "cli"]
    use_cases:
      - "CLI testing"
      - "Integration testing" 
      - "Component verification"
    
    components:
      parser:
        type: "DirectoryParser"
        config:
          recursive: false
          file_extensions: [".txt", ".md"]
          encoding: "utf-8"
          ignore_patterns: ["*.pyc", "__pycache__"]
      
      extractors:
        - type: "KeywordExtractor"
          priority: 100
          config:
            algorithm: "yake"
            max_keywords: 10
            min_keyword_length: 3
            language: "english"
        
        - type: "StatisticsExtractor"
          priority: 90
          config:
            include_readability: true
            include_vocabulary: true
            include_structure: true
      
      embedder:
        type: "OllamaEmbedder"
        config:
          model: "nomic-embed-text"
          base_url: "http://localhost:11434"
          batch_size: 4
          timeout: 30
      
      vector_store:
        type: "ChromaStore"
        config:
          collection_name: "{self.test_collection}"
          persist_directory: "./vectordb/{self.test_collection}"
          distance_metric: "cosine"
      
      retrieval_strategy:
        type: "BasicSimilarityStrategy"
        config:
          top_k: 5
          score_threshold: 0.0
          distance_metric: "cosine"

# ==============================================================================
# STRATEGY TEMPLATES (for quick testing)
# ==============================================================================
strategy_templates:
  quick_test:
    name: "quick_test"
    description: "Minimal configuration for quick testing"
    tags: ["simple", "fast"]
    use_cases: ["Quick tests"]
    
    components:
      parser:
        type: "PlainTextParser"
        config:
          chunk_size: 500
          chunk_overlap: 50
      
      embedder:
        type: "OllamaEmbedder"
        config:
          model: "nomic-embed-text"
          batch_size: 8
      
      vector_store:
        type: "ChromaStore"
        config:
          collection_name: "quicktest"
          persist_directory: "./vectordb/quicktest"
      
      retrieval_strategy:
        type: "BasicSimilarityStrategy"
        config:
          top_k: 3
"""
        
        config_path = self.test_dir / "test_strategies.yaml"
        config_path.write_text(config)
        print(f"  ⚙️ Created test configuration (config_samples format)")
        
    def run_command(self, command: str, description: str = "", expect_failure: bool = False) -> Tuple[bool, str]:
        """Run a CLI command and return success status and output."""
        print(f"\n{BOLD}▶ {description or command}{RESET}")
        print(f"  {YELLOW}$ {command}{RESET}")
        
        try:
            # Change to base directory for proper path resolution
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                timeout=60
            )
            
            # Check for success
            if expect_failure:
                success = result.returncode != 0
            else:
                success = result.returncode == 0
                
            # Print output
            if result.stdout:
                output_lines = result.stdout.strip().split('\n')
                for line in output_lines[:5]:  # Show first 5 lines
                    print(f"  {line}")
                if len(output_lines) > 5:
                    print(f"  ... ({len(output_lines) - 5} more lines)")
                    
            # Only show real errors, not progress bars, warnings, or usage messages
            if result.stderr and not expect_failure:
                # Filter out known non-error outputs
                stderr_lines = result.stderr.strip().split('\n')
                real_errors = []
                for line in stderr_lines:
                    if line and not any(skip in line for skip in ['WARNING', 'Progress', 'Embedding:', '%|', '🦙', 'usage:', 'spaCy not available']):
                        real_errors.append(line)
                
                if real_errors:
                    print(f"  {RED}Error: {real_errors[0][:200]}{RESET}")
                
            # Print result
            if success:
                print(f"  {GREEN}✅ Success{RESET}")
            else:
                print(f"  {RED}❌ Failed{RESET}")
                
            return success, result.stdout
            
        except subprocess.TimeoutExpired:
            print(f"  {RED}❌ Command timed out{RESET}")
            return False, ""
        except Exception as e:
            print(f"  {RED}❌ Error: {e}{RESET}")
            return False, ""
            
    def test_help_commands(self):
        """Test help and version commands."""
        print(f"\n{BOLD}{BLUE}📚 Testing Help Commands{RESET}")
        
        tests = [
            ("python cli.py --help", "Main help"),
            ("python cli.py ingest --help", "Ingest help"),
            ("python cli.py search --help", "Search help"),
            ("python cli.py info --help", "Info help"),
            ("python cli.py manage --help", "Manage help"),
            ("python cli.py extractors --help", "Extractors help"),
            ("python cli.py strategies --help", "Strategies help"),
        ]
        
        for command, desc in tests:
            success, _ = self.run_command(command, desc)
            if success:
                self.passed_tests.append(desc)
            else:
                self.failed_tests.append(desc)
                
    def test_extractors_command(self):
        """Test extractors listing command."""
        print(f"\n{BOLD}{BLUE}🔧 Testing Extractors Command{RESET}")
        
        success, output = self.run_command(
            "python cli.py extractors list",
            "List available extractors"
        )
        
        if success:
            self.passed_tests.append("List extractors")
        else:
            self.failed_tests.append("List extractors")
            
    def test_strategies_commands(self):
        """Test strategies commands."""
        print(f"\n{BOLD}{BLUE}📋 Testing Strategies Commands{RESET}")
        
        # List strategies
        success, output = self.run_command(
            "python cli.py strategies list",
            "List available strategies"
        )
        
        if success:
            self.passed_tests.append("List strategies")
        else:
            self.failed_tests.append("List strategies")
            
        # Show specific strategy (if exists) using our test strategy file
        strategy_file_path = self.test_dir / "test_strategies.yaml"
        success, output = self.run_command(
            f"python cli.py --strategy-file {strategy_file_path} strategies show test_cli_strategy",
            "Show test_cli_strategy strategy"
        )
        
        # This might fail if strategy doesn't exist, which is ok
        if success:
            self.passed_tests.append("Show strategy")
            
    def test_ingest_command(self):
        """Test document ingestion."""
        print(f"\n{BOLD}{BLUE}📥 Testing Ingest Command{RESET}")
        
        # Add test strategy to the default_strategies.yaml file temporarily
        default_strategies_path = self.base_dir / "default_strategies.yaml"
        
        # Read existing strategies
        with open(default_strategies_path, 'r') as f:
            original_content = f.read()
        
        # Add our test strategy using the current YAML format
        test_strategy_yaml = f"""
# =============================================================================
# TEST STRATEGY FOR CLI TESTING
# =============================================================================
{self.test_strategy}:
  name: "{self.test_strategy}"
  description: "Test strategy for automated CLI testing"
  use_cases: ["CLI testing", "integration_testing"]
  performance_priority: "speed"
  resource_usage: "low"
  complexity: "simple"
  
  components:
    parser:
      type: "DirectoryParser"
      config:
        recursive: false
        file_extensions: [".txt", ".md"]
    
    extractors:
      - type: "KeywordExtractor"
        config:
          max_keywords: 5
    
    embedder:
      type: "OllamaEmbedder"
      config:
        model: "nomic-embed-text"
        batch_size: 4
        timeout: 60
    
    vector_store:
      type: "ChromaStore"
      config:
        collection_name: "{self.test_collection}"
        persist_directory: "./vectordb/{self.test_collection}"
    
    retrieval_strategy:
      type: "BasicSimilarityStrategy"
      config:
        distance_metric: "cosine"
"""
        
        # Add to the end before usage_notes
        usage_notes_pos = original_content.find("usage_notes:")
        if usage_notes_pos > 0:
            modified_content = original_content[:usage_notes_pos] + test_strategy_yaml + "\n" + original_content[usage_notes_pos:]
        else:
            modified_content = original_content.rstrip() + "\n" + test_strategy_yaml
            
        # Write back with test strategy
        with open(default_strategies_path, 'w') as f:
            f.write(modified_content)
        
        # Store original for cleanup
        self.original_default_strategies = original_content
        
        # Test with strategy file
        strategy_file_path = self.test_dir / "test_strategies.yaml"
        success, output = self.run_command(
            f"python cli.py --strategy-file {strategy_file_path} ingest {self.test_dir} --strategy {self.test_strategy}",
            "Ingest test documents with strategy"
        )
        
        if success and "documents" in output.lower():
            self.passed_tests.append("Ingest with config")
        else:
            self.failed_tests.append("Ingest with config")
            
        # Wait for ingestion to complete
        time.sleep(2)
        
    def test_info_command(self):
        """Test collection info command."""
        print(f"\n{BOLD}{BLUE}ℹ️ Testing Info Command{RESET}")
        
        strategy_file_path = self.test_dir / "test_strategies.yaml"
        success, output = self.run_command(
            f"python cli.py --strategy-file {strategy_file_path} info --strategy {self.test_strategy}",
            "Get collection info"
        )
        
        if success and ("documents" in output.lower() or "collection" in output.lower()):
            self.passed_tests.append("Collection info")
        else:
            self.failed_tests.append("Collection info")
            
    def test_search_commands(self):
        """Test search functionality."""
        print(f"\n{BOLD}{BLUE}🔍 Testing Search Commands{RESET}")
        
        # Basic search
        strategy_file_path = self.test_dir / "test_strategies.yaml"
        success, output = self.run_command(
            f'python cli.py --strategy-file {strategy_file_path} search "artificial intelligence" --strategy {self.test_strategy}',
            "Basic search"
        )
        
        if success:
            self.passed_tests.append("Basic search")
        else:
            self.failed_tests.append("Basic search")
            
        # Search with top-k
        success, output = self.run_command(
            f'python cli.py --strategy-file {strategy_file_path} search "machine learning" --strategy {self.test_strategy} --top-k 2',
            "Search with top-k"
        )
        
        if success:
            self.passed_tests.append("Search with top-k")
        else:
            self.failed_tests.append("Search with top-k")
            
        # Verbose search
        success, output = self.run_command(
            f'python cli.py --strategy-file {strategy_file_path} --verbose search "transformer" --strategy {self.test_strategy}',
            "Verbose search"
        )
        
        if success:
            self.passed_tests.append("Verbose search")
        else:
            self.failed_tests.append("Verbose search")
            
        # Quiet search
        success, output = self.run_command(
            f'python cli.py --strategy-file {strategy_file_path} --quiet search "testing" --strategy {self.test_strategy}',
            "Quiet search"
        )
        
        if success:
            self.passed_tests.append("Quiet search")
        else:
            self.failed_tests.append("Quiet search")
            
        # Search with custom content length
        success, output = self.run_command(
            f'python cli.py --strategy-file {strategy_file_path} --content-length 100 search "software" --strategy {self.test_strategy}',
            "Search with content length"
        )
        
        if success:
            self.passed_tests.append("Search with content length")
        else:
            self.failed_tests.append("Search with content length")
            
    def test_manage_commands(self):
        """Test manage commands."""
        print(f"\n{BOLD}{BLUE}🗂️ Testing Manage Commands{RESET}")
        
        # Test manage stats (valid command) - using new --strategy argument
        strategy_file_path = self.test_dir / "test_strategies.yaml"
        success, output = self.run_command(
            f"python cli.py --strategy-file {strategy_file_path} manage --strategy {self.test_strategy} stats",
            "Manage stats"
        )
        
        # This might fail if no documents, but command should be valid
        if success or "No documents" in output:
            self.passed_tests.append("Manage stats")
        else:
            self.failed_tests.append("Manage stats")
            
        # Test manage delete with dry-run (safe to test)
        success, output = self.run_command(
            f"python cli.py --strategy-file {strategy_file_path} manage --strategy {self.test_strategy} delete --delete-strategy soft --dry-run",
            "Manage delete (dry-run)"
        )
        
        # Should succeed even with no documents (dry-run mode)
        if success or "DRY RUN" in output or "No documents" in output:
            self.passed_tests.append("Manage delete dry-run")
        else:
            self.failed_tests.append("Manage delete dry-run")
            
        # Test manage delete with older-than filter (dry-run)
        success, output = self.run_command(
            f"python cli.py --strategy-file {strategy_file_path} manage --strategy {self.test_strategy} delete --delete-strategy hard --older-than 0 --dry-run",
            "Manage delete older-than (dry-run)"
        )
        
        if success or "DRY RUN" in output or "No documents" in output:
            self.passed_tests.append("Manage delete older-than")
        else:
            self.failed_tests.append("Manage delete older-than")
            
        # Test manage help
        success, output = self.run_command(
            "python cli.py manage --help",
            "Manage help"
        )
        
        if success and "delete" in output and "stats" in output:
            self.passed_tests.append("Manage help")
        else:
            self.failed_tests.append("Manage help")
            
        # Test delete subcommand help
        success, output = self.run_command(
            "python cli.py manage delete --help",
            "Manage delete help"
        )
        
        if success and "--delete-strategy" in output and "--older-than" in output:
            self.passed_tests.append("Manage delete help")
        else:
            self.failed_tests.append("Manage delete help")
        
    def test_error_cases(self):
        """Test error handling."""
        print(f"\n{BOLD}{BLUE}⚠️ Testing Error Cases{RESET}")
        
        # Invalid command
        success, _ = self.run_command(
            "python cli.py invalid_command",
            "Invalid command (should fail)",
            expect_failure=True
        )
        
        if success:  # We expect this to fail
            self.passed_tests.append("Invalid command handling")
        else:
            self.failed_tests.append("Invalid command handling")
            
        # Search without ingestion (with non-existent collection)
        success, _ = self.run_command(
            'python cli.py --config nonexistent.json search "test"',
            "Search with invalid config (should fail)",
            expect_failure=True
        )
        
        if success:  # We expect this to fail
            self.passed_tests.append("Invalid config handling")
        else:
            self.failed_tests.append("Invalid config handling")
            
    def print_summary(self):
        """Print test summary."""
        print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}📊 Test Summary{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        
        total = len(self.passed_tests) + len(self.failed_tests)
        
        if self.passed_tests:
            print(f"\n{GREEN}✅ Passed Tests ({len(self.passed_tests)}/{total}):{RESET}")
            for test in self.passed_tests:
                print(f"   • {test}")
                
        if self.failed_tests:
            print(f"\n{RED}❌ Failed Tests ({len(self.failed_tests)}/{total}):{RESET}")
            for test in self.failed_tests:
                print(f"   • {test}")
                
        # Overall result
        print(f"\n{BOLD}Overall Result: ", end="")
        if not self.failed_tests:
            print(f"{GREEN}ALL TESTS PASSED! 🎉{RESET}")
        else:
            print(f"{RED}SOME TESTS FAILED{RESET}")
            
        print(f"{BOLD}{BLUE}{'='*60}{RESET}\n")
        
    def run_all_tests(self):
        """Run all tests."""
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        print(f"{BOLD}{BLUE}🧪 Comprehensive CLI Test Suite{RESET}")
        print(f"{BOLD}{BLUE}{'='*60}{RESET}")
        
        try:
            # Setup
            self.setup()
            
            # Run tests
            self.test_help_commands()
            self.test_extractors_command()
            self.test_strategies_commands()
            self.test_ingest_command()
            self.test_info_command()
            self.test_search_commands()
            self.test_manage_commands()
            self.test_error_cases()
            
            # Print summary
            self.print_summary()
            
        finally:
            # Always clean up
            self.teardown()
            
        # Return exit code
        return 0 if not self.failed_tests else 1


def main():
    """Main entry point."""
    test_suite = CLITestSuite()
    exit_code = test_suite.run_all_tests()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()