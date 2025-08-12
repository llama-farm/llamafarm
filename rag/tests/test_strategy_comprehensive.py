#!/usr/bin/env python3
"""
Comprehensive Strategy System Tester

Tests the RAG strategy system with various combinations of:
- Different file types (CSV, PDF, TXT, MD, HTML, DOCX, XLSX)
- Multiple extractors (Entity, Keyword, Summary, Pattern, Link, etc.)
- Various parsers (CSVParser, PDFParser, PlainTextParser, etc.)
- Different retrieval strategies (BasicSimilarity, MetadataFiltered, MultiQuery, etc.)

Uses OllamaEmbedder and ChromaStore consistently since they're fully configured.
"""

import subprocess
import sys
import os
import tempfile
import shutil
from pathlib import Path
import time
import yaml
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

class StrategyTester:
    """Comprehensive strategy system tester."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.cli_path = self.base_dir / "cli.py"
        self.test_dir = Path(tempfile.mkdtemp(prefix="strategy_test_"))
        self.strategy_file = self.test_dir / "test_strategies.yaml"
        self.results = []
        self.passed = 0
        self.failed = 0
        
    def setup(self):
        """Set up test environment."""
        print(f"\n{BOLD}{BLUE}🔧 Setting up strategy test environment...{RESET}")
        
        # Create test documents of different types
        self._create_test_documents()
        
        # Create comprehensive test strategies
        self._create_test_strategies()
        
        print(f"{GREEN}✅ Test environment ready at {self.test_dir}{RESET}")
        
    def teardown(self):
        """Clean up test environment."""
        print(f"\n{BOLD}{BLUE}🧹 Cleaning up test environment...{RESET}")
        
        # Remove test directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
            
        # Remove test vector databases
        for strategy in self.get_test_strategies():
            db_path = self.base_dir / "vectordb" / f"test_{strategy}"
            if db_path.exists():
                shutil.rmtree(db_path)
                
        print(f"{GREEN}✅ Cleanup complete{RESET}")
        
    def _create_test_documents(self):
        """Create test documents of various types."""
        print(f"  📄 Creating test documents...")
        
        # CSV document
        csv_content = """id,title,content,priority
1,Test Issue,"This is a test support ticket about login problems",high
2,Feature Request,"User wants dark mode feature in the application",medium
3,Bug Report,"Application crashes when uploading large files",critical
"""
        (self.test_dir / "test_data.csv").write_text(csv_content)
        
        # Plain text document
        txt_content = """# Research on Artificial Intelligence

Artificial intelligence (AI) is transforming various industries.
Machine learning algorithms enable computers to learn from data.
Deep learning neural networks have revolutionized computer vision.

Key researchers include Geoffrey Hinton, Yann LeCun, and Yoshua Bengio.
Major companies investing in AI: Google, Microsoft, Amazon, Meta.

Published: 2024-01-15
Author: Dr. Jane Smith
Organization: AI Research Institute
"""
        (self.test_dir / "research.txt").write_text(txt_content)
        
        # Markdown document
        md_content = """# Technical Documentation

## Overview
This document describes the API endpoints for our system.

## Authentication
All requests require an API key: `Authorization: Bearer YOUR_API_KEY`

## Endpoints

### GET /api/users
Returns list of users. Example:
```json
{
  "users": [
    {"id": 1, "name": "Alice", "email": "alice@example.com"},
    {"id": 2, "name": "Bob", "email": "bob@example.com"}
  ]
}
```

### POST /api/data
Submit new data entry. See [documentation](https://docs.example.com/api)

Contact: support@example.com | Phone: +1-555-0123
"""
        (self.test_dir / "api_docs.md").write_text(md_content)
        
        # HTML document
        html_content = """<!DOCTYPE html>
<html>
<head><title>Breaking News: AI Breakthrough</title></head>
<body>
<h1>Revolutionary AI Model Achieves Human-Level Performance</h1>
<p>Published: <time>2024-11-15</time> by <span class="author">John Doe</span></p>
<p>A new artificial intelligence model developed by <strong>TechCorp</strong> has achieved 
unprecedented performance on language understanding tasks.</p>
<p>The model, called <em>SuperAI</em>, scored 95% on benchmark tests.</p>
<p>Investment: $10 million from venture capital firms.</p>
<a href="https://techcorp.com/superai">Learn more</a>
<a href="mailto:press@techcorp.com">Contact press team</a>
</body>
</html>
"""
        (self.test_dir / "news.html").write_text(html_content)
        
        # Simple DOCX-like text (we'll treat as text for testing)
        docx_content = """Contract Agreement

This agreement is entered into on January 1, 2024, between:
- Party A: Acme Corporation (ID: 12-3456789)
- Party B: Tech Solutions Inc. (ID: 98-7654321)

Terms:
1. Services to be provided by Party B
2. Payment of $50,000 upon completion
3. Deadline: December 31, 2024

Contact: legal@acme.com | contracts@techsolutions.com
Case Reference: 2024/US/123456
"""
        (self.test_dir / "contract.txt").write_text(docx_content)  # Simulate DOCX as text
        
        # Simple Excel-like CSV (for Excel parser testing)
        xlsx_content = """Quarter,Revenue,Expenses,Profit,Growth
Q1 2024,1000000,750000,250000,15%
Q2 2024,1200000,800000,400000,20%
Q3 2024,1500000,900000,600000,25%
Q4 2024,1800000,950000,850000,20%
"""
        (self.test_dir / "financials.csv").write_text(xlsx_content)  # Simulate XLSX as CSV
        
        print(f"    ✓ Created 6 test documents")
        
    def _create_test_strategies(self):
        """Create comprehensive test strategies."""
        print(f"  ⚙️ Creating test strategy configurations...")
        
        strategies = {
            "strategies": [
                # Strategy 1: CSV with full extraction
                {
                    "name": "test_csv_full",
                    "description": "Test CSV parsing with all extractors",
                    "tags": ["test", "csv", "full"],
                    "use_cases": ["CSV processing test"],
                    "components": {
                        "parser": {
                            "type": "CSVParser",
                            "config": {
                                "delimiter": ",",
                                "has_header": True
                            }
                        },
                        "extractors": [
                            {"type": "EntityExtractor", "priority": 100, "config": {}},
                            {"type": "KeywordExtractor", "priority": 90, "config": {"max_keywords": 5}},
                            {"type": "PatternExtractor", "priority": 80, "config": {}},
                            {"type": "SummaryExtractor", "priority": 70, "config": {"summary_sentences": 2}}
                        ],
                        "embedder": {
                            "type": "OllamaEmbedder",
                            "config": {
                                "model": "nomic-embed-text",
                                "batch_size": 4
                            }
                        },
                        "vector_store": {
                            "type": "ChromaStore",
                            "config": {
                                "collection_name": "test_csv_full",
                                "persist_directory": "./vectordb/test_csv_full"
                            }
                        },
                        "retrieval_strategy": {
                            "type": "BasicSimilarityStrategy",
                            "config": {"top_k": 3}
                        }
                    }
                },
                
                # Strategy 2: Text with entity and keyword extraction
                {
                    "name": "test_text_entities",
                    "description": "Test text parsing with entity extraction",
                    "tags": ["test", "text", "entities"],
                    "use_cases": ["Text entity extraction test"],
                    "components": {
                        "parser": {
                            "type": "PlainTextParser",
                            "config": {
                                "chunk_size": 500,
                                "chunk_overlap": 50
                            }
                        },
                        "extractors": [
                            {"type": "EntityExtractor", "priority": 100, "config": {
                                "entity_types": ["PERSON", "ORG", "DATE"]
                            }},
                            {"type": "DateTimeExtractor", "priority": 90, "config": {}},
                            {"type": "StatisticsExtractor", "priority": 80, "config": {}}
                        ],
                        "embedder": {
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"}
                        },
                        "vector_store": {
                            "type": "ChromaStore",
                            "config": {
                                "collection_name": "test_text_entities",
                                "persist_directory": "./vectordb/test_text_entities"
                            }
                        },
                        "retrieval_strategy": {
                            "type": "MetadataFilteredStrategy",
                            "config": {"top_k": 5}
                        }
                    }
                },
                
                # Strategy 3: Markdown with heading and link extraction
                {
                    "name": "test_markdown_structure",
                    "description": "Test markdown parsing with structure extraction",
                    "tags": ["test", "markdown", "structure"],
                    "use_cases": ["Markdown structure test"],
                    "components": {
                        "parser": {
                            "type": "MarkdownParser",
                            "config": {
                                "extract_metadata": True,
                                "preserve_formatting": True
                            }
                        },
                        "extractors": [
                            {"type": "HeadingExtractor", "priority": 100, "config": {"max_level": 3}},
                            {"type": "LinkExtractor", "priority": 90, "config": {}},
                            {"type": "PathExtractor", "priority": 80, "config": {}},
                            {"type": "PatternExtractor", "priority": 70, "config": {
                                "predefined_patterns": ["email", "url", "phone"]
                            }}
                        ],
                        "embedder": {
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"}
                        },
                        "vector_store": {
                            "type": "ChromaStore",
                            "config": {
                                "collection_name": "test_markdown",
                                "persist_directory": "./vectordb/test_markdown"
                            }
                        },
                        "retrieval_strategy": {
                            "type": "MultiQueryStrategy",
                            "config": {
                                "num_queries": 3,
                                "top_k": 3
                            }
                        }
                    }
                },
                
                # Strategy 4: HTML with entity and link extraction
                {
                    "name": "test_html_news",
                    "description": "Test HTML parsing for news content",
                    "tags": ["test", "html", "news"],
                    "use_cases": ["HTML news extraction test"],
                    "components": {
                        "parser": {
                            "type": "HTMLParser",
                            "config": {
                                "extract_links": True,
                                "extract_metadata": True
                            }
                        },
                        "extractors": [
                            {"type": "EntityExtractor", "priority": 100, "config": {
                                "entity_types": ["PERSON", "ORG", "DATE", "MONEY"]
                            }},
                            {"type": "LinkExtractor", "priority": 90, "config": {}},
                            {"type": "SummaryExtractor", "priority": 80, "config": {"summary_sentences": 3}}
                        ],
                        "embedder": {
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"}
                        },
                        "vector_store": {
                            "type": "ChromaStore",
                            "config": {
                                "collection_name": "test_html",
                                "persist_directory": "./vectordb/test_html"
                            }
                        },
                        "retrieval_strategy": {
                            "type": "RerankedStrategy",
                            "config": {
                                "initial_k": 10,
                                "final_k": 3,
                                "rerank_factors": {
                                    "similarity_weight": 0.7,
                                    "recency_weight": 0.3
                                }
                            }
                        }
                    }
                },
                
                # Strategy 5: Mixed documents with hybrid retrieval
                {
                    "name": "test_hybrid_universal",
                    "description": "Test hybrid universal strategy with mixed content",
                    "tags": ["test", "hybrid", "universal"],
                    "use_cases": ["Hybrid retrieval test"],
                    "components": {
                        "parser": {
                            "type": "DirectoryParser",
                            "config": {
                                "recursive": False,
                                "file_extensions": [".txt", ".md", ".csv"]
                            }
                        },
                        "extractors": [
                            {"type": "KeywordExtractor", "priority": 100, "config": {"algorithm": "yake"}},
                            {"type": "EntityExtractor", "priority": 90, "config": {}},
                            {"type": "TableExtractor", "priority": 80, "config": {}}
                        ],
                        "embedder": {
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text", "batch_size": 8}
                        },
                        "vector_store": {
                            "type": "ChromaStore",
                            "config": {
                                "collection_name": "test_hybrid",
                                "persist_directory": "./vectordb/test_hybrid"
                            }
                        },
                        "retrieval_strategy": {
                            "type": "HybridUniversalStrategy",
                            "config": {
                                "strategies": [
                                    {
                                        "type": "BasicSimilarityStrategy",
                                        "weight": 0.5,
                                        "config": {"top_k": 5}
                                    },
                                    {
                                        "type": "MetadataFilteredStrategy",
                                        "weight": 0.5,
                                        "config": {"top_k": 5}
                                    }
                                ],
                                "combination_method": "weighted_average",
                                "final_k": 3
                            }
                        }
                    }
                },
                
                # Strategy 6: Pattern extraction focus
                {
                    "name": "test_pattern_extraction",
                    "description": "Test pattern extraction capabilities",
                    "tags": ["test", "patterns"],
                    "use_cases": ["Pattern extraction test"],
                    "components": {
                        "parser": {
                            "type": "PlainTextParser",
                            "config": {"chunk_size": 1000}
                        },
                        "extractors": [
                            {"type": "PatternExtractor", "priority": 100, "config": {
                                "predefined_patterns": ["email", "phone", "url", "ip_address"],
                                "custom_patterns": [
                                    {
                                        "name": "case_id",
                                        "pattern": "\\d{4}/[A-Z]{2}/\\d{6}",
                                        "description": "Case reference number"
                                    },
                                    {
                                        "name": "company_id",
                                        "pattern": "\\d{2}-\\d{7}",
                                        "description": "Company ID"
                                    }
                                ]
                            }},
                            {"type": "PathExtractor", "priority": 90, "config": {}}
                        ],
                        "embedder": {
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"}
                        },
                        "vector_store": {
                            "type": "ChromaStore",
                            "config": {
                                "collection_name": "test_patterns",
                                "persist_directory": "./vectordb/test_patterns"
                            }
                        },
                        "retrieval_strategy": {
                            "type": "BasicSimilarityStrategy",
                            "config": {"top_k": 5}
                        }
                    }
                }
            ]
        }
        
        # Write strategies to file
        with open(self.strategy_file, 'w') as f:
            yaml.dump(strategies, f, default_flow_style=False)
            
        print(f"    ✓ Created 6 test strategies")
        
    def get_test_strategies(self) -> List[str]:
        """Get list of test strategy names."""
        return [
            "test_csv_full",
            "test_text_entities",
            "test_markdown_structure",
            "test_html_news",
            "test_hybrid_universal",
            "test_pattern_extraction"
        ]
        
    def run_command(self, command: str, description: str = "") -> Tuple[bool, str, str]:
        """Run a CLI command and return success status and output."""
        if description:
            print(f"\n  {CYAN}▶ {description}{RESET}")
        print(f"    {YELLOW}$ {command}{RESET}")
        
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                timeout=60,
                env={**os.environ, "PYTHONPATH": str(self.base_dir)}
            )
            
            success = result.returncode == 0
            
            if success:
                print(f"    {GREEN}✅ Success{RESET}")
            else:
                print(f"    {RED}❌ Failed (exit code: {result.returncode}){RESET}")
                if result.stderr:
                    # Filter out non-error messages
                    error_lines = [l for l in result.stderr.split('\n') 
                                 if l and not any(skip in l for skip in 
                                 ['WARNING', 'Progress', 'Embedding:', '%|', '🦙'])]
                    if error_lines:
                        print(f"    {RED}Error: {error_lines[0][:100]}...{RESET}")
                        
            return success, result.stdout, result.stderr
            
        except subprocess.TimeoutExpired:
            print(f"    {RED}❌ Command timed out{RESET}")
            return False, "", "Timeout"
        except Exception as e:
            print(f"    {RED}❌ Error: {e}{RESET}")
            return False, "", str(e)
            
    def test_strategy(self, strategy_name: str) -> Dict[str, Any]:
        """Test a single strategy with full workflow."""
        print(f"\n{BOLD}{BLUE}Testing Strategy: {strategy_name}{RESET}")
        print(f"{BLUE}{'─' * 50}{RESET}")
        
        result = {
            "strategy": strategy_name,
            "steps": {},
            "success": False,
            "errors": []
        }
        
        # Load strategy config to get appropriate test files
        with open(self.strategy_file, 'r') as f:
            strategies = yaml.safe_load(f)
            
        strategy_config = None
        for s in strategies['strategies']:
            if s['name'] == strategy_name:
                strategy_config = s
                break
                
        if not strategy_config:
            result["errors"].append("Strategy configuration not found")
            return result
            
        # Determine test files based on parser type
        parser_type = strategy_config['components']['parser']['type']
        if parser_type == "CSVParser":
            test_files = [self.test_dir / "test_data.csv", self.test_dir / "financials.csv"]
        elif parser_type == "PlainTextParser":
            test_files = [self.test_dir / "research.txt", self.test_dir / "contract.txt"]
        elif parser_type == "MarkdownParser":
            test_files = [self.test_dir / "api_docs.md"]
        elif parser_type == "HTMLParser":
            test_files = [self.test_dir / "news.html"]
        elif parser_type == "DirectoryParser":
            test_files = [self.test_dir]
        else:
            test_files = [self.test_dir / "research.txt"]
            
        # Step 1: Ingest documents
        for test_file in test_files:
            success, stdout, stderr = self.run_command(
                f"python cli.py ingest '{test_file}' --strategy {strategy_name} --strategy-file '{self.strategy_file}'",
                f"Ingesting {test_file.name if hasattr(test_file, 'name') else 'directory'}"
            )
            result["steps"][f"ingest_{test_file.name if hasattr(test_file, 'name') else 'directory'}"] = success
            if not success:
                result["errors"].append(f"Ingest failed for {test_file}")
                
        # Step 2: Get collection info
        success, stdout, stderr = self.run_command(
            f"python cli.py info --strategy {strategy_name} --strategy-file '{self.strategy_file}'",
            "Getting collection info"
        )
        result["steps"]["info"] = success
        
        # Parse document count from info
        doc_count = 0
        if success and "count:" in stdout:
            for line in stdout.split('\n'):
                if "count:" in line:
                    try:
                        doc_count = int(line.split("count:")[1].strip())
                    except:
                        pass
                        
        result["document_count"] = doc_count
        
        # Step 3: Test different search queries
        test_queries = [
            ("AI machine learning", "Technical search"),
            ("error bug crash", "Problem search"),
            ("email phone contact", "Contact search"),
            ("2024", "Date search"),
            ("$", "Financial search")
        ]
        
        search_results = []
        for query, desc in test_queries:
            success, stdout, stderr = self.run_command(
                f"python cli.py search '{query}' --strategy {strategy_name} --strategy-file '{self.strategy_file}' --top-k 2",
                f"Search: {desc}"
            )
            result["steps"][f"search_{desc.lower().replace(' ', '_')}"] = success
            
            # Check if results were returned
            if success and "Search Results:" in stdout:
                search_results.append(query)
                
        result["successful_searches"] = len(search_results)
        
        # Step 4: Test manage stats
        success, stdout, stderr = self.run_command(
            f"python cli.py manage --rag-strategy {strategy_name} --strategy-file '{self.strategy_file}' stats",
            "Collection statistics"
        )
        result["steps"]["stats"] = success
        
        # Determine overall success
        total_steps = len(result["steps"])
        successful_steps = sum(1 for v in result["steps"].values() if v)
        result["success_rate"] = successful_steps / total_steps if total_steps > 0 else 0
        result["success"] = result["success_rate"] >= 0.7  # 70% threshold
        
        return result
        
    def run_all_tests(self):
        """Run tests for all strategies."""
        print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}{'Comprehensive Strategy System Test':^60}{RESET}")
        print(f"{BOLD}{CYAN}{'='*60}{RESET}")
        
        # Setup
        self.setup()
        
        # Test each strategy
        for strategy_name in self.get_test_strategies():
            result = self.test_strategy(strategy_name)
            self.results.append(result)
            
            if result["success"]:
                self.passed += 1
            else:
                self.failed += 1
                
            # Brief pause between strategies
            time.sleep(1)
            
        # Print summary
        self.print_summary()
        
        # Teardown
        self.teardown()
        
        return self.failed == 0
        
    def print_summary(self):
        """Print test summary."""
        print(f"\n{BOLD}{CYAN}{'='*60}{RESET}")
        print(f"{BOLD}{CYAN}{'Test Summary':^60}{RESET}")
        print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")
        
        # Detailed results
        print(f"{BOLD}Strategy Results:{RESET}")
        print(f"{BLUE}{'─' * 50}{RESET}")
        
        for result in self.results:
            status = f"{GREEN}✅ PASS{RESET}" if result["success"] else f"{RED}❌ FAIL{RESET}"
            rate = result.get("success_rate", 0) * 100
            docs = result.get("document_count", 0)
            searches = result.get("successful_searches", 0)
            
            print(f"\n{BOLD}{result['strategy']}{RESET}: {status}")
            print(f"  Success Rate: {rate:.0f}%")
            print(f"  Documents: {docs}")
            print(f"  Successful Searches: {searches}/5")
            
            # Show failed steps
            failed_steps = [k for k, v in result["steps"].items() if not v]
            if failed_steps:
                print(f"  {RED}Failed Steps: {', '.join(failed_steps)}{RESET}")
                
            # Show errors
            if result.get("errors"):
                for error in result["errors"][:2]:  # Show first 2 errors
                    print(f"  {RED}Error: {error}{RESET}")
                    
        # Overall summary
        print(f"\n{BOLD}{BLUE}{'='*50}{RESET}")
        print(f"{BOLD}Overall Results:{RESET}")
        print(f"  {GREEN}Passed: {self.passed}/{len(self.results)}{RESET}")
        print(f"  {RED}Failed: {self.failed}/{len(self.results)}{RESET}")
        
        # Component coverage
        print(f"\n{BOLD}Component Coverage:{RESET}")
        parsers_tested = set()
        extractors_tested = set()
        retrievers_tested = set()
        
        for result in self.results:
            # Parse strategy file to get components
            with open(self.strategy_file, 'r') as f:
                strategies = yaml.safe_load(f)
                for s in strategies['strategies']:
                    if s['name'] == result['strategy']:
                        parsers_tested.add(s['components']['parser']['type'])
                        for e in s['components'].get('extractors', []):
                            extractors_tested.add(e['type'])
                        retrievers_tested.add(s['components']['retrieval_strategy']['type'])
                        
        print(f"  Parsers Tested: {len(parsers_tested)} - {', '.join(sorted(parsers_tested))}")
        print(f"  Extractors Tested: {len(extractors_tested)} - {', '.join(sorted(extractors_tested))}")
        print(f"  Retrievers Tested: {len(retrievers_tested)} - {', '.join(sorted(retrievers_tested))}")
        
        # Final verdict
        print(f"\n{BOLD}Final Result: ", end="")
        if self.failed == 0:
            print(f"{GREEN}ALL STRATEGIES PASSED! 🎉{RESET}")
        elif self.failed <= 2:
            print(f"{YELLOW}MOSTLY PASSED (with {self.failed} failures){RESET}")
        else:
            print(f"{RED}MULTIPLE FAILURES ({self.failed} strategies failed){RESET}")
            
        print(f"{BOLD}{CYAN}{'='*60}{RESET}\n")


def main():
    """Main entry point."""
    print(f"{BOLD}{CYAN}Starting Comprehensive Strategy System Test{RESET}")
    print(f"{CYAN}This will test various combinations of parsers, extractors, and retrievers{RESET}\n")
    
    tester = StrategyTester()
    success = tester.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()