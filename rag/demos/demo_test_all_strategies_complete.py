#!/usr/bin/env python3
"""
Complete Demo: Test All RAG Strategies from default.yaml
This demo tests both ingestion and search for each data processing strategy.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
import tempfile
import shutil
import json

console = Console()

class CompleteStrategyTester:
    """Test all strategies from default.yaml with both ingestion and search"""
    
    def __init__(self):
        self.console = console
        self.results = []
        self.cli_path = Path(__file__).parent.parent / "cli.py"
        self.config_file = Path(__file__).parent.parent.parent / "config" / "templates" / "default.yaml"
        self.base_temp_dir = None
        
    def setup(self):
        """Create base temporary directory for all tests"""
        self.base_temp_dir = tempfile.mkdtemp(prefix="rag_test_complete_")
        self.console.print(f"[dim]Using temp directory: {self.base_temp_dir}[/dim]")
        
    def cleanup(self):
        """Clean up temporary directories"""
        if self.base_temp_dir and Path(self.base_temp_dir).exists():
            shutil.rmtree(self.base_temp_dir)
            self.console.print("[dim]Cleanup complete[/dim]")
                
    def run_cli_command(self, args: List[str], timeout: int = 30) -> Tuple[bool, str, str]:
        """Run a CLI command and return success status, stdout, and stderr"""
        try:
            result = subprocess.run(
                ["python", str(self.cli_path)] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.cli_path.parent),
                env={**os.environ, "TOKENIZERS_PARALLELISM": "false"}  # Suppress tokenizer warnings
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def create_test_config(self, strategy_name: str) -> str:
        """Create a temporary config file with unique vector store for this test"""
        # Read the default config
        with open(self.config_file, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
        
        # Modify the vector store path to be unique for this test
        if 'rag' in config and 'databases' in config['rag']:
            for db in config['rag']['databases']:
                if db.get('name') == 'main_database':
                    db['config']['persist_directory'] = f"{self.base_temp_dir}/{strategy_name}_db"
        
        # Save to temporary config file
        temp_config = f"{self.base_temp_dir}/{strategy_name}_config.yaml"
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        return temp_config
            
    def test_strategy_complete(self, strategy_name: str, test_files: List[str], 
                               test_query: str, description: str) -> Dict[str, Any]:
        """Test a single strategy with both ingestion and search"""
        
        self.console.print(f"\n[bold cyan]Testing Strategy: {strategy_name}[/bold cyan]")
        self.console.print(f"[dim]{description}[/dim]")
        
        result = {
            "strategy": strategy_name,
            "description": description,
            "ingestion_success": False,
            "search_success": False,
            "files_processed": 0,
            "search_results": 0,
            "errors": []
        }
        
        # Create unique config for this test
        temp_config = self.create_test_config(strategy_name)
        full_strategy_name = f"{strategy_name}_main_database"
        
        # Step 1: Ingest documents
        self.console.print("\n  [yellow]Step 1: Ingesting documents[/yellow]")
        
        for test_file in test_files:
            file_path = Path(__file__).parent / "static_samples" / test_file
            
            if not file_path.exists():
                self.console.print(f"    [yellow]⚠ File not found: {test_file}[/yellow]")
                continue
                
            self.console.print(f"    Processing: [green]{test_file}[/green]")
            
            # Build CLI command for ingestion
            cli_args = [
                "--strategy-file", temp_config,
                "--quiet",  # Less verbose
                "ingest",
                str(file_path),
                "--strategy", full_strategy_name
            ]
            
            # Run the CLI command
            success, stdout, stderr = self.run_cli_command(cli_args, timeout=20)
            
            if success:
                self.console.print(f"      ✓ Successfully ingested")
                result["files_processed"] += 1
            else:
                error_msg = stderr.split('\n')[0] if stderr else "Unknown error"
                self.console.print(f"      [red]✗ Failed: {error_msg}[/red]")
                result["errors"].append(f"Ingest {test_file}: {error_msg}")
        
        result["ingestion_success"] = result["files_processed"] > 0
        
        # Step 2: Test search
        if result["ingestion_success"]:
            self.console.print(f"\n  [yellow]Step 2: Testing search[/yellow]")
            self.console.print(f"    Query: '{test_query}'")
            
            cli_args = [
                "--strategy-file", temp_config,
                "--quiet",
                "search",
                test_query,
                "--strategy", full_strategy_name,
                "--top-k", "3"
            ]
            
            success, stdout, stderr = self.run_cli_command(cli_args, timeout=15)
            
            if success:
                # Count results in output
                result_count = stdout.count("Result #")
                if result_count > 0:
                    self.console.print(f"      ✓ Search returned {result_count} results")
                    result["search_results"] = result_count
                    result["search_success"] = True
                    
                    # Show first result snippet
                    if "Content:" in stdout:
                        content_start = stdout.find("Content:") + 8
                        content_end = stdout.find("\n", content_start + 1)
                        if content_end > content_start:
                            snippet = stdout[content_start:content_end].strip()[:100]
                            self.console.print(f"      [dim]Sample: {snippet}...[/dim]")
                else:
                    self.console.print(f"      [yellow]⚠ No results found[/yellow]")
                    result["errors"].append(f"Search: No results for query '{test_query}'")
            else:
                error_msg = stderr.split('\n')[0] if stderr else "Unknown error"
                self.console.print(f"      [red]✗ Search failed: {error_msg}[/red]")
                result["errors"].append(f"Search: {error_msg}")
        
        # Determine overall status
        if result["ingestion_success"] and result["search_success"]:
            result["status"] = "success"
        elif result["ingestion_success"] or result["search_success"]:
            result["status"] = "partial"
        else:
            result["status"] = "failed"
            
        self.results.append(result)
        return result
        
    def run_all_tests(self):
        """Run tests for all strategies defined in default.yaml"""
        
        self.console.print(Panel.fit(
            "[bold magenta]Complete RAG Strategy Test Suite[/bold magenta]\n"
            "Testing ingestion AND search for all strategies from default.yaml",
            title="🧪 Complete Test Suite"
        ))
        
        self.setup()
        
        # Define test cases for each strategy in default.yaml
        test_cases = [
            {
                "strategy": "pdf_processing",
                "description": "Standard PDF document processing",
                "files": ["fda_letters/761315_2025_Orig1s000OtherActionLtrs.pdf"],
                "query": "FDA approval"
            },
            {
                "strategy": "text_processing",
                "description": "Plain text document processing",
                "files": [
                    "research_papers/llm_scaling_laws.txt",
                    "documents/test_doc.txt"
                ],
                "query": "scaling"
            },
            {
                "strategy": "markdown_processing",
                "description": "Markdown document processing with structure preservation",
                "files": ["code_documentation/api_reference.md"],
                "query": "API endpoint"
            },
            {
                "strategy": "csv_processing",
                "description": "CSV and structured data processing",
                "files": ["customer_support/support_tickets.csv"],
                "query": "customer issue"
            },
            {
                "strategy": "multi_format_llamaindex",
                "description": "Multi-format document processing using LlamaIndex parsers",
                "files": [
                    "code_documentation/api_reference.md",
                    "research_papers/test_paper.txt"
                ],
                "query": "research"
            },
            {
                "strategy": "auto_processing",
                "description": "Generic text processing for various file types",
                "files": ["documents/test_doc.txt"],
                "query": "test"
            }
        ]
        
        # Test each strategy
        for test_case in test_cases:
            self.test_strategy_complete(
                test_case["strategy"],
                test_case["files"],
                test_case["query"],
                test_case["description"]
            )
            time.sleep(1)  # Small delay between tests
            
        # Display results summary
        self.display_results()
        
    def display_results(self):
        """Display test results in a formatted table"""
        
        self.console.print("\n" + "="*80)
        self.console.print(Panel.fit(
            "[bold green]Complete Test Results Summary[/bold green]",
            title="📊 Results"
        ))
        
        # Create results table
        table = Table(title="Strategy Test Results", show_header=True, header_style="bold magenta")
        table.add_column("Strategy", style="cyan", width=25)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Ingested", justify="center", width=10)
        table.add_column("Search", justify="center", width=10)
        table.add_column("Results", justify="center", width=8)
        
        total_success = 0
        total_partial = 0
        total_failed = 0
        
        for result in self.results:
            status = result.get("status", "unknown")
            status_style = "green" if status == "success" else "yellow" if status == "partial" else "red"
            status_icon = "✅" if status == "success" else "⚠️" if status == "partial" else "❌"
            
            ingestion_icon = "✓" if result["ingestion_success"] else "✗"
            ingestion_style = "green" if result["ingestion_success"] else "red"
            
            search_icon = "✓" if result["search_success"] else "✗"
            search_style = "green" if result["search_success"] else "red"
            
            table.add_row(
                result["strategy"],
                f"[{status_style}]{status_icon}[/{status_style}]",
                f"[{ingestion_style}]{ingestion_icon} ({result['files_processed']})[/{ingestion_style}]",
                f"[{search_style}]{search_icon}[/{search_style}]",
                str(result["search_results"])
            )
            
            if status == "success":
                total_success += 1
            elif status == "partial":
                total_partial += 1
            else:
                total_failed += 1
                
        self.console.print(table)
        
        # Print summary statistics
        self.console.print("\n[bold]Summary:[/bold]")
        self.console.print(f"  ✅ Fully Successful: {total_success}/{len(self.results)}")
        self.console.print(f"  ⚠️  Partially Successful: {total_partial}/{len(self.results)}")
        self.console.print(f"  ❌ Failed: {total_failed}/{len(self.results)}")
        
        # Print any errors
        if any(r["errors"] for r in self.results):
            self.console.print("\n[bold red]Issues Encountered:[/bold red]")
            for result in self.results:
                if result["errors"]:
                    self.console.print(f"\n[cyan]{result['strategy']}:[/cyan]")
                    for error in result["errors"][:2]:  # Show first 2 errors
                        self.console.print(f"  • {error[:100]}")

def main():
    """Main function to run the complete demo"""
    
    tester = CompleteStrategyTester()
    
    try:
        # Check if Ollama is running
        console.print("[dim]Checking requirements...[/dim]")
        
        # Run all tests
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
    finally:
        # Clean up temporary directories
        tester.cleanup()

if __name__ == "__main__":
    main()