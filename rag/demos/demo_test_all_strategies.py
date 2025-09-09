#!/usr/bin/env python3
"""
Demo: Test All RAG Strategies from default.yaml using the CLI
This demo tests each data processing strategy with appropriate sample files.
"""

import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import time
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich import print as rprint
import tempfile
import shutil

console = Console()

class StrategyTester:
    """Test all strategies from default.yaml using the CLI"""
    
    def __init__(self):
        self.console = console
        self.results = []
        self.temp_dirs = []
        self.cli_path = Path(__file__).parent.parent / "cli.py"
        self.config_file = Path(__file__).parent.parent.parent / "config" / "templates" / "default.yaml"
        
    def cleanup(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
                
    def run_cli_command(self, args: List[str]) -> tuple[bool, str, str]:
        """Run a CLI command and return success status, stdout, and stderr"""
        try:
            result = subprocess.run(
                ["python", str(self.cli_path)] + args,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.cli_path.parent)
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
            
    def test_strategy(self, strategy_name: str, test_files: List[str], description: str) -> Dict[str, Any]:
        """Test a single strategy with given files"""
        
        self.console.print(f"\n[bold cyan]Testing Strategy: {strategy_name}[/bold cyan]")
        self.console.print(f"[dim]{description}[/dim]")
        
        result = {
            "strategy": strategy_name,
            "description": description,
            "files_tested": test_files,
            "status": "pending",
            "documents_processed": 0,
            "errors": []
        }
        
        # Strategy naming convention: {data_processing_strategy}_{database_name}
        full_strategy_name = f"{strategy_name}_main_database"
        
        # Process each test file
        for test_file in test_files:
            file_path = Path(__file__).parent / "static_samples" / test_file
            
            if not file_path.exists():
                self.console.print(f"[yellow]⚠ File not found: {test_file}[/yellow]")
                continue
                
            self.console.print(f"  Processing: [green]{test_file}[/green]")
            
            # Build CLI command for ingestion
            # Note: --strategy-file must come BEFORE the command
            cli_args = [
                "--strategy-file", str(self.config_file),
                "--verbose",
                "ingest",
                str(file_path),
                "--strategy", full_strategy_name
            ]
            
            # Run the CLI command
            success, stdout, stderr = self.run_cli_command(cli_args)
            
            if success:
                self.console.print(f"    ✓ Successfully processed")
                result["documents_processed"] += 1
            else:
                error_msg = stderr if stderr else "Unknown error"
                self.console.print(f"    [red]✗ Failed: {error_msg}[/red]")
                result["errors"].append(f"{test_file}: {error_msg}")
        
        # Determine overall status
        if result["documents_processed"] == len(test_files):
            result["status"] = "success"
        elif result["documents_processed"] > 0:
            result["status"] = "partial"
        else:
            result["status"] = "failed"
            
        self.results.append(result)
        return result
        
    def test_query(self, strategy_name: str, query: str):
        """Test querying with a strategy"""
        self.console.print(f"\n[bold yellow]Testing Query for {strategy_name}:[/bold yellow]")
        self.console.print(f"Query: '{query}'")
        
        full_strategy_name = f"{strategy_name}_main_database"
        
        # Note: --strategy-file must come BEFORE the command
        cli_args = [
            "--strategy-file", str(self.config_file),
            "--quiet",  # Use quiet mode for cleaner output
            "search",  # CLI uses 'search' not 'query'
            query,
            "--strategy", full_strategy_name,
            "--top-k", "3"  # Limit results
        ]
        
        success, stdout, stderr = self.run_cli_command(cli_args)
        
        if success:
            # Count and display results
            result_count = stdout.count("Result #")
            if result_count > 0:
                self.console.print(f"[green]✓ Query returned {result_count} results[/green]")
                # Extract and show first result snippet
                if "Content:" in stdout:
                    content_start = stdout.find("Content:") + 8
                    content_end = stdout.find("\n", content_start + 1)
                    if content_end > content_start:
                        snippet = stdout[content_start:content_end].strip()[:100]
                        self.console.print(f"[dim]First result: {snippet}...[/dim]")
            else:
                self.console.print("[yellow]⚠ No results found[/yellow]")
        else:
            self.console.print(f"[red]✗ Query failed: {stderr[:200]}[/red]")
        
    def run_all_tests(self):
        """Run tests for all strategies defined in default.yaml"""
        
        self.console.print(Panel.fit(
            "[bold magenta]RAG Strategy Test Suite[/bold magenta]\n"
            "Testing all data processing strategies from default.yaml using CLI",
            title="🧪 Test Suite"
        ))
        
        # Check if CLI is accessible
        self.console.print("\n[dim]Checking CLI availability...[/dim]")
        success, stdout, stderr = self.run_cli_command(["--help"])
        if not success:
            self.console.print(f"[red]CLI not accessible: {stderr}[/red]")
            return
        self.console.print("[green]✓ CLI is accessible[/green]")
        
        # Define test cases for each strategy in default.yaml
        test_cases = [
            {
                "strategy": "pdf_processing",
                "description": "Standard PDF document processing",
                "files": [
                    "fda_letters/761315_2025_Orig1s000OtherActionLtrs.pdf",
                    "business_reports/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf"
                ],
                "test_query": "What are the key findings?"
            },
            {
                "strategy": "text_processing",
                "description": "Plain text document processing",
                "files": [
                    "research_papers/llm_scaling_laws.txt",
                    "research_papers/transformer_architecture.txt",
                    "customer_support/knowledge_base.txt"
                ],
                "test_query": "What are the main concepts discussed?"
            },
            {
                "strategy": "markdown_processing",
                "description": "Markdown document processing with structure preservation",
                "files": [
                    "code_documentation/api_reference.md",
                    "code_documentation/implementation_guide.md",
                    "code_documentation/best_practices.md"
                ],
                "test_query": "What are the API endpoints?"
            },
            {
                "strategy": "csv_processing",
                "description": "CSV and structured data processing",
                "files": [
                    "customer_support/support_tickets.csv",
                    "business_reports/supply_chain_metrics.csv"
                ],
                "test_query": "What issues are reported?"
            },
            {
                "strategy": "multi_format_llamaindex",
                "description": "Multi-format document processing using LlamaIndex parsers",
                "files": [
                    "fda_letters/761258_2025_Orig1s000OtherActionLtrs.pdf",
                    "customer_support/support_tickets.csv",
                    "code_documentation/api_reference.md",
                    "research_papers/test_paper.txt"
                ],
                "test_query": "Summarize the main points"
            },
            {
                "strategy": "auto_processing",
                "description": "Generic text processing for various file types",
                "files": [
                    "news_articles/ai_breakthrough.html",
                    "documents/test_doc.txt"
                ],
                "test_query": "What is this about?"
            }
        ]
        
        # Test each strategy
        for test_case in test_cases:
            self.test_strategy(
                test_case["strategy"],
                test_case["files"],
                test_case["description"]
            )
            
            # Optionally test query after ingestion
            if test_case.get("test_query"):
                self.test_query(test_case["strategy"], test_case["test_query"])
            
            time.sleep(1)  # Small delay between tests
            
        # Display results summary
        self.display_results()
        
    def display_results(self):
        """Display test results in a formatted table"""
        
        self.console.print("\n" + "="*80)
        self.console.print(Panel.fit(
            "[bold green]Test Results Summary[/bold green]",
            title="📊 Results"
        ))
        
        # Create results table
        table = Table(title="Strategy Test Results", show_header=True, header_style="bold magenta")
        table.add_column("Strategy", style="cyan", width=25)
        table.add_column("Status", justify="center", width=10)
        table.add_column("Files Processed", justify="center", width=15)
        table.add_column("Errors", justify="center", width=8)
        
        total_success = 0
        total_partial = 0
        total_failed = 0
        
        for result in self.results:
            status_style = "green" if result["status"] == "success" else "yellow" if result["status"] == "partial" else "red"
            status_icon = "✅" if result["status"] == "success" else "⚠️" if result["status"] == "partial" else "❌"
            
            table.add_row(
                result["strategy"],
                f"[{status_style}]{status_icon} {result['status'].upper()}[/{status_style}]",
                f"{result['documents_processed']}/{len(result['files_tested'])}",
                str(len(result["errors"]))
            )
            
            if result["status"] == "success":
                total_success += 1
            elif result["status"] == "partial":
                total_partial += 1
            else:
                total_failed += 1
                
        self.console.print(table)
        
        # Print summary statistics
        self.console.print("\n[bold]Summary:[/bold]")
        self.console.print(f"  ✅ Successful: {total_success}")
        self.console.print(f"  ⚠️  Partial: {total_partial}")
        self.console.print(f"  ❌ Failed: {total_failed}")
        
        # Print any errors
        if any(r["errors"] for r in self.results):
            self.console.print("\n[bold red]Errors Encountered:[/bold red]")
            for result in self.results:
                if result["errors"]:
                    self.console.print(f"\n[cyan]{result['strategy']}:[/cyan]")
                    for error in result["errors"][:3]:  # Show first 3 errors
                        self.console.print(f"  • {error[:200]}...")  # Truncate long errors

def main():
    """Main function to run the demo"""
    
    tester = StrategyTester()
    
    try:
        # Run all tests
        tester.run_all_tests()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Tests interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
    finally:
        # Clean up temporary directories
        tester.cleanup()
        console.print("\n[dim]Cleanup complete[/dim]")

if __name__ == "__main__":
    main()