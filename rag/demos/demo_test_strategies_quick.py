#!/usr/bin/env python3
"""
Quick Demo: Test All RAG Strategies from default.yaml
This demo quickly tests each data processing strategy with sample files.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import tempfile
import shutil

console = Console()

def main():
    """Quick test of all strategies from default.yaml"""
    
    console.print(Panel.fit(
        "[bold magenta]RAG Strategy Quick Test[/bold magenta]\n"
        "Testing data processing strategies from default.yaml",
        title="🚀 Quick Test"
    ))
    
    cli_path = Path(__file__).parent.parent / "cli.py"
    config_file = Path(__file__).parent.parent.parent / "config" / "templates" / "default.yaml"
    temp_dirs = []
    results = []
    
    # Define test cases for each strategy in default.yaml
    test_cases = [
        ("pdf_processing", "fda_letters/761315_2025_Orig1s000OtherActionLtrs.pdf", "PDF Processing"),
        ("text_processing", "documents/test_doc.txt", "Text Processing"),
        ("markdown_processing", "code_documentation/api_reference.md", "Markdown Processing"),
        ("csv_processing", "customer_support/support_tickets.csv", "CSV Processing"),
        ("multi_format_llamaindex", "research_papers/test_paper.txt", "Multi-format LlamaIndex"),
        ("auto_processing", "documents/test_doc.txt", "Auto Processing (Generic)")
    ]
    
    for strategy, test_file, description in test_cases:
        console.print(f"\n[cyan]Testing: {description}[/cyan]")
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp(prefix=f"test_{strategy}_")
        temp_dirs.append(temp_dir)
        
        # Build file path
        file_path = Path(__file__).parent / "static_samples" / test_file
        
        if not file_path.exists():
            console.print(f"  [yellow]⚠ File not found: {test_file}[/yellow]")
            results.append((strategy, "SKIP", "File not found"))
            continue
        
        # Run CLI command
        full_strategy = f"{strategy}_main_database"
        cmd = [
            "python", str(cli_path),
            "--strategy-file", str(config_file),
            "--quiet",  # Less verbose output
            "ingest",
            str(file_path),
            "--strategy", full_strategy
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(cli_path.parent))
            if result.returncode == 0:
                console.print(f"  [green]✅ Success[/green]")
                results.append((strategy, "PASS", ""))
            else:
                error = result.stderr.split('\n')[0] if result.stderr else "Unknown error"
                console.print(f"  [red]❌ Failed: {error[:50]}...[/red]")
                results.append((strategy, "FAIL", error[:50]))
        except subprocess.TimeoutExpired:
            console.print(f"  [yellow]⏱ Timeout[/yellow]")
            results.append((strategy, "TIMEOUT", "Processing took too long"))
        except Exception as e:
            console.print(f"  [red]❌ Error: {str(e)[:50]}[/red]")
            results.append((strategy, "ERROR", str(e)[:50]))
    
    # Display summary
    console.print("\n" + "="*60)
    table = Table(title="Test Results Summary", show_header=True)
    table.add_column("Strategy", style="cyan")
    table.add_column("Result", justify="center")
    table.add_column("Notes", style="dim")
    
    for strategy, status, notes in results:
        color = "green" if status == "PASS" else "yellow" if status in ["SKIP", "TIMEOUT"] else "red"
        table.add_row(strategy, f"[{color}]{status}[/{color}]", notes)
    
    console.print(table)
    
    # Cleanup
    for temp_dir in temp_dirs:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
    
    # Summary
    passed = sum(1 for _, s, _ in results if s == "PASS")
    total = len(results)
    console.print(f"\n[bold]Result: {passed}/{total} strategies passed[/bold]")
    
    if passed == total:
        console.print("[green]✅ All strategies working![/green]")
    elif passed > 0:
        console.print("[yellow]⚠ Some strategies need attention[/yellow]")
    else:
        console.print("[red]❌ Critical issues detected[/red]")

if __name__ == "__main__":
    main()