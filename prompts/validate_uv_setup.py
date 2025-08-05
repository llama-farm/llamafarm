#!/usr/bin/env python3
"""
Validate UV Setup for LlamaFarm Prompts System

This script validates that the entire system works properly with UV.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_command(cmd, description):
    """Run a command and return success/failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            console.print(f"[green]✅ {description}[/green]")
            return True, result.stdout
        else:
            console.print(f"[red]❌ {description}[/red]")
            console.print(f"[dim]Error: {result.stderr}[/dim]")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        console.print(f"[yellow]⏱️  {description} (timeout)[/yellow]")
        return False, "Timeout"
    except Exception as e:
        console.print(f"[red]❌ {description} (exception: {e})[/red]")
        return False, str(e)

def main():
    """Main validation function."""
    console.print(Panel(
        "🔍 LlamaFarm Prompts System - UV Setup Validation",
        border_style="blue"
    ))
    console.print()
    
    # Validation tests
    tests = [
        (["uv", "--version"], "UV installation"),
        (["uv", "sync", "--dry-run"], "UV dependency resolution"),
        (["uv", "run", "python", "--version"], "UV Python execution"),
        (["uv", "run", "python", "-c", "import prompts; print('Package import OK')"], "Package import"),
        (["uv", "run", "python", "-c", "from strategies import StrategyManager; print('Strategy manager OK')"], "Strategy manager import"),
        (["uv", "run", "python", "-m", "prompts.core.cli.strategy_cli", "--help"], "Strategy CLI"),
        (["uv", "run", "python", "-c", "from strategies import StrategyManager; m = StrategyManager(); print(f'Strategies: {len(m.list_strategies())}')"], "Strategy loading"),
        (["uv", "run", "pytest", "--version"], "Pytest availability"),
        (["uv", "run", "black", "--version"], "Black formatter"),
        (["uv", "run", "python", "demos/quick_start.py", "--help"], "Demo scripts (if --help supported)"),
    ]
    
    passed = 0
    failed = 0
    
    console.print("[cyan]Running validation tests...[/cyan]")
    console.print()
    
    for cmd, description in tests:
        success, output = run_command(cmd, description)
        if success:
            passed += 1
        else:
            failed += 1
    
    console.print()
    
    # Results table
    table = Table(title="Validation Results")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    
    table.add_row("[green]Passed[/green]", str(passed))
    table.add_row("[red]Failed[/red]", str(failed))
    table.add_row("[blue]Total[/blue]", str(passed + failed))
    
    console.print(table)
    console.print()
    
    # Final result
    if failed == 0:
        console.print(Panel(
            "🎉 All validations passed! The system is ready to use with UV.",
            border_style="green"
        ))
        
        console.print("\n[bold]Next steps:[/bold]")
        console.print("• Run: [cyan]uv run python demos/quick_start.py[/cyan]")
        console.print("• Run: [cyan]make run-demos[/cyan]")
        console.print("• Run: [cyan]uv run pytest[/cyan]")
        
        return 0
    else:
        console.print(Panel(
            f"❌ {failed} validation(s) failed. Please check the errors above.",
            border_style="red"
        ))
        
        console.print("\n[bold]Troubleshooting:[/bold]")
        console.print("• Ensure UV is installed: [cyan]curl -LsSf https://astral.sh/uv/install.sh | sh[/cyan]")
        console.print("• Run setup: [cyan]uv run python setup.py[/cyan]")
        console.print("• Install dependencies: [cyan]uv sync[/cyan]")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())