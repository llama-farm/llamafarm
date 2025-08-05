#!/usr/bin/env python3
"""
Setup script for LlamaFarm Prompts System using UV.

This script handles initial setup, dependency installation, and system validation.
"""

import subprocess
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def print_banner():
    """Print setup banner."""
    banner = """
🚀 LlamaFarm Prompts System Setup

This script will:
• Check for UV installation
• Install Python dependencies
• Validate system components
• Run basic functionality tests
• Show quick start examples
    """
    console.print(Panel(banner, title="Setup", border_style="blue"))
    console.print()

def check_uv():
    """Check if UV is installed."""
    console.print("[cyan]Checking UV installation...[/cyan]")
    
    try:
        result = subprocess.run(["uv", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            console.print(f"[green]✅ UV found: {version}[/green]")
            return True
        else:
            console.print("[red]❌ UV not working properly[/red]")
            return False
    except FileNotFoundError:
        console.print("[red]❌ UV not found[/red]")
        console.print("\n[yellow]Please install UV first:[/yellow]")
        console.print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        console.print("or visit: https://docs.astral.sh/uv/getting-started/installation/")
        return False

def install_dependencies():
    """Install dependencies using UV."""
    console.print("[cyan]Installing dependencies with UV...[/cyan]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Install main dependencies
        task1 = progress.add_task("Installing main dependencies...", total=None)
        result = subprocess.run(["uv", "sync"], capture_output=True, text=True)
        progress.remove_task(task1)
        
        if result.returncode == 0:
            console.print("[green]✅ Main dependencies installed[/green]")
        else:
            console.print(f"[red]❌ Failed to install dependencies: {result.stderr}[/red]")
            return False
        
        # Install development dependencies
        task2 = progress.add_task("Installing dev dependencies...", total=None)
        result = subprocess.run(["uv", "sync", "--extra", "dev"], capture_output=True, text=True)
        progress.remove_task(task2)
        
        if result.returncode == 0:
            console.print("[green]✅ Development dependencies installed[/green]")
        else:
            console.print(f"[yellow]⚠️  Dev dependencies failed: {result.stderr}[/yellow]")
        
        # Install test dependencies
        task3 = progress.add_task("Installing test dependencies...", total=None)
        result = subprocess.run(["uv", "sync", "--extra", "test"], capture_output=True, text=True)
        progress.remove_task(task3)
        
        if result.returncode == 0:
            console.print("[green]✅ Test dependencies installed[/green]")
        else:
            console.print(f"[yellow]⚠️  Test dependencies failed: {result.stderr}[/yellow]")
    
    return True

def validate_system():
    """Validate system components."""
    console.print("[cyan]Validating system components...[/cyan]")
    
    # Test strategy CLI
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-m", "prompts.core.cli.strategy_cli", "--help"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            console.print("[green]✅ Strategy CLI working[/green]")
        else:
            console.print(f"[red]❌ Strategy CLI failed: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ Strategy CLI error: {e}[/red]")
        return False
    
    # Test strategy manager
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "from strategies import StrategyManager; print('StrategyManager OK')"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            console.print("[green]✅ StrategyManager working[/green]")
        else:
            console.print(f"[red]❌ StrategyManager failed: {result.stderr}[/red]")
            return False
    except Exception as e:
        console.print(f"[red]❌ StrategyManager error: {e}[/red]")
        return False
    
    # Check templates
    templates_dir = Path("templates")
    if templates_dir.exists():
        template_count = len(list(templates_dir.rglob("schema.json")))
        console.print(f"[green]✅ Found {template_count} templates[/green]")
    else:
        console.print("[yellow]⚠️  Templates directory not found[/yellow]")
    
    # Check strategies
    strategies_dir = Path("strategies/examples")
    if strategies_dir.exists():
        strategy_count = len(list(strategies_dir.glob("*.yaml")))
        console.print(f"[green]✅ Found {strategy_count} example strategies[/green]")
    else:
        console.print("[yellow]⚠️  Strategies examples directory not found[/yellow]")
    
    return True

def run_basic_tests():
    """Run basic functionality tests."""
    console.print("[cyan]Running basic tests...[/cyan]")
    
    # Run a simple strategy execution
    try:
        result = subprocess.run([
            "uv", "run", "python", "-c",
            """
from strategies import StrategyManager
manager = StrategyManager()
strategies = manager.list_strategies()
print(f'Available strategies: {len(strategies)}')
if strategies:
    result = manager.execute_strategy(
        strategy_name=strategies[0],
        inputs={'query': 'Test query', 'context': []}
    )
    print('Basic execution: OK')
"""
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            console.print("[green]✅ Basic execution test passed[/green]")
            console.print(f"[dim]{result.stdout.strip()}[/dim]")
        else:
            console.print(f"[yellow]⚠️  Basic test had issues: {result.stderr}[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Basic test error: {e}[/yellow]")

def show_quick_start():
    """Show quick start commands."""
    console.print()
    console.print(Panel("""
🎯 Quick Start Commands (using UV):

# List available strategies
uv run python -m prompts.core.cli.strategy_cli strategy list

# Execute a strategy
uv run python -m prompts.core.cli.strategy_cli strategy execute simple_qa -q "What is AI?"

# Run demos
uv run python demos/quick_start.py
uv run python demos/run_all_demos.py

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Install additional dependencies
uv add package-name

# Install optional dependencies
uv sync --extra langgraph
""", title="Quick Start", border_style="green"))

def main():
    """Main setup function."""
    print_banner()
    
    # Check UV
    if not check_uv():
        return 1
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    # Validate system
    if not validate_system():
        console.print("[red]❌ System validation failed[/red]")
        return 1
    
    # Run basic tests
    run_basic_tests()
    
    # Show quick start
    show_quick_start()
    
    console.print()
    console.print(Panel(
        "🎉 Setup complete! The LlamaFarm Prompts System is ready to use.",
        border_style="green"
    ))
    
    return 0

if __name__ == "__main__":
    sys.exit(main())