#!/usr/bin/env python3
"""Quick start guide for the LlamaFarm Prompts System.

This script provides a quick introduction to the strategy-based prompt system
with live examples that can be run immediately.
"""

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from strategies import StrategyManager

console = Console()

def print_header():
    """Print the quick start header."""
    console.print(Panel(
        """🚀 LlamaFarm Prompts System - Quick Start

Welcome! This guide will help you get started with the strategy-based
prompt management system in just a few minutes.

We'll cover:
1. Loading strategies
2. Executing prompts
3. Using the CLI
4. Customizing strategies""",
        title="Quick Start Guide",
        border_style="blue"
    ))
    console.print()

def show_basic_usage():
    """Show basic usage examples."""
    console.print("[bold cyan]1. Basic Usage[/bold cyan]")
    console.print()
    
    # Show code example
    code = '''from prompts.strategies import StrategyManager

# Initialize manager
manager = StrategyManager()

# Execute a strategy
result = manager.execute_strategy(
    strategy_name="simple_qa",
    inputs={
        "query": "What is machine learning?",
        "context": []
    }
)

print(result)'''
    
    syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Python Code"))
    console.print()
    
    # Actually run the example
    try:
        console.print("[dim]Running example...[/dim]")
        manager = StrategyManager()
        result = manager.execute_strategy(
            strategy_name="simple_qa",
            inputs={
                "query": "What is machine learning?",
                "context": []
            }
        )
        console.print(Panel(result, title="Output", border_style="green"))
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
    
    console.print()

def show_cli_usage():
    """Show CLI usage examples."""
    console.print("[bold cyan]2. CLI Usage[/bold cyan]")
    console.print()
    
    # Create table of CLI commands
    table = Table(title="Common CLI Commands")
    table.add_column("Command", style="cyan")
    table.add_column("Description")
    
    commands = [
        ("strategy list", "List all available strategies"),
        ("strategy show simple_qa", "Show details of a strategy"),
        ("strategy execute simple_qa -q 'Your question'", "Execute a strategy"),
        ("strategy recommend --use-case qa", "Get strategy recommendations"),
        ("template usage", "Show template usage across strategies"),
    ]
    
    for cmd, desc in commands:
        table.add_row(f"python -m prompts.cli_strategy {cmd}", desc)
    
    console.print(table)
    console.print()

def show_available_strategies():
    """Show available strategies."""
    console.print("[bold cyan]3. Available Strategies[/bold cyan]")
    console.print()
    
    manager = StrategyManager()
    strategies = manager.list_strategies()
    
    table = Table(title="Pre-configured Strategies")
    table.add_column("Strategy", style="cyan")
    table.add_column("Description")
    table.add_column("Use Cases")
    
    # Get strategy details
    for name in strategies[:8]:  # Show first 8
        strategy = manager.get_strategy(name)
        if strategy:
            use_cases = ", ".join(strategy.use_cases[:3]) if strategy.use_cases else "General"
            table.add_row(
                name,
                strategy.description[:50] + "..." if len(strategy.description) > 50 else strategy.description,
                use_cases
            )
    
    console.print(table)
    console.print()

def show_customization():
    """Show how to customize strategies."""
    console.print("[bold cyan]4. Customizing Strategies[/bold cyan]")
    console.print()
    
    yaml_example = '''# my_custom_strategy.yaml
my_strategy:
  name: "My Custom Strategy"
  description: "A strategy for my specific use case"
  use_cases: ["custom", "specific"]
  
  templates:
    default:
      template: "qa_basic"
      config:
        temperature: 0.3
    
    specialized:
      - condition:
          query_type: "technical"
        template: "chain_of_thought"
        priority: 10
  
  global_config:
    temperature: 0.7
    max_tokens: 1000'''
    
    syntax = Syntax(yaml_example, "yaml", theme="monokai", line_numbers=True)
    console.print(Panel(syntax, title="Custom Strategy Example"))
    console.print()

def show_next_steps():
    """Show next steps."""
    console.print("[bold cyan]Next Steps[/bold cyan]")
    console.print()
    
    console.print("📚 [bold]Explore More:[/bold]")
    console.print("   • Run individual demos: [cyan]python demos/demo1_simple_qa.py[/cyan]")
    console.print("   • Read the full docs: [cyan]cat README.md[/cyan]")
    console.print("   • Check architecture: [cyan]cat STRUCTURE.md[/cyan]")
    console.print("   • Browse strategies: [cyan]ls strategies/examples/[/cyan]")
    console.print()
    
    console.print("🛠️  [bold]Start Building:[/bold]")
    console.print("   • Create custom strategies in [cyan]strategies/examples/[/cyan]")
    console.print("   • Add new templates in [cyan]templates/[/cyan]")
    console.print("   • Integrate with your app using the [cyan]StrategyManager[/cyan]")
    console.print()

def main():
    """Run the quick start guide."""
    print_header()
    show_basic_usage()
    show_cli_usage()
    show_available_strategies()
    show_customization()
    show_next_steps()
    
    console.print(Panel(
        "🎉 You're ready to start using the LlamaFarm Prompts System!",
        border_style="green"
    ))

if __name__ == "__main__":
    main()