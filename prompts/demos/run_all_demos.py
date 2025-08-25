#!/usr/bin/env python3
"""Run all demos for the LlamaFarm Prompts System.

This script runs through all 5 comprehensive demos showing different use cases
of the strategy-based prompt management system.
"""

import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

def print_banner():
    """Print the demo banner."""
    banner = """
    🚀 LlamaFarm Prompts System - Demo Suite
    
    Running all 5 comprehensive demos:
    • Simple Q&A
    • Customer Support  
    • Code Assistant
    • RAG Research
    • Advanced Reasoning
    """
    console.print(Panel(banner, title="Demo Runner", border_style="blue"))
    console.print()

def run_demo(demo_path: Path, description: str):
    """Run a single demo."""
    console.print(f"[cyan]Running: {description}[/cyan]")
    console.print(f"[dim]Script: {demo_path.name}[/dim]")
    console.print()
    
    try:
        # Run the demo using UV
        result = subprocess.run(
            ["uv", "run", "python", str(demo_path)],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            console.print(result.stdout)
            console.print(f"[green]✅ {description} completed successfully[/green]")
        else:
            console.print(f"[red]❌ {description} failed[/red]")
            console.print(f"[red]{result.stderr}[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Error running {description}: {e}[/red]")
    
    console.print()
    console.print("-" * 80)
    console.print()

def main():
    """Run all demos."""
    print_banner()
    
    demos_dir = Path(__file__).parent
    
    demos = [
        ("demo1_simple_qa.py", "Simple Q&A Demo"),
        ("demo2_customer_support.py", "Customer Support Demo"),
        ("demo3_code_assistant.py", "Code Assistant Demo"),
        ("demo4_rag_research.py", "RAG Research Demo"),
        ("demo5_advanced_reasoning.py", "Advanced Reasoning Demo"),
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running demos...", total=len(demos))
        
        for demo_file, description in demos:
            demo_path = demos_dir / demo_file
            if demo_path.exists():
                run_demo(demo_path, description)
                progress.advance(task)
            else:
                console.print(f"[yellow]⚠️  Demo file not found: {demo_file}[/yellow]")
    
    console.print()
    console.print(Panel(
        "✅ All demos completed! Check the output above for results.",
        title="Demo Suite Complete",
        border_style="green"
    ))

if __name__ == "__main__":
    main()