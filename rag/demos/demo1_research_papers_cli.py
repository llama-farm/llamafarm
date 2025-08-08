#!/usr/bin/env python3
"""
Demo 1: Research Paper Analysis System (CLI Version)
Demonstrates advanced RAG capabilities using the CLI for all operations.
This showcases the platform's capabilities through CLI commands rather than direct API usage.
"""

import subprocess
import sys
import time
import json
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Setup rich console for beautiful output
console = Console()


def run_cli_command(command: str, verbose: bool = False, show_output: bool = True) -> tuple[int, str, str]:
    """
    Run a CLI command and return the result.
    
    Args:
        command: The CLI command to run
        verbose: Whether to add --verbose flag
        show_output: Whether to display the command being run
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    # Add strategy file if not already present
    if '--strategy-file' not in command and 'cli.py' in command:
        parts = command.split()
        cli_index = next((i for i, part in enumerate(parts) if 'cli.py' in part), -1)
        if cli_index >= 0:
            parts.insert(cli_index + 1, '--strategy-file')
            parts.insert(cli_index + 2, 'demos/demo_strategies.yaml')
            command = ' '.join(parts)
    
    if verbose:
        # Add --verbose flag if not already present
        if '--verbose' not in command:
            # Insert --verbose flag right after 'cli.py' and strategy-file
            parts = command.split()
            cli_index = next((i for i, part in enumerate(parts) if 'cli.py' in part), -1)
            if cli_index >= 0:
                # Find insertion point after strategy-file args
                insert_index = cli_index + 1
                if insert_index < len(parts) and parts[insert_index] == '--strategy-file':
                    insert_index += 2  # Skip --strategy-file and its value
                parts.insert(insert_index, '--verbose')
                command = ' '.join(parts)
    
    if show_output:
        console.print(f"\n[bold cyan]Running command:[/bold cyan]")
        console.print(f"[dim]$ {command}[/dim]")
    
    # Run the command
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent  # Run from rag directory
    )
    
    return result.returncode, result.stdout, result.stderr


def print_section_header(title: str, emoji: str = "🔬"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def demonstrate_research_paper_rag_cli():
    """Demonstrate RAG system using CLI commands exclusively."""
    
    print_section_header("🦙 Demo 1: Research Paper Analysis System (CLI Version)", "🔬")
    
    console.print("\n[bold green]This demo showcases the platform capabilities using CLI commands:[/bold green]")
    console.print("• [bold cyan]All operations through CLI[/bold cyan] (no direct API usage)")
    console.print("• Strategy-based configuration from demo_strategies.yaml")
    console.print("• Advanced parsing and extraction capabilities")
    console.print("• Semantic search with verbose output options")
    console.print("• Complete end-to-end workflow using CLI")
    
    # Test CLI availability
    print_section_header("CLI System Check", "🔧")
    
    console.print("🔍 [bold cyan]Checking CLI availability...[/bold cyan]")
    
    returncode, stdout, stderr = run_cli_command("python cli.py --help", show_output=False)
    if returncode != 0:
        console.print("❌ [red]CLI not available. Make sure you're in the rag directory.[/red]")
        return
    
    console.print("✅ [bold green]CLI is available and ready![/bold green]")
    
    # Initialize the collection for research papers
    print_section_header("Initialize Research Papers Collection", "🗄️")
    
    console.print("🚀 [bold cyan]Initializing research papers collection...[/bold cyan]")
    
    # First, let's check if the collection already exists
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy research_papers_demo"
    )
    
    if "document_count" in stdout:
        console.print("[dim]💡 Collection already exists, continuing...[/dim]")
    else:
        console.print("[dim]💡 Creating new collection...[/dim]")
    
    # Ingest research papers using the strategy
    print_section_header("Research Paper Ingestion", "📚")
    
    console.print("[bold cyan]🔄 Ingesting research papers using strategy configuration...[/bold cyan]")
    console.print("[dim]💡 All parsing and extraction configured in demo_strategies.yaml[/dim]")
    
    # Run ingestion with verbose output
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --verbose --strategy-file demos/demo_strategies.yaml ingest --strategy research_papers_demo demos/static_samples/research_papers/"
    )
    
    if returncode == 0:
        console.print("✅ [bold green]Research papers successfully ingested![/bold green]")
        
        # Parse output to show statistics if available
        if "Documents processed:" in stdout:
            for line in stdout.split('\n'):
                if "Documents processed:" in line or "Errors encountered:" in line:
                    console.print(f"   {line.strip()}")
    else:
        console.print(f"⚠️ [yellow]Ingestion completed with warnings[/yellow]")
        if stderr:
            console.print(f"[dim]{stderr[:500]}[/dim]")
    
    # Show collection information
    print_section_header("Collection Statistics", "📊")
    
    console.print("📈 [bold cyan]Retrieving collection statistics...[/bold cyan]")
    
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy research_papers_demo"
    )
    
    if returncode == 0 and stdout:
        # Parse and display the info nicely
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        for line in stdout.split('\n'):
            if ':' in line and not line.startswith('python'):
                key, value = line.split(':', 1)
                stats_table.add_row(key.strip(), value.strip())
        
        console.print(stats_table)
    
    # Demonstrate research queries using CLI search
    print_section_header("Research Query Demonstration", "🔍")
    
    research_queries = [
        "What are the key findings about transformer architecture performance?",
        "How do scaling laws affect neural language model performance?",
        "What statistical evidence supports the scaling law relationships?"
    ]
    
    console.print("🎯 [bold cyan]Running research-focused queries using CLI search:[/bold cyan]")
    console.print("[dim]💡 Using --verbose flag to show detailed results[/dim]")
    
    for i, query in enumerate(research_queries, 1):
        console.print(f"\n[bold cyan]Research Query #{i}:[/bold cyan]")
        console.print(f"Query: [yellow]{query}[/yellow]")
        
        # Run search with verbose output
        returncode, stdout, stderr = run_cli_command(
            f'python cli.py --verbose --strategy-file demos/demo_strategies.yaml search --strategy research_papers_demo "{query}"'
        )
        
        if returncode == 0:
            # Show the output directly - it's already formatted nicely by the CLI
            console.print(stdout)
        else:
            console.print(f"⚠️ [yellow]Search completed with warnings[/yellow]")
        
        if i < len(research_queries):
            time.sleep(1.5)  # Pause for readability
    
    # Demonstrate quiet mode for scripting
    print_section_header("Quiet Mode for Scripting", "🤫")
    
    console.print("🔇 [bold cyan]Demonstrating quiet mode for script integration...[/bold cyan]")
    console.print("[dim]💡 Useful for parsing output in scripts[/dim]")
    
    returncode, stdout, stderr = run_cli_command(
        'python cli.py --quiet --strategy-file demos/demo_strategies.yaml search --strategy research_papers_demo "transformer performance"'
    )
    
    console.print("\n[bold]Quiet mode output (structured for scripts):[/bold]")
    console.print(Panel(stdout[:1000] if stdout else "No output", title="Quiet Mode Output", border_style="dim"))
    
    # Show advanced CLI features
    print_section_header("Advanced CLI Features", "🚀")
    
    console.print("🎯 [bold green]Additional CLI capabilities demonstrated:[/bold green]")
    
    features_table = Table(show_header=True, header_style="bold cyan")
    features_table.add_column("Feature", style="yellow")
    features_table.add_column("CLI Command", style="white")
    features_table.add_column("Description", style="dim")
    
    features_table.add_row(
        "Verbose Output",
        "--verbose",
        "Show detailed document metadata and processing info"
    )
    features_table.add_row(
        "Quiet Mode",
        "--quiet",
        "Suppress decorative output for scripting"
    )
    features_table.add_row(
        "Strategy Selection",
        "--strategy <name>",
        "Use predefined configurations from demo_strategies.yaml"
    )
    features_table.add_row(
        "Custom Config",
        "--config <path>",
        "Use custom configuration file"
    )
    features_table.add_row(
        "Component Override",
        "--embedder <name>",
        "Override specific components"
    )
    features_table.add_row(
        "Top-K Results",
        "--top-k <num>",
        "Control number of search results"
    )
    
    console.print(features_table)
    
    # Summary
    print_section_header("CLI Demo Summary", "🎓")
    
    console.print("🚀 [bold green]Research Paper Analysis via CLI Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ [bold cyan]Complete RAG workflow using CLI commands[/bold cyan]")
    console.print("✅ Strategy-based configuration without code changes")
    console.print("✅ Verbose output for detailed analysis")
    console.print("✅ Quiet mode for script integration")
    console.print("✅ Semantic search with rich output formatting")
    console.print("✅ Collection management and statistics")
    
    console.print(f"\n[bold]CLI Commands Used:[/bold]")
    console.print("📋 `cli.py --strategy-file demos/demo_strategies.yaml ingest --strategy <name> <path>` - Ingest documents")
    console.print("🔍 `cli.py --strategy-file demos/demo_strategies.yaml search --strategy <name> '<query>'` - Search documents")  
    console.print("📊 `cli.py --strategy-file demos/demo_strategies.yaml info --strategy <name>` - Show collection info")
    console.print("🔧 `cli.py --verbose --strategy-file <file> <command>` - Show detailed output")
    console.print("🤫 `cli.py --quiet --strategy-file <file> <command>` - Suppress decorative output")
    
    console.print(f"\n[bold]Platform Benefits Shown:[/bold]")
    console.print("🎯 No need to write custom code for RAG operations")
    console.print("⚡ Quick experimentation with different strategies")
    console.print("🔄 Easy integration into scripts and pipelines")
    console.print("📋 Clean separation of configuration from execution")
    
    console.print(f"\n📁 Research database saved to: [bold]./demos/vectordb/research_papers[/bold]")
    console.print("🔄 You can continue using these CLI commands to query the database:")
    console.print("[dim]$ python cli.py --strategy-file demos/demo_strategies.yaml search --strategy research_papers_demo 'your query here'[/dim]")
    console.print("[dim]$ python cli.py --verbose --strategy-file demos/demo_strategies.yaml search --strategy research_papers_demo 'detailed results'[/dim]")
    
    # Clean up the database after demo
    console.print("\n🧹 [bold cyan]Cleaning up demo database...[/bold cyan]")
    console.print("[dim]💡 Removing demo data to keep your system clean[/dim]")
    
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy research_papers_demo delete --all"
    )
    
    if returncode == 0:
        console.print("✅ [bold green]Demo database cleaned up successfully![/bold green]")
    else:
        console.print("⚠️ [bold yellow]Database cleanup had issues, trying direct cleanup...[/bold yellow]")
        # Fallback to direct cleanup if manage command fails
        import shutil
        from pathlib import Path
        
        db_path = Path("./demos/vectordb/research_papers")
        try:
            if db_path.exists():
                shutil.rmtree(db_path)
                console.print("✅ [bold green]Fallback cleanup successful![/bold green]")
            else:
                console.print("ℹ️ [bold blue]Database directory not found (already clean)[/bold blue]")
        except Exception as e:
            console.print(f"⚠️ [bold yellow]All cleanup methods failed: {e}[/bold yellow]")
            console.print("[dim]💡 You can manually delete ./demos/vectordb/research_papers/ if needed[/dim]")


if __name__ == "__main__":
    try:
        demonstrate_research_paper_rag_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Research demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Research demo failed: {str(e)}", style="red")
        console.print("Make sure you're running from the rag directory")
        sys.exit(1)