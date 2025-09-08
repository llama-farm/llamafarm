#!/usr/bin/env python3
"""
Demo 3: Code Documentation Analysis System (CLI Version)
Demonstrates code documentation processing using CLI commands exclusively.
Shows how the platform handles technical documentation through CLI.
"""

import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

# Setup rich console for beautiful output
console = Console()


def run_cli_command(command: str, verbose: bool = False, quiet: bool = False, show_output: bool = True) -> tuple[int, str, str]:
    """
    Run a CLI command and return the result.
    
    Args:
        command: The CLI command to run
        verbose: Whether to add --verbose flag
        quiet: Whether to add --quiet flag
        show_output: Whether to display the command being run
    
    Returns:
        Tuple of (return_code, stdout, stderr)
    """
    # Add flags if requested
    parts = command.split()
    cli_index = next((i for i, part in enumerate(parts) if 'cli.py' in part), -1)
    
    if cli_index >= 0:
        flags_to_add = []
        if verbose and '--verbose' not in command:
            flags_to_add.append('--verbose')
        if quiet and '--quiet' not in command:
            flags_to_add.append('--quiet')
        
        for flag in flags_to_add:
            parts.insert(cli_index + 1, flag)
        
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


def print_section_header(title: str, emoji: str = "📚"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def demonstrate_code_documentation_cli():
    """Demonstrate code documentation system using CLI commands exclusively."""
    
    print_section_header("🦙 Demo 3: Code Documentation System (CLI Version)", "📚")
    
    console.print("\n[bold green]This demo showcases code documentation capabilities via CLI:[/bold green]")
    console.print("• [bold cyan]Technical documentation processing through CLI[/bold cyan]")
    console.print("• Markdown parsing with structure preservation")
    console.print("• Code block and API reference extraction")
    console.print("• Technical term recognition")
    console.print("• Semantic search for developer queries")
    
    # Verify system
    print_section_header("System Check", "🔧")
    
    console.print("🔍 [bold cyan]Checking system readiness...[/bold cyan]")
    
    # Test CLI is available
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --help",
        show_output=False
    )
    
    if returncode != 0:
        console.print("❌ [red]CLI not available. Check your environment.[/red]")
        return
    
    console.print("✅ [bold green]CLI system ready![/bold green]")
    
    # Initialize code documentation collection
    print_section_header("Initialize Documentation Database", "🗄️")
    
    console.print("🚀 [bold cyan]Setting up code documentation database...[/bold cyan]")
    
    # Check collection status
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy markdown_processing_main_chroma_db",
        quiet=True
    )
    
    if "document_count" in stdout:
        doc_count = "0"
        for line in stdout.split('\n'):
            if "document_count" in line:
                doc_count = line.split(':')[1].strip()
                break
        console.print(f"[dim]💡 Collection exists with {doc_count} documents[/dim]")
    
    # Ingest documentation files
    print_section_header("Documentation Ingestion", "📥")
    
    console.print("[bold cyan]🔄 Processing technical documentation files...[/bold cyan]")
    console.print("[dim]💡 Using markdown_processing_main_chroma_db strategy for optimal parsing[/dim]")
    
    # Show what we're ingesting
    docs_table = Table(show_header=True, header_style="bold yellow")
    docs_table.add_column("Document", style="cyan")
    docs_table.add_column("Type", style="white")
    docs_table.add_column("Purpose", style="dim")
    
    docs_table.add_row(
        "api_reference.md",
        "Markdown",
        "API endpoints and parameters"
    )
    docs_table.add_row(
        "implementation_guide.md",
        "Markdown",
        "Code examples and patterns"
    )
    docs_table.add_row(
        "best_practices.md",
        "Markdown",
        "Coding standards and guidelines"
    )
    
    console.print(docs_table)
    
    # Ingest with verbose output
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml --verbose ingest --strategy markdown_processing_main_chroma_db demos/static_samples/code_documentation/"
    )
    
    if returncode == 0:
        console.print("✅ [bold green]Documentation successfully processed![/bold green]")
        
        # Show processing details
        for line in stdout.split('\n'):
            if any(k in line for k in ["Documents processed:", "chunks", "sections"]):
                console.print(f"   {line.strip()}")
    else:
        console.print(f"⚠️ [yellow]Ingestion completed with warnings[/yellow]")
    
    # Show database statistics
    print_section_header("Documentation Database Stats", "📊")
    
    console.print("📈 [bold cyan]Analyzing documentation database...[/bold cyan]")
    
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy markdown_processing_main_chroma_db"
    )
    
    if returncode == 0:
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        for line in stdout.split('\n'):
            if ':' in line and not line.startswith('python'):
                key, value = line.split(':', 1)
                stats_table.add_row(key.strip(), value.strip())
        
        console.print(stats_table)
    
    # Demonstrate developer queries
    print_section_header("Developer Query Examples", "🔍")
    
    developer_queries = [
        "How do I authenticate API requests?",
        "What are the rate limiting best practices?",
        "Show me examples of error handling patterns",
        "What's the recommended project structure?",
        "How to implement caching in the application?"
    ]
    
    console.print("🎯 [bold cyan]Testing common developer queries:[/bold cyan]")
    console.print("[dim]💡 Each query searches across all documentation[/dim]")
    
    for i, query in enumerate(developer_queries, 1):
        console.print(f"\n[bold cyan]Developer Query #{i}:[/bold cyan]")
        console.print(f"Question: [yellow]\"{query}\"[/yellow]")
        
        # Use verbose for first 2 queries, normal for others
        verbose = (i <= 2)
        
        returncode, stdout, stderr = run_cli_command(
            f'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy markdown_processing_main_chroma_db "{query}" --top-k 2',
            verbose=verbose
        )
        
        if returncode == 0:
            if verbose:
                # Show full output for verbose queries
                console.print(stdout)
            else:
                # Show summary for non-verbose
                console.print("[dim]Found relevant documentation sections[/dim]")
                for line in stdout.split('\n'):
                    if any(k in line for k in ["Source:", "Content Preview:", "Similarity:"]):
                        console.print(f"   {line.strip()}")
        
        if i < len(developer_queries):
            time.sleep(1)
    
    # Demonstrate code-specific searches
    print_section_header("Code-Specific Searches", "💻")
    
    console.print("🎯 [bold cyan]Testing code pattern searches:[/bold cyan]")
    
    code_searches = [
        ("API Endpoints", "GET POST endpoints parameters"),
        ("Error Codes", "error codes status HTTP responses"),
        ("Configuration", "config settings environment variables")
    ]
    
    for search_type, query in code_searches:
        console.print(f"\n[bold]{search_type} Search:[/bold]")
        
        returncode, stdout, stderr = run_cli_command(
            f'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy markdown_processing_main_chroma_db "{query}" --top-k 1',
            quiet=False
        )
        
        if returncode == 0:
            # Extract key information
            found_docs = False
            for line in stdout.split('\n'):
                if "Result #" in line or "Source:" in line:
                    found_docs = True
                    console.print(f"   ✓ {line.strip()}")
            
            if found_docs:
                console.print(f"   [green]Found relevant {search_type.lower()} documentation[/green]")
    
    # Show advanced features
    print_section_header("Advanced CLI Features for Docs", "🚀")
    
    console.print("📋 [bold cyan]Documentation-specific CLI capabilities:[/bold cyan]")
    
    features_table = Table(show_header=True, header_style="bold cyan")
    features_table.add_column("Feature", style="yellow")
    features_table.add_column("Command Example", style="white")
    features_table.add_column("Use Case", style="dim")
    
    features_table.add_row(
        "Batch Ingestion",
        "cli.py ingest docs/*.md",
        "Process multiple doc files"
    )
    features_table.add_row(
        "Verbose Analysis",
        "cli.py --verbose search 'API'",
        "See full context and metadata"
    )
    features_table.add_row(
        "Quiet Integration",
        "cli.py --quiet search 'error'",
        "For IDE plugins/tools"
    )
    features_table.add_row(
        "Top-K Control",
        "cli.py search 'auth' --top-k 10",
        "Get more results"
    )
    features_table.add_row(
        "Strategy Override",
        "cli.py --strategy-file demos/demo_strategies.yaml --strategy custom search",
        "Use different configurations"
    )
    
    console.print(features_table)
    
    # Integration examples
    print_section_header("IDE Integration Examples", "🔌")
    
    console.print("🤖 [bold cyan]How to integrate with development tools:[/bold cyan]")
    
    integration_panel = Panel(
        """[bold]VS Code Extension:[/bold]
result = subprocess.run(
    'python cli.py --quiet search "' + selected_text + '"',
    capture_output=True
)

[bold]Vim Plugin:[/bold]
:!python cli.py --strategy-file demos/demo_strategies.yaml search "<cword>" --strategy markdown_processing_main_chroma_db

[bold]Shell Function:[/bold]
docsearch() {
    python cli.py --strategy-file demos/demo_strategies.yaml search "$1" --strategy markdown_processing_main_chroma_db
}

[bold]Git Hook:[/bold]
# In .git/hooks/pre-commit
python cli.py --quiet search "TODO FIXME" | grep -q "Result" && exit 1
""",
        title="Integration Code Examples",
        border_style="blue"
    )
    
    console.print(integration_panel)
    
    # Summary
    print_section_header("Code Documentation CLI Summary", "🎯")
    
    console.print("🚀 [bold green]Code Documentation System via CLI Complete![/bold green]")
    
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ [bold cyan]Complete documentation workflow using CLI[/bold cyan]")
    console.print("✅ Markdown parsing with structure preservation")
    console.print("✅ Code-aware search capabilities")
    console.print("✅ Developer-focused query handling")
    console.print("✅ Integration-ready quiet mode")
    console.print("✅ Verbose mode for detailed analysis")
    
    console.print(f"\n[bold]Developer Benefits:[/bold]")
    console.print("🔍 Instant documentation search from terminal")
    console.print("📚 Unified access to all documentation")
    console.print("💻 Easy integration with development tools")
    console.print("🤖 Scriptable for automation")
    console.print("📈 Searchable documentation metrics")
    
    console.print(f"\n[bold]CLI Commands for Developers:[/bold]")
    console.print("📥 `cli.py --strategy-file demos/demo_strategies.yaml --strategy markdown_processing_main_chroma_db ingest docs/`")
    console.print("🔍 `cli.py --strategy-file demos/demo_strategies.yaml --strategy markdown_processing_main_chroma_db search '<query>'`")
    console.print("📊 `cli.py --strategy-file demos/demo_strategies.yaml --strategy markdown_processing_main_chroma_db info`")
    console.print("🤫 `cli.py --quiet` for tool integration")
    console.print("📝 `cli.py --verbose` for detailed results")
    
    console.print(f"\n📁 Documentation database: [bold]./demos/vectordb/code_documentation[/bold]")
    console.print("🔄 Continue using these commands in your development workflow:")
    console.print("[dim]$ python cli.py --strategy-file demos/demo_strategies.yaml search 'your question' --strategy markdown_processing_main_chroma_db[/dim]")
    console.print("[dim]$ alias docsearch='python cli.py --strategy-file demos/demo_strategies.yaml search --strategy markdown_processing_main_chroma_db'[/dim]")
    
    # Clean up the database after demo
    console.print("\n🧹 [bold cyan]Cleaning up demo database...[/bold cyan]")
    console.print("[dim]💡 Removing demo data to keep your system clean[/dim]")
    
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy markdown_processing_main_chroma_db delete --all"
    )
    
    if returncode == 0:
        console.print("✅ [bold green]Demo database cleaned up successfully![/bold green]")
    else:
        console.print("⚠️ [bold yellow]Database cleanup had issues, trying direct cleanup...[/bold yellow]")
        # Fallback to direct cleanup if manage command fails
        import shutil
        from pathlib import Path
        
        db_path = Path("./demos/vectordb/code_documentation")
        try:
            if db_path.exists():
                shutil.rmtree(db_path)
                console.print("✅ [bold green]Fallback cleanup successful![/bold green]")
            else:
                console.print("ℹ️ [bold blue]Database directory not found (already clean)[/bold blue]")
        except Exception as e:
            console.print(f"⚠️ [bold yellow]All cleanup methods failed: {e}[/bold yellow]")
            console.print("[dim]💡 You can manually delete ./demos/vectordb/code_documentation/ if needed[/dim]")


if __name__ == "__main__":
    try:
        demonstrate_code_documentation_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Documentation demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Documentation demo failed: {str(e)}", style="red")
        console.print("Make sure you're running from the rag directory")
        sys.exit(1)