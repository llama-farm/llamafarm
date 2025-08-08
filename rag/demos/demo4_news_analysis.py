#!/usr/bin/env python3
"""
Demo 4: News Article Analysis System (CLI-based)

This demo showcases the RAG system's news analysis capabilities using the
strategy-first approach. All configuration is in demo_strategies.yaml.

The demo uses ONLY CLI commands to demonstrate:
- HTML parsing for web-based news articles
- Entity extraction for people, organizations, locations, events
- Date/time extraction for timeline analysis
- Summary extraction for article abstracts
- Sentiment analysis and categorization
- Advanced search with metadata filtering

NO LOGIC IN THIS FILE - just CLI commands!
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

def run_cli_command(command: str, description: str = None) -> tuple[bool, str]:
    """Run a CLI command and return success status and output."""
    if description:
        console.print(f"\n[bold cyan]→ {description}[/bold cyan]")
    
    console.print(f"[dim]$ {command}[/dim]")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            cwd=Path(__file__).parent.parent
        )
        
        # Print output with formatting
        if result.stdout:
            for line in result.stdout.split('\n'):
                if '✅' in line:
                    console.print(f"[green]{line}[/green]")
                elif '❌' in line or 'ERROR' in line:
                    console.print(f"[red]{line}[/red]")
                elif '🔍' in line or '📊' in line or '💾' in line:
                    console.print(f"[yellow]{line}[/yellow]")
                elif 'Strategy:' in line or 'Using strategy:' in line:
                    console.print(f"[bold magenta]{line}[/bold magenta]")
                else:
                    console.print(line)
        
        # Only show stderr if it contains actual errors (not progress bars)
        if result.stderr:
            # Filter out progress bar output and other benign stderr output
            stderr_lines = result.stderr.strip().split('\n')
            error_lines = []
            for line in stderr_lines:
                # Skip progress bars, warnings, and empty lines
                if (line and 
                    'WARNING' not in line and
                    '%' not in line and  # Progress indicators
                    '█' not in line and  # Progress bars
                    'Processing' not in line and
                    'Extracting' not in line and
                    'Embedding' not in line and
                    'Adding' not in line):
                    error_lines.append(line)
            
            if error_lines:
                console.print(f"[red]Error: {' '.join(error_lines)}[/red]")
            
        return result.returncode == 0, result.stdout
    except Exception as e:
        console.print(f"[red]Command failed: {e}[/red]")
        return False, str(e)


def print_section_header(title: str, emoji: str = "📰"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def demonstrate_news_analysis_cli():
    """Demonstrate news analysis using CLI commands only."""
    
    # Header
    console.print(Panel.fit(
        "[bold cyan]🦙 Demo 4: News Article Analysis System[/bold cyan]\n"
        "[yellow]Strategy-based CLI demonstration[/yellow]",
        border_style="cyan"
    ))
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• [bold cyan]100% CLI-based operation[/bold cyan] - no hardcoded logic!")
    console.print("• Strategy-first configuration (news_analysis_demo)")
    console.print("• HTML news article parsing")
    console.print("• Entity and date extraction")
    console.print("• Sentiment analysis")
    console.print("• Timeline-aware search")
    console.print("\n[dim]All configuration is in demo_strategies.yaml[/dim]")
    
    # Step 1: Show strategy information
    print_section_header("Strategy Information", "📋")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml strategies show news_analysis_demo",
        "Viewing news analysis strategy configuration"
    )
    
    time.sleep(2)
    
    # Step 2: Clean any existing collection
    print_section_header("Database Cleanup", "🧹")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy news_analysis_demo delete --all",
        "Cleaning up any existing news collection (dry run first)"
    )
    
    # Step 3: Ingest news articles
    print_section_header("News Article Ingestion", "📥")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml ingest --strategy news_analysis_demo demos/static_samples/news_articles/",
        "Ingesting news articles with strategy-configured components"
    )
    
    time.sleep(2)
    
    # Step 4: Show collection statistics
    print_section_header("Collection Statistics", "📊")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy news_analysis_demo",
        "Viewing collection statistics"
    )
    
    # Step 5: Perform news-specific searches
    print_section_header("News Query Demonstrations", "🔍")
    
    # Query 1: Breaking news
    console.print("\n[bold yellow]Query 1: Breaking News Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo "breaking news technology AI"',
        "Searching for breaking technology news"
    )
    
    time.sleep(2)
    
    # Query 2: Entity search
    console.print("\n[bold yellow]Query 2: Entity-based Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo "major tech companies announcements"',
        "Finding news about tech companies"
    )
    
    time.sleep(2)
    
    # Query 3: Event search
    console.print("\n[bold yellow]Query 3: Event Timeline Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo "recent product launches events"',
        "Searching for product launch events"
    )
    
    time.sleep(2)
    
    # Query 4: Topic search
    console.print("\n[bold yellow]Query 4: Topic Analysis Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo "artificial intelligence regulation policy"',
        "Finding AI regulation news"
    )
    
    time.sleep(2)
    
    # Query 5: Sentiment search
    console.print("\n[bold yellow]Query 5: Market Sentiment Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo "stock market trends analysis"',
        "Analyzing market sentiment"
    )
    
    # Step 6: Advanced filtering (if metadata filtering is configured)
    print_section_header("Advanced Filtering", "🎯")
    
    console.print("[dim]Note: The RerankedStrategy in news_analysis_demo prioritizes:")
    console.print("  • Similarity to query (50% weight)")
    console.print("  • Recency of articles (30% weight)")
    console.print("[dim]  • Metadata relevance (20% weight)[/dim]")
    
    # Step 7: Export capabilities
    print_section_header("Export & Integration", "📤")
    
    console.print("\n[bold]CLI commands for production use:[/bold]")
    
    export_commands = [
        ("Export search results as JSON:", 
         "python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo 'AI news' --format json > results.json"),
        ("Batch process multiple queries:",
         "python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo --batch queries.txt"),
        ("Monitor collection growth:",
         "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy news_analysis_demo stats"),
        ("API endpoint for integration:",
         "python cli.py --strategy-file demos/demo_strategies.yaml serve --strategy news_analysis_demo --port 8000")
    ]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Use Case", style="cyan")
    table.add_column("Command", style="white")
    
    for use_case, command in export_commands:
        table.add_row(use_case, command)
    
    console.print(table)
    
    # Summary
    print_section_header("Demo Summary", "🎓")
    
    summary_points = [
        ("Strategy-based", "All configuration in demo_strategies.yaml"),
        ("CLI-driven", "No hardcoded logic in demo file"),
        ("Production-ready", "Same commands work in production"),
        ("Flexible", "Easy to modify strategy without code changes"),
        ("Integrated", "Works with existing CLI tools and workflows")
    ]
    
    summary_table = Table(show_header=False, show_edge=False)
    summary_table.add_column("", style="bold green", width=20)
    summary_table.add_column("", style="white")
    
    for point, description in summary_points:
        summary_table.add_row(f"✅ {point}", description)
    
    console.print(summary_table)
    
    console.print("\n[bold cyan]News collection ready for production use![/bold cyan]")
    console.print(f"[dim]Collection: news_analysis (strategy: news_analysis_demo)[/dim]")
    console.print(f"[dim]To query this collection later:[/dim]")
    console.print('[dim]$ python cli.py --strategy-file demos/demo_strategies.yaml search --strategy news_analysis_demo "your query"[/dim]')
    
    # Final cleanup
    print_section_header("Final Cleanup", "🧹")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy news_analysis_demo delete --all",
        "Cleaning up ALL documents in the collection"
    )


if __name__ == "__main__":
    try:
        demonstrate_news_analysis_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 News analysis demo interrupted by user", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n\n❌ News analysis demo failed: {str(e)}", style="red")
        console.print("Ensure the CLI is working and demo_strategies.yaml contains news_analysis_demo", style="dim")
        sys.exit(1)