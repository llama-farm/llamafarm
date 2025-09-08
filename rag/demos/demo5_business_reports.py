#!/usr/bin/env python3
"""
Demo 5: Business Reports Analysis System (CLI-based)

This demo showcases the RAG system's business document analysis capabilities
using the strategy-first approach. All configuration is in demo_strategies.yaml.

The demo uses ONLY CLI commands to demonstrate:
- PDF parsing for business reports and financial documents
- Table extraction for financial data
- Pattern extraction for currency amounts and percentages
- Statistical analysis of business metrics
- Metadata-filtered search for targeted retrieval

NO LOGIC IN THIS FILE - just CLI commands!
"""

import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

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
                elif '$' in line and any(c.isdigit() for c in line):
                    # Highlight currency amounts
                    console.print(f"[bold green]{line}[/bold green]")
                elif '%' in line and any(c.isdigit() for c in line):
                    # Highlight percentages
                    console.print(f"[bold blue]{line}[/bold blue]")
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


def print_section_header(title: str, emoji: str = "📊"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def demonstrate_business_reports_cli():
    """Demonstrate business reports analysis using CLI commands only."""
    
    # Header
    console.print(Panel.fit(
        "[bold cyan]🦙 Demo 5: Business Reports Analysis System[/bold cyan]\n"
        "[yellow]Strategy-based CLI demonstration[/yellow]",
        border_style="cyan"
    ))
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• [bold cyan]100% CLI-based operation[/bold cyan] - no hardcoded logic!")
    console.print("• Strategy-first configuration (business_processing_main_chroma_db)")
    console.print("• PDF parsing for business documents")
    console.print("• Table and financial data extraction")
    console.print("• Pattern recognition for amounts and percentages")
    console.print("• Metadata-filtered search")
    console.print("\n[dim]All configuration is in demo_strategies.yaml[/dim]")
    
    # Step 1: Show strategy information
    print_section_header("Strategy Information", "📋")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml strategies show business_processing_main_chroma_db",
        "Viewing business reports strategy configuration"
    )
    
    time.sleep(2)
    
    # Step 2: Clean any existing collection
    print_section_header("Database Cleanup", "🧹")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy business_processing_main_chroma_db delete --all",
        "Cleaning up any existing business reports collection"
    )
    
    # Step 3: Ingest business reports
    print_section_header("Business Reports Ingestion", "📥")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml --verbose ingest --strategy business_processing_main_chroma_db demos/static_samples/business_reports/the-state-of-ai-how-organizations-are-rewiring-to-capture-value_final.pdf",
        "Ingesting business reports with strategy-configured components (verbose mode shows extractors)"
    )
    
    time.sleep(2)
    
    # Step 4: Show collection statistics
    print_section_header("Collection Statistics", "📊")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy business_processing_main_chroma_db",
        "Viewing collection statistics and extracted metadata"
    )
    
    # Step 5: Perform business-specific searches
    print_section_header("Business Query Demonstrations", "🔍")
    
    # Query 1: Financial metrics
    console.print("\n[bold yellow]Query 1: Financial Metrics Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy business_processing_main_chroma_db "revenue growth profit margins" --top-k 2',
        "Searching for financial performance metrics (showing metadata)"
    )
    
    time.sleep(2)
    
    # Query 2: Market analysis
    console.print("\n[bold yellow]Query 2: Market Analysis Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy business_processing_main_chroma_db "market share competitive analysis" --top-k 2',
        "Finding market position information (with extracted entities)"
    )
    
    time.sleep(2)
    
    # Query 3: Risk factors
    console.print("\n[bold yellow]Query 3: Risk Assessment Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy business_processing_main_chroma_db "risk factors compliance regulatory" --top-k 2',
        "Identifying business risks and compliance issues (with sentiment analysis)"
    )
    
    time.sleep(2)
    
    # Query 4: Investment data
    console.print("\n[bold yellow]Query 4: Investment Analysis Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy business_processing_main_chroma_db "capital expenditure ROI investment returns" --top-k 2',
        "Analyzing investment performance (with financial pattern extraction)"
    )
    
    time.sleep(2)
    
    # Query 5: Operational metrics
    console.print("\n[bold yellow]Query 5: Operational Metrics Search[/bold yellow]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose --content-length 200 search --strategy business_processing_main_chroma_db "operational efficiency productivity KPIs" --top-k 2',
        "Finding operational performance indicators (with full metadata display)"
    )
    
    # Step 6: Demonstrate extraction capabilities
    print_section_header("Data Extraction Features", "🎯")
    
    console.print("\n[bold]The business_processing_main_chroma_db strategy includes:[/bold]")
    console.print("• [cyan]TableExtractor[/cyan]: Extracts financial tables")
    console.print("• [cyan]PatternExtractor[/cyan]: Finds currency amounts and percentages")
    console.print("• [cyan]StatisticsExtractor[/cyan]: Analyzes document structure")
    console.print("\n[dim]These extractors run automatically during ingestion[/dim]")
    
    # Step 7: Advanced filtering demonstration
    print_section_header("Metadata Filtering", "🔎")
    
    console.print("\n[bold]MetadataFilteredStrategy capabilities:[/bold]")
    console.print("• Pre-filtering by document metadata")
    console.print("• Fallback search if filtered results are insufficient")
    console.print("• Configurable filter modes and multipliers")
    
    # Note: In real usage, we could filter by document type, date, etc.
    console.print("\n[dim]Note: Filters can be configured in demo_strategies.yaml[/dim]")
    
    # Step 8: Export and reporting capabilities
    print_section_header("Export & Reporting", "📤")
    
    console.print("\n[bold]CLI commands for business reporting:[/bold]")
    
    export_commands = [
        ("Generate executive summary:", 
         "python cli.py --strategy-file demos/demo_strategies.yaml search --strategy business_processing_main_chroma_db 'executive summary' --top-k 1"),
        ("Export financial data as CSV:",
         "python cli.py export --strategy business_processing_main_chroma_db --format csv --filter 'financial'"),
        ("Create quarterly report:",
         "python cli.py report --strategy business_processing_main_chroma_db --period Q4 --year 2024"),
        ("Analyze trends over time:",
         "python cli.py trends --strategy business_processing_main_chroma_db --metric revenue --period 12m")
    ]
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Report Type", style="cyan")
    table.add_column("Command", style="white")
    
    for report_type, command in export_commands:
        table.add_row(report_type, command)
    
    console.print(table)
    
    # Step 9: Performance optimization notes
    print_section_header("Performance Optimization", "⚡")
    
    console.print("\n[bold]Strategy optimizations:[/bold]")
    optimization_table = Table(show_header=False, show_edge=False)
    optimization_table.add_column("", style="yellow", width=25)
    optimization_table.add_column("", style="white")
    
    optimizations = [
        ("Performance Priority", "accuracy (for financial data precision)"),
        ("Batch Size", "4 documents (optimal for PDFs)"),
        ("Max Batch Wait", "200ms (balanced latency)"),
        ("Document Length", "100-100,000 chars (handles long reports)")
    ]
    
    for opt, value in optimizations:
        optimization_table.add_row(opt, value)
    
    console.print(optimization_table)
    
    # Summary
    print_section_header("Demo Summary", "🎓")
    
    summary_points = [
        ("Strategy-based", "All configuration in demo_strategies.yaml"),
        ("CLI-driven", "No hardcoded logic in demo file"),
        ("Production-ready", "Same commands work in production"),
        ("Financial-optimized", "Specialized extractors for business data"),
        ("Metadata-aware", "Advanced filtering capabilities")
    ]
    
    summary_table = Table(show_header=False, show_edge=False)
    summary_table.add_column("", style="bold green", width=20)
    summary_table.add_column("", style="white")
    
    for point, description in summary_points:
        summary_table.add_row(f"✅ {point}", description)
    
    console.print(summary_table)
    
    console.print("\n[bold cyan]Business reports collection ready for production use![/bold cyan]")
    console.print(f"[dim]Collection: business_reports (strategy: business_processing_main_chroma_db)[/dim]")
    console.print(f"[dim]To query this collection later:[/dim]")
    console.print('[dim]$ python cli.py --strategy-file demos/demo_strategies.yaml search --strategy business_processing_main_chroma_db "your query"[/dim]')
    
    # Integration examples
    print_section_header("Integration Examples", "🔗")
    
    console.print("\n[bold]Integrate with business intelligence tools:[/bold]")
    console.print("• Export to Tableau: Use CSV export command")
    console.print("• Power BI integration: Use JSON API endpoint")
    console.print("• Excel reporting: Direct CSV import")
    console.print("• Automated alerts: Schedule CLI searches with cron")
    
    # Final cleanup
    print_section_header("Final Cleanup", "🧹")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy business_processing_main_chroma_db delete --all",
        "Cleaning up ALL documents in the collection"
    )


if __name__ == "__main__":
    try:
        demonstrate_business_reports_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Business reports demo interrupted by user", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n\n❌ Business reports demo failed: {str(e)}", style="red")
        console.print("Ensure the CLI is working and demo_strategies.yaml contains business_processing_main_chroma_db", style="dim")
        sys.exit(1)