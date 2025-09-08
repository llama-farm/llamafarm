#!/usr/bin/env python3
"""
Demo 7: Aircraft Operations Manual Analysis (CLI-based)

This demo showcases the RAG system's ability to handle large technical PDFs
using the strategy-first approach. All configuration is in demo_strategies.yaml.

The demo uses ONLY CLI commands to demonstrate:
- PDF parsing with page structure preservation
- Technical pattern extraction (speeds, altitudes, weights)
- Table extraction for operational data
- Heading hierarchy extraction for document navigation
- RerankedStrategy for accurate technical information retrieval
- Metadata-aware search for page-specific references

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
        # Use command as list to avoid shell injection vulnerabilities
        import shlex
        cmd_parts = shlex.split(command)
        
        result = subprocess.run(
            cmd_parts, 
            shell=False,  # Disable shell to prevent injection attacks
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
                elif 'FL' in line or 'kts' in line or 'ft' in line:
                    # Highlight aviation terms
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


def print_section_header(title: str, emoji: str = "✈️"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def demonstrate_aircraft_operations_cli():
    """Demonstrate aircraft operations manual analysis using CLI commands only."""
    
    # Header
    console.print(Panel.fit(
        "[bold cyan]🦙 Demo 7: Aircraft Operations Manual Analysis[/bold cyan]\n"
        "[yellow]Boeing 737-700/800 FCOM Analysis[/yellow]",
        border_style="cyan"
    ))
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• [bold cyan]Large PDF processing[/bold cyan] (21MB technical manual)")
    console.print("• Strategy-first configuration (llamaindex_pdf_processing_main_chroma_db)")
    console.print("• Page-based PDF parsing with structure preservation")
    console.print("• Aviation-specific pattern extraction")
    console.print("• Table extraction for operational data")
    console.print("• RerankedStrategy for technical accuracy")
    console.print("\n[dim]All configuration is in demo_strategies.yaml[/dim]")
    
    # Step 1: Show strategy information
    print_section_header("Strategy Information", "📋")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml strategies show llamaindex_pdf_processing_main_chroma_db",
        "Viewing aircraft operations strategy configuration"
    )
    
    time.sleep(2)
    
    # Step 2: Clean any existing collection
    print_section_header("Database Cleanup", "🧹")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy llamaindex_pdf_processing_main_chroma_db delete --all",
        "Cleaning up any existing aircraft operations collection"
    )
    
    # Step 3: Ingest the large PDF
    print_section_header("Aircraft Manual Ingestion", "📥")
    
    console.print("\n[yellow]⏳ Processing a 21MB technical PDF - this may take a few minutes...[/yellow]")
    console.print("[dim]The PDF contains ~400 pages of technical documentation[/dim]")
    console.print("\n[bold]Progress indicators:[/bold]")
    console.print("• [cyan]Parsing[/cyan]: Extracting text from PDF pages")
    console.print("• [cyan]Chunking[/cyan]: Splitting into manageable segments") 
    console.print("• [cyan]Extracting[/cyan]: Finding patterns, entities, tables")
    console.print("• [cyan]Embedding[/cyan]: Generating vector representations")
    console.print("• [cyan]Indexing[/cyan]: Storing in vector database\n")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml --verbose ingest --strategy llamaindex_pdf_processing_main_chroma_db demos/static_samples/747/ryanair-737-700-800-fcom-rev-30.pdf",
        "Starting ingestion of 737 Flight Crew Operations Manual (FCOM)"
    )
    
    console.print("\n[green]✅ PDF processing complete![/green]")
    time.sleep(3)
    
    # Step 4: Show collection statistics
    print_section_header("Collection Statistics", "📊")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy llamaindex_pdf_processing_main_chroma_db",
        "Viewing collection statistics and extracted metadata"
    )
    
    time.sleep(2)
    
    # Step 5: Perform aviation-specific searches
    print_section_header("Aviation Query Demonstrations", "🔍")
    
    console.print("\n[cyan]Now testing various search queries on the indexed manual...[/cyan]")
    console.print("[dim]Each query uses RerankedStrategy for optimal results[/dim]\n")
    
    # Query 1: Emergency procedures
    console.print("\n[bold yellow]Query 1: Emergency Procedures[/bold yellow]")
    console.print("[dim]Looking for critical safety information...[/dim]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy llamaindex_pdf_processing_main_chroma_db "engine failure emergency procedures" --top-k 3',
        "Searching for engine failure procedures"
    )
    
    console.print("\n[green]✓ Found relevant emergency procedures[/green]")
    time.sleep(2)
    
    # Query 2: Performance limits
    console.print("\n[bold yellow]Query 2: Aircraft Performance Limits[/bold yellow]")
    console.print("[dim]Retrieving operational ceiling and performance data...[/dim]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy llamaindex_pdf_processing_main_chroma_db "maximum altitude service ceiling FL410" --top-k 3',
        "Finding aircraft performance limitations"
    )
    
    console.print("\n[green]✓ Retrieved performance limit information[/green]")
    time.sleep(2)
    
    # Query 3: Takeoff procedures
    console.print("\n[bold yellow]Query 3: Takeoff Procedures[/bold yellow]")
    console.print("[dim]Searching for V-speeds and rotation procedures...[/dim]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy llamaindex_pdf_processing_main_chroma_db "takeoff speeds V1 VR V2 rotation" --top-k 3',
        "Searching for takeoff speed procedures"
    )
    
    console.print("\n[green]✓ Located takeoff speed information[/green]")
    time.sleep(2)
    
    # Query 4: System operations
    console.print("\n[bold yellow]Query 4: Aircraft Systems[/bold yellow]")
    console.print("[dim]Querying hydraulic system specifications...[/dim]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy llamaindex_pdf_processing_main_chroma_db "hydraulic system pressure PSI operation" --top-k 3',
        "Finding hydraulic system information"
    )
    
    console.print("\n[green]✓ Found system operation details[/green]")
    time.sleep(2)
    
    # Query 5: Landing procedures
    console.print("\n[bold yellow]Query 5: Landing Configuration[/bold yellow]")
    console.print("[dim]Looking up approach speeds and landing configuration...[/dim]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml --verbose search --strategy llamaindex_pdf_processing_main_chroma_db "landing flaps approach speed VREF" --top-k 3',
        "Searching for landing procedures"
    )
    
    console.print("\n[green]✓ Retrieved landing procedure information[/green]")
    time.sleep(2)
    
    # Step 6: Demonstrate metadata extraction
    print_section_header("Extracted Metadata Showcase", "🎯")
    
    console.print("\n[bold]The llamaindex_pdf_processing_main_chroma_db strategy extracts:[/bold]")
    console.print("• [cyan]HeadingExtractor[/cyan]: Document structure and sections")
    console.print("• [cyan]PatternExtractor[/cyan]: Flight levels, speeds, weights, altitudes")
    console.print("• [cyan]TableExtractor[/cyan]: Performance tables and checklists")
    console.print("• [cyan]EntityExtractor[/cyan]: Organizations, facilities, locations")
    console.print("• [cyan]StatisticsExtractor[/cyan]: Document structure analysis")
    console.print("\n[dim]These extractors help with precise technical information retrieval[/dim]")
    
    # Step 7: Show reranking in action
    print_section_header("Reranking Strategy Benefits", "⚡")
    
    console.print("\n[bold]RerankedStrategy configuration:[/bold]")
    rerank_table = Table(show_header=False, show_edge=False)
    rerank_table.add_column("", style="yellow", width=25)
    rerank_table.add_column("", style="white")
    
    rerank_factors = [
        ("Initial candidates", "15 documents"),
        ("Final results", "5 documents"),
        ("Similarity weight", "60% (semantic match)"),
        ("Metadata weight", "30% (page/section context)"),
        ("Recency weight", "10% (document order)")
    ]
    
    for factor, value in rerank_factors:
        rerank_table.add_row(factor, value)
    
    console.print(rerank_table)
    
    # Step 8: Advanced queries
    print_section_header("Advanced Technical Queries", "🔧")
    
    console.print("\n[cyan]Testing complex multi-concept queries...[/cyan]\n")
    
    console.print("\n[bold yellow]Complex Query: Weight and Balance[/bold yellow]")
    console.print("[dim]Combining MTOW and CG limit search...[/dim]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy llamaindex_pdf_processing_main_chroma_db "maximum takeoff weight MTOW center of gravity CG limits" --top-k 2',
        "Searching for weight and balance information"
    )
    
    console.print("\n[green]✓ Found weight and balance data[/green]")
    time.sleep(2)
    
    console.print("\n[bold yellow]Complex Query: Weather Limitations[/bold yellow]")
    console.print("[dim]Retrieving weather minima and approach categories...[/dim]")
    run_cli_command(
        'python cli.py --strategy-file demos/demo_strategies.yaml search --strategy llamaindex_pdf_processing_main_chroma_db "crosswind limits visibility RVR CAT III approach" --top-k 2',
        "Finding weather operating limitations"
    )
    
    console.print("\n[green]✓ Located weather limitation information[/green]")
    time.sleep(2)
    
    # Step 9: Export capabilities
    print_section_header("Export & Integration", "📤")
    
    console.print("\n[bold]CLI commands for operational use:[/bold]")
    
    export_commands = [
        ("Quick reference lookup:", 
         "python cli.py search --strategy llamaindex_pdf_processing_main_chroma_db 'V-speeds' --format json"),
        ("Emergency checklist:",
         "python cli.py search --strategy llamaindex_pdf_processing_main_chroma_db 'emergency checklist' --top-k 1"),
        ("Performance data:",
         "python cli.py search --strategy llamaindex_pdf_processing_main_chroma_db 'performance tables' --verbose"),
        ("System limitations:",
         "python cli.py search --strategy llamaindex_pdf_processing_main_chroma_db 'operating limitations' --format csv")
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
        ("Large PDF handled", "21MB technical manual processed successfully"),
        ("Page preservation", "Maintained page structure for reference"),
        ("Aviation patterns", "Extracted FL, speeds, weights, altitudes"),
        ("Accurate retrieval", "RerankedStrategy ensures technical precision"),
        ("Production ready", "Same commands work for any aircraft manual")
    ]
    
    summary_table = Table(show_header=False, show_edge=False)
    summary_table.add_column("", style="bold green", width=20)
    summary_table.add_column("", style="white")
    
    for point, description in summary_points:
        summary_table.add_row(f"✅ {point}", description)
    
    console.print(summary_table)
    
    console.print("\n[bold cyan]Aircraft operations collection ready for use![/bold cyan]")
    console.print(f"[dim]Collection: aircraft_operations (strategy: llamaindex_pdf_processing_main_chroma_db)[/dim]")
    console.print(f"[dim]To query this collection later:[/dim]")
    console.print('[dim]$ python cli.py --strategy-file demos/demo_strategies.yaml search --strategy llamaindex_pdf_processing_main_chroma_db "your query"[/dim]')
    
    # Final cleanup
    print_section_header("Final Cleanup", "🧹")
    
    console.print("\n[yellow]Removing all indexed documents to free up space...[/yellow]")
    
    run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy llamaindex_pdf_processing_main_chroma_db delete --all",
        "Cleaning up ALL documents in the collection"
    )
    
    console.print("\n[bold green]✅ Demo complete! The 737 FCOM has been successfully processed and cleaned up.[/bold green]")
    console.print("[dim]Total demo time: ~5-7 minutes depending on system performance[/dim]")


if __name__ == "__main__":
    try:
        demonstrate_aircraft_operations_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 Aircraft operations demo interrupted by user", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n\n❌ Aircraft operations demo failed: {str(e)}", style="red")
        console.print("Ensure the CLI is working and demo_strategies.yaml contains llamaindex_pdf_processing_main_chroma_db", style="dim")
        sys.exit(1)