#!/usr/bin/env python3
"""
Demo 7: Air Force Instruction (AFI) Document Analysis System (CLI Version)
Demonstrates advanced RAG capabilities for military technical documentation using the CLI.
Showcases smart PDF parsing, page-level extraction, and compliance-focused search.
"""

import subprocess
import sys
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn

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


def print_section_header(title: str, emoji: str = "✈️"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def demonstrate_afi_document_rag_cli():
    """Demonstrate AFI document processing using CLI commands exclusively."""
    
    print_section_header("🦙 Demo 7: Air Force Instruction Document Analysis (CLI Version)", "✈️")
    
    console.print("\n[bold green]This demo showcases advanced technical document processing:[/bold green]")
    console.print("• [bold cyan]Smart PDF parsing with page-level extraction[/bold cyan]")
    console.print("• Military-specific pattern recognition (paragraph refs, forms, T.O.s)")
    console.print("• Compliance and requirement extraction (shall/must/will statements)")
    console.print("• Hybrid retrieval strategy for precise technical search")
    console.print("• Page number and exact location tracking for citations")
    
    # Test CLI availability
    print_section_header("CLI System Check", "🔧")
    
    console.print("🔍 [bold cyan]Checking CLI availability...[/bold cyan]")
    
    returncode, stdout, stderr = run_cli_command("python cli.py --help", show_output=False)
    if returncode != 0:
        console.print("❌ [red]CLI not available. Make sure you're in the rag directory.[/red]")
        return
    
    console.print("✅ [bold green]CLI is available and ready![/bold green]")
    
    # Check if the AFI document exists
    print_section_header("Document Verification", "📄")
    
    afi_path = Path("demos/static_samples/dafi21-101.pdf")
    if not afi_path.exists():
        console.print("❌ [red]AFI document not found at demos/static_samples/dafi21-101.pdf[/red]")
        return
    
    console.print(f"✅ [bold green]Found AFI document: DAFI 21-101[/bold green]")
    console.print(f"   [dim]File size: {afi_path.stat().st_size / 1024 / 1024:.2f} MB[/dim]")
    
    # Initialize the collection for AFI documents
    print_section_header("Initialize AFI Documents Collection", "🗄️")
    
    console.print("🚀 [bold cyan]Initializing AFI documents collection...[/bold cyan]")
    
    # First, check if the collection already exists
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy afi_document_demo"
    )
    
    if "document_count" in stdout:
        console.print("[dim]💡 Collection already exists, continuing...[/dim]")
    else:
        console.print("[dim]💡 Creating new collection for AFI documents...[/dim]")
    
    # Ingest the AFI document using the strategy
    print_section_header("AFI Document Ingestion", "📚")
    
    console.print("[bold cyan]🔄 Ingesting DAFI 21-101 with advanced extraction...[/bold cyan]")
    console.print("[dim]💡 Configuration includes:[/dim]")
    console.print("[dim]   • Page-level chunking with page number extraction[/dim]")
    console.print("[dim]   • AFI paragraph reference extraction (e.g., 3.2.1.4)[/dim]")
    console.print("[dim]   • Compliance requirement extraction (shall/must/will)[/dim]")
    console.print("[dim]   • Warning/Caution/Note statement extraction[/dim]")
    console.print("[dim]   • Military form and T.O. reference extraction[/dim]")
    
    # Show progress during ingestion
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing DAFI 21-101...", total=None)
        
        # Run ingestion with verbose output
        returncode, stdout, stderr = run_cli_command(
            "python cli.py --verbose --strategy-file demos/demo_strategies.yaml ingest --strategy afi_document_demo demos/static_samples/dafi21-101.pdf",
            show_output=False
        )
        
        progress.update(task, completed=True)
    
    if returncode == 0:
        console.print("✅ [bold green]AFI document successfully ingested![/bold green]")
        
        # Parse output to show statistics if available
        if "Documents processed:" in stdout:
            for line in stdout.split('\n'):
                if any(keyword in line for keyword in ["Documents processed:", "Chunks created:", "Extraction complete:"]):
                    console.print(f"   {line.strip()}")
    else:
        console.print(f"⚠️ [yellow]Ingestion completed with warnings[/yellow]")
        if stderr:
            console.print(f"[dim]{stderr[:500]}[/dim]")
    
    # Show collection information and extraction statistics
    print_section_header("Collection & Extraction Statistics", "📊")
    
    console.print("📈 [bold cyan]Retrieving collection statistics and extraction results...[/bold cyan]")
    
    returncode, stdout, stderr = run_cli_command(
        "python cli.py --strategy-file demos/demo_strategies.yaml info --strategy afi_document_demo"
    )
    
    if returncode == 0 and stdout:
        # Parse and display the info nicely
        stats_table = Table(show_header=True, header_style="bold magenta", title="AFI Document Collection")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        for line in stdout.split('\n'):
            if ':' in line and not line.startswith('python'):
                key, value = line.split(':', 1)
                stats_table.add_row(key.strip(), value.strip())
        
        console.print(stats_table)
    
    # Demonstrate AFI-specific queries using CLI search
    print_section_header("AFI-Specific Query Demonstration", "🔍")
    
    afi_queries = [
        "What are the maintenance officer responsibilities?",
        "What forms are required for aircraft maintenance documentation?",
        "What are the warning statements for safety procedures?",
        "Find requirements using 'shall' for compliance",
        "What are the procedures in section 3.2?",
        "Technical Order references for maintenance"
    ]
    
    console.print("🎯 [bold cyan]Running AFI-focused queries with page-level precision:[/bold cyan]")
    console.print("[dim]💡 Using --verbose flag to show page numbers and metadata[/dim]")
    
    for i, query in enumerate(afi_queries, 1):
        console.print(f"\n[bold cyan]AFI Query #{i}:[/bold cyan]")
        console.print(f"Query: [yellow]{query}[/yellow]")
        
        # Run search with verbose output to show page numbers and metadata
        # Use --content-length 800 to show more context (dozens of lines)
        returncode, stdout, stderr = run_cli_command(
            f'python cli.py --verbose --content-length 800 --strategy-file demos/demo_strategies.yaml search --strategy afi_document_demo "{query}"'
        )
        
        if returncode == 0:
            # Parse and highlight important metadata
            lines = stdout.split('\n')
            for line in lines:
                if 'Page' in line or 'page_number' in line:
                    console.print(f"[bold yellow]{line}[/bold yellow]")
                elif 'paragraph_reference' in line or any(ref in line for ref in ['3.', '4.', '5.']):
                    console.print(f"[bold cyan]{line}[/bold cyan]")
                elif any(keyword in line.lower() for keyword in ['shall', 'must', 'will', 'warning', 'caution']):
                    console.print(f"[bold red]{line}[/bold red]")
                else:
                    console.print(line)
        else:
            console.print(f"⚠️ [yellow]Search completed with warnings[/yellow]")
        
        if i < len(afi_queries):
            time.sleep(1.5)  # Pause for readability
    
    # Demonstrate metadata filtering for precise location
    print_section_header("Metadata-Based Precision Search", "🎯")
    
    console.print("🔍 [bold cyan]Demonstrating metadata filtering for exact location retrieval...[/bold cyan]")
    console.print("[dim]💡 The hybrid strategy uses metadata to find specific pages/sections[/dim]")
    
    # Search for specific page or section references
    # Use --content-length 1000 for maximum context
    returncode, stdout, stderr = run_cli_command(
        'python cli.py --verbose --content-length 1000 --strategy-file demos/demo_strategies.yaml search --strategy afi_document_demo "maintenance procedures page 45"'
    )
    
    console.print("\n[bold]Page-specific search results:[/bold]")
    if stdout:
        # Show first few results with page highlighting
        for line in stdout.split('\n')[:20]:
            if 'page' in line.lower() or 'Page' in line:
                console.print(f"[yellow]{line}[/yellow]")
            else:
                console.print(line)
    
    # Show extraction capabilities summary
    print_section_header("Extraction Capabilities Demonstrated", "🎖️")
    
    extraction_table = Table(show_header=True, header_style="bold cyan", title="AFI-Specific Extractions")
    extraction_table.add_column("Extraction Type", style="yellow")
    extraction_table.add_column("Pattern/Entity", style="white")
    extraction_table.add_column("Purpose", style="dim")
    
    extraction_table.add_row(
        "Page Numbers",
        "Page X of Y",
        "Exact citation and reference"
    )
    extraction_table.add_row(
        "Paragraph References",
        "3.2.1.4 format",
        "Navigate AFI structure"
    )
    extraction_table.add_row(
        "AF Forms",
        "AF Form 123",
        "Required documentation"
    )
    extraction_table.add_row(
        "Technical Orders",
        "T.O. references",
        "Maintenance procedures"
    )
    extraction_table.add_row(
        "Compliance Terms",
        "shall/must/will",
        "Mandatory requirements"
    )
    extraction_table.add_row(
        "Safety Statements",
        "WARNING/CAUTION/NOTE",
        "Critical safety info"
    )
    extraction_table.add_row(
        "Military Entities",
        "MAJCOM, AFSC, MDS",
        "Organization context"
    )
    extraction_table.add_row(
        "Tables & Figures",
        "Table/Figure refs",
        "Visual content location"
    )
    
    console.print(extraction_table)
    
    # Show advanced search techniques
    print_section_header("Advanced Search Techniques", "🚀")
    
    console.print("🎯 [bold green]Advanced AFI search capabilities:[/bold green]")
    
    techniques_table = Table(show_header=True, header_style="bold cyan")
    techniques_table.add_column("Technique", style="yellow")
    techniques_table.add_column("Example Query", style="white")
    techniques_table.add_column("Result", style="dim")
    
    techniques_table.add_row(
        "Page-specific",
        "procedures page 45",
        "Content from specific page"
    )
    techniques_table.add_row(
        "Paragraph reference",
        "section 3.2.1",
        "Specific paragraph content"
    )
    techniques_table.add_row(
        "Compliance search",
        "shall requirements",
        "All mandatory requirements"
    )
    techniques_table.add_row(
        "Form lookup",
        "AF Form 2005",
        "Form usage and requirements"
    )
    techniques_table.add_row(
        "Safety search",
        "WARNING statements",
        "All safety warnings"
    )
    techniques_table.add_row(
        "T.O. reference",
        "Technical Order maintenance",
        "Related T.O. procedures"
    )
    
    console.print(techniques_table)
    
    # Summary
    print_section_header("AFI Demo Summary", "🎓")
    
    console.print("✈️ [bold green]AFI Document Analysis Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ [bold cyan]Smart PDF parsing with page-level precision[/bold cyan]")
    console.print("✅ Military-specific pattern and entity extraction")
    console.print("✅ Compliance requirement identification (shall/must/will)")
    console.print("✅ Exact page number and paragraph reference tracking")
    console.print("✅ Safety statement extraction (WARNING/CAUTION/NOTE)")
    console.print("✅ Hybrid retrieval strategy for technical documentation")
    console.print("✅ Metadata-based filtering for precise location")
    
    console.print(f"\n[bold]Key Benefits for Maintenance Officers:[/bold]")
    console.print("🎯 Quick access to specific procedures by page/paragraph")
    console.print("📋 Instant compliance requirement identification")
    console.print("⚠️ Highlighted safety warnings and cautions")
    console.print("📄 Accurate citations with page numbers")
    console.print("🔍 Semantic search across entire AFI")
    console.print("📊 Structured extraction of forms and T.O. references")
    
    console.print(f"\n[bold]CLI Commands for AFI Processing:[/bold]")
    console.print("📋 `cli.py --strategy-file demos/demo_strategies.yaml ingest --strategy afi_document_demo <pdf>` - Ingest AFI")
    console.print("🔍 `cli.py --content-length 800 --strategy-file demos/demo_strategies.yaml search --strategy afi_document_demo '<query>'` - Search AFI with context")
    console.print("📊 `cli.py --verbose --content-length 1000 ... search ...` - Show full content with metadata")
    console.print("🎯 `cli.py --content-length -1 ... search ... 'page 45'` - Show complete chunks from page 45")
    console.print("⚠️ `cli.py ... search ... 'WARNING'` - Find safety statements")
    
    console.print(f"\n📁 AFI database saved to: [bold]./demos/vectordb/afi_documents[/bold]")
    console.print("🔄 You can continue querying the AFI database:")
    console.print("[dim]$ python cli.py --content-length 800 --strategy-file demos/demo_strategies.yaml search --strategy afi_document_demo 'maintenance procedures'[/dim]")
    console.print("[dim]$ python cli.py --verbose --content-length 1000 --strategy-file demos/demo_strategies.yaml search --strategy afi_document_demo 'section 3.2'[/dim]")
    
    # Optional: Clean up the database after demo
    console.print("\n🧹 [bold cyan]Keeping database for continued use...[/bold cyan]")
    console.print("[dim]💡 To clean up later, run:[/dim]")
    console.print("[dim]$ python cli.py --strategy-file demos/demo_strategies.yaml manage --strategy afi_document_demo delete --all[/dim]")


if __name__ == "__main__":
    try:
        demonstrate_afi_document_rag_cli()
    except KeyboardInterrupt:
        console.print("\n\n👋 AFI demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ AFI demo failed: {str(e)}", style="red")
        console.print("Make sure you're running from the rag directory")
        sys.exit(1)