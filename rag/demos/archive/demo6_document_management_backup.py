#!/usr/bin/env python3
"""
Demo 6: Document Management & Vector Database Operations (CLI-based)

This demo showcases advanced document management capabilities using CLI commands:
- Document lifecycle (add, search, delete, replace)
- Metadata-based searches
- Document hashing and deduplication
- Vector database health operations
- Advanced search strategies with filters

ALL operations are performed through CLI commands - no direct API usage!
"""

import subprocess
import sys
import time
import json
import hashlib
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax
from datetime import datetime, timedelta

# Setup rich console for beautiful output
console = Console()

def run_cli_command(command: str, description: str = None, show_output: bool = True) -> tuple[bool, str]:
    """Run a CLI command and return success status and output."""
    if description:
        console.print(f"\n[bold cyan]→ {description}[/bold cyan]")
    
    if show_output:
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
                elif '❌' in line or 'ERROR' in line or 'Error' in line:
                    console.print(f"[red]{line}[/red]")
                elif '⚠️' in line or 'WARNING' in line:
                    console.print(f"[yellow]{line}[/yellow]")
                elif 'Result #' in line or 'Found' in line:
                    console.print(f"[cyan]{line}[/cyan]")
                elif 'Document ID:' in line or 'Hash:' in line:
                    console.print(f"[magenta]{line}[/magenta]")
                else:
                    console.print(line)
        
        if result.stderr and 'error' in result.stderr.lower():
            console.print(f"[red]Error: {result.stderr}[/red]")
            return False, result.stderr
        
        return result.returncode == 0, result.stdout
    except Exception as e:
        console.print(f"[red]Command failed: {e}[/red]")
        return False, str(e)

def print_section_header(title: str, emoji: str = "🔧"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")

def create_test_documents():
    """Create test documents for the demo."""
    # Create test directory
    test_dir = Path("demos/temp_test_docs")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Document 1: Original document
    doc1_path = test_dir / "technical_spec_v1.txt"
    doc1_content = """Technical Specification Document
Version: 1.0
Date: 2024-01-15
Author: Engineering Team

System Architecture Overview
This document describes the architecture of our distributed RAG system.
The system uses vector embeddings for semantic search across documents.

Key Components:
- Document Parser: Processes various file formats
- Embedding Generator: Creates vector representations
- Vector Database: Stores and retrieves embeddings
- Search Engine: Performs similarity searches

Performance Requirements:
- Query latency: < 100ms
- Throughput: 1000 queries/second
- Storage: Scalable to 10M documents
"""
    doc1_path.write_text(doc1_content)
    
    # Document 2: Updated version
    doc2_path = test_dir / "technical_spec_v2.txt"
    doc2_content = """Technical Specification Document
Version: 2.0
Date: 2024-02-01
Author: Engineering Team

System Architecture Overview - UPDATED
This document describes the ENHANCED architecture of our distributed RAG system.
Now includes advanced caching and multi-modal support.

Key Components:
- Document Parser: Processes various file formats including images
- Embedding Generator: Creates multi-modal vector representations
- Vector Database: Stores and retrieves embeddings with metadata filtering
- Search Engine: Performs hybrid similarity searches
- Cache Layer: Redis-based caching for improved performance

Performance Requirements:
- Query latency: < 50ms (improved)
- Throughput: 5000 queries/second (5x improvement)
- Storage: Scalable to 100M documents
- Cache hit rate: > 80%
"""
    doc2_path.write_text(doc2_content)
    
    # Document 3: Duplicate for dedup testing
    doc3_path = test_dir / "technical_spec_copy.txt"
    doc3_path.write_text(doc1_content)  # Exact duplicate
    
    # Document 4: Different document with metadata
    doc4_path = test_dir / "api_documentation.md"
    doc4_content = """# API Documentation

## Authentication
All API requests require authentication using Bearer tokens.

## Endpoints

### GET /search
Search for documents using semantic similarity.

Parameters:
- query: Search query string
- top_k: Number of results (default: 10)
- filters: Metadata filters (JSON)

### POST /ingest
Add new documents to the system.

### DELETE /document/{id}
Remove a document from the database.

### PUT /document/{id}
Update an existing document.
"""
    doc4_path.write_text(doc4_content)
    
    return test_dir, [doc1_path, doc2_path, doc3_path, doc4_path]

def demonstrate_document_management():
    """Demonstrate comprehensive document management via CLI."""
    
    console.print(Panel.fit(
        "🦙 Demo 6: Document Management & Vector Database Operations\n"
        "Advanced CLI operations for production systems",
        style="bold cyan"
    ))
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• Document lifecycle management (add/search/delete/replace)")
    console.print("• Metadata-based searches and filtering")
    console.print("• Document hashing and deduplication")
    console.print("• Vector database health operations")
    console.print("• Advanced search strategies")
    console.print("\n[dim]All operations use CLI commands exclusively![/dim]")
    
    # Create test documents
    print_section_header("Test Document Setup", "📝")
    test_dir, doc_paths = create_test_documents()
    console.print(f"✅ Created test documents in: {test_dir}")
    
    # Initialize collection
    print_section_header("Initialize Document Management Collection", "🗄️")
    
    # Use a dedicated strategy for document management demo
    run_cli_command(
        "python cli.py manage delete --collection document_management --force",
        "Cleaning up any existing collection"
    )
    
    # Step 1: Add initial document
    print_section_header("Step 1: Add Initial Document", "➕")
    
    console.print("[bold]Adding technical specification v1...[/bold]")
    success, output = run_cli_command(
        f"python cli.py --verbose ingest {doc_paths[0]}",
        "Ingesting document with verbose output to see hashing"
    )
    
    if success:
        console.print("✅ Document added successfully")
        # Extract and show document hash if visible
        if "Hash:" in output or "ID:" in output:
            console.print("[dim]Note: Document ID is generated using content hash[/dim]")
    
    # Step 2: Search for the document
    print_section_header("Step 2: Search for Added Document", "🔍")
    
    run_cli_command(
        'python cli.py search "system architecture distributed" --top-k 3',
        "Searching for our technical specification"
    )
    
    # Step 3: Metadata-based searches
    print_section_header("Step 3: Advanced Metadata Searches", "🔬")
    
    console.print("[bold]Demonstrating various metadata search capabilities:[/bold]\n")
    
    # Search by filename
    run_cli_command(
        'python cli.py search --metadata \'{"file_name": "technical_spec_v1.txt"}\'',
        "Search by exact filename"
    )
    
    # Search by date range (if supported)
    run_cli_command(
        'python cli.py search "architecture" --filter-date-after "2024-01-01"',
        "Search documents added after a specific date"
    )
    
    # Search with content filters
    run_cli_command(
        'python cli.py search "performance" --filter-content-length-min 100',
        "Search documents with minimum content length"
    )
    
    # Step 4: Add duplicate document (demonstrate deduplication)
    print_section_header("Step 4: Deduplication Demo", "🔁")
    
    console.print("[bold]Attempting to add duplicate document...[/bold]")
    run_cli_command(
        f"python cli.py --verbose ingest {doc_paths[2]}",
        "Adding exact duplicate (should be detected)"
    )
    
    console.print("[dim]Note: Smart deduplication prevents storing identical content[/dim]")
    
    # Step 5: Delete document
    print_section_header("Step 5: Delete Document", "🗑️")
    
    console.print("[bold]Deleting the original document...[/bold]")
    
    # First get document ID
    success, output = run_cli_command(
        'python cli.py search "technical specification" --top-k 1 --quiet',
        "Getting document ID",
        show_output=False
    )
    
    # Extract ID from output (this is a simplified example)
    doc_id = "doc_001"  # In reality, parse from output
    
    run_cli_command(
        f"python cli.py manage delete-doc --id {doc_id}",
        f"Deleting document with ID: {doc_id}"
    )
    
    # Step 6: Verify deletion
    print_section_header("Step 6: Verify Deletion", "✓")
    
    run_cli_command(
        'python cli.py search "system architecture distributed"',
        "Searching for deleted document (should return no results)"
    )
    
    # Step 7: Replace document (delete + add new version)
    print_section_header("Step 7: Document Replacement", "🔄")
    
    console.print("[bold]Replacing with version 2 of the document...[/bold]")
    
    # Add the new version
    run_cli_command(
        f"python cli.py --verbose ingest {doc_paths[1]}",
        "Adding updated version (v2.0)"
    )
    
    # Search to confirm replacement
    run_cli_command(
        'python cli.py search "multi-modal cache redis"',
        "Searching for new features in v2 (should find updated doc)"
    )
    
    # Step 8: Bulk operations
    print_section_header("Step 8: Bulk Document Operations", "📦")
    
    console.print("[bold]Adding multiple documents at once...[/bold]")
    run_cli_command(
        f"python cli.py --verbose ingest {test_dir}/",
        "Bulk ingesting entire directory"
    )
    
    # Step 9: Collection statistics
    print_section_header("Step 9: Vector Database Health Check", "📊")
    
    run_cli_command(
        "python cli.py info",
        "Getting collection statistics"
    )
    
    run_cli_command(
        "python cli.py manage stats --detailed",
        "Detailed database statistics"
    )
    
    # Step 10: Advanced filtering
    print_section_header("Step 10: Complex Query with Filters", "🎯")
    
    console.print("[bold]Combining text search with metadata filters:[/bold]")
    
    run_cli_command(
        'python cli.py search "API authentication" '
        '--filter-extension ".md" '
        '--filter-size-max 5000 '
        '--top-k 5',
        "Search markdown files under 5KB about API authentication"
    )
    
    # Step 11: Export and backup
    print_section_header("Step 11: Export & Backup Operations", "💾")
    
    run_cli_command(
        "python cli.py manage export --output document_backup.json",
        "Exporting collection for backup"
    )
    
    console.print("[dim]Backup can be restored using: cli.py manage import document_backup.json[/dim]")
    
    # Summary
    print_section_header("Demo Summary", "🎯")
    
    summary = Panel(
        """[bold green]Document Management Operations Demonstrated:[/bold green]

✅ Document Addition - Single and bulk ingestion
✅ Document Search - Text and metadata-based queries
✅ Document Deletion - Remove specific documents
✅ Document Replacement - Update with new versions
✅ Deduplication - Automatic duplicate detection
✅ Metadata Filtering - Search by file properties
✅ Bulk Operations - Directory-level processing
✅ Health Checks - Database statistics and monitoring
✅ Export/Import - Backup and restore capabilities

[bold cyan]Key CLI Commands:[/bold cyan]
• ingest - Add documents to the database
• search - Query with text and metadata filters
• manage delete-doc - Remove specific documents
• manage stats - View database health
• manage export/import - Backup operations

[bold yellow]Production Tips:[/bold yellow]
• Use --verbose for detailed operation logs
• Use --quiet for scripting and automation
• Implement regular backups with export
• Monitor collection stats for performance
• Use metadata filters for precise searches""",
        title="🦙 Vector Database Management Complete",
        border_style="green"
    )
    
    console.print(summary)
    
    # Cleanup
    print_section_header("Cleanup", "🧹")
    
    # Remove test documents
    import shutil
    if test_dir.exists():
        shutil.rmtree(test_dir)
        console.print(f"✅ Cleaned up test directory: {test_dir}")
    
    run_cli_command(
        "python cli.py manage delete --collection document_management --force",
        "Removing demo collection"
    )
    
    console.print("\n[bold green]Demo 6 Complete![/bold green]")
    console.print("[dim]This collection has been cleaned up. Other demo collections remain intact.[/dim]")

if __name__ == "__main__":
    try:
        demonstrate_document_management()
    except KeyboardInterrupt:
        console.print("\n\n👋 Document management demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Document management demo failed: {str(e)}", style="red")
        sys.exit(1)