#!/usr/bin/env python3
"""
Shared utilities for RAG demos to enhance educational value.
"""

from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from datetime import datetime
import hashlib

from core.base import Document

console = Console()


def display_document_with_metadata(doc: Document, index: int = 0, title: str = "Document") -> None:
    """Display a document with all its metadata in an educational format."""
    
    # Create metadata table
    metadata_table = Table(show_header=True, header_style="bold yellow", title=f"📄 {title} #{index + 1}")
    metadata_table.add_column("Metadata Field", style="cyan", width=20)
    metadata_table.add_column("Value", style="white")
    metadata_table.add_column("Description", style="dim")
    
    # Basic document info
    metadata_table.add_row("Document ID", doc.id or "Not set", "Unique identifier for retrieval")
    metadata_table.add_row("Source", doc.source or "Unknown", "Original file/location")
    
    # Content hash for deduplication
    content_hash = hashlib.md5(doc.content.encode()).hexdigest()[:12]
    metadata_table.add_row("Content Hash", content_hash, "For deduplication")
    
    # Content stats
    metadata_table.add_row("Content Length", f"{len(doc.content)} chars", "Raw text size")
    word_count = len(doc.content.split())
    metadata_table.add_row("Word Count", str(word_count), "Total words")
    
    # Timestamp
    if doc.metadata and "timestamp" in doc.metadata:
        timestamp = doc.metadata["timestamp"]
    else:
        timestamp = datetime.now().isoformat()
    metadata_table.add_row("Processed At", timestamp, "Processing timestamp")
    
    # Custom metadata
    if doc.metadata:
        for key, value in doc.metadata.items():
            if key not in ["timestamp", "id", "source"]:  # Avoid duplicates
                # Format the value based on type
                if isinstance(value, list):
                    formatted_value = f"[{len(value)} items]"
                    description = f"List: {', '.join(str(v)[:20] + '...' if len(str(v)) > 20 else str(v) for v in value[:3])}"
                    if len(value) > 3:
                        description += f" ... ({len(value) - 3} more)"
                elif isinstance(value, dict):
                    formatted_value = f"[{len(value)} fields]"
                    description = "Nested data structure"
                elif isinstance(value, (int, float)):
                    formatted_value = str(value)
                    description = "Numeric value"
                elif isinstance(value, bool):
                    formatted_value = "✓" if value else "✗"
                    description = "Boolean flag"
                else:
                    formatted_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                    description = "Text value"
                
                metadata_table.add_row(key.replace("_", " ").title(), formatted_value, description)
    
    # Embeddings info
    if doc.embeddings:
        embedding_dim = len(doc.embeddings)
        embedding_preview = f"[{doc.embeddings[0]:.3f}, {doc.embeddings[1]:.3f}, ..., {doc.embeddings[-1]:.3f}]"
        metadata_table.add_row("Embedding Dimension", str(embedding_dim), "Vector size")
        metadata_table.add_row("Embedding Preview", embedding_preview, "First, second, and last values")
    else:
        metadata_table.add_row("Embeddings", "Not generated yet", "Will be added during processing")
    
    console.print(metadata_table)
    
    # Show content preview
    content_preview = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
    console.print(Panel(content_preview, title="[bold green]Content Preview[/bold green]", border_style="green"))
    console.print()


def display_embedding_process(documents: List[Document], embedder_name: str = "Ollama") -> None:
    """Show the embedding process with metadata preservation."""
    
    console.print(f"\n[bold cyan]🧠 Embedding Process with {embedder_name}[/bold cyan]")
    console.print("=" * 80)
    
    # Create embedding process table
    process_table = Table(show_header=True, header_style="bold green")
    process_table.add_column("Step", style="yellow", width=5)
    process_table.add_column("Document ID", style="cyan", width=20)
    process_table.add_column("Content Size", style="white", width=15)
    process_table.add_column("Metadata Fields", style="magenta", width=20)
    process_table.add_column("Status", style="green", width=20)
    
    for i, doc in enumerate(documents):
        metadata_count = len(doc.metadata) if doc.metadata else 0
        metadata_fields = ", ".join(list(doc.metadata.keys())[:3]) if doc.metadata else "None"
        if doc.metadata and len(doc.metadata) > 3:
            metadata_fields += f" +{len(doc.metadata) - 3} more"
        
        process_table.add_row(
            str(i + 1),
            doc.id or f"doc_{i}",
            f"{len(doc.content)} chars",
            metadata_fields,
            "✓ Ready for embedding"
        )
    
    console.print(process_table)
    console.print(f"\n[dim]Note: All metadata is preserved during embedding and will be stored in the vector database.[/dim]")


def display_search_results_with_metadata(results: List[Document], query: str) -> None:
    """Display search results with full metadata context."""
    
    console.print(f"\n[bold blue]🔍 Search Results for: '{query}'[/bold blue]")
    console.print(f"[dim]Found {len(results)} relevant documents[/dim]\n")
    
    for i, doc in enumerate(results):
        # Create result panel
        # Prefer pre-computed similarity score, fall back to converting distance
        similarity_score = doc.metadata.get('_similarity_score')
        if similarity_score is not None:
            result_content = f"[bold]Relevance Score:[/bold] {similarity_score:.4f}\n"
        else:
            # Fall back to raw distance conversion
            score = doc.metadata.get('search_score', doc.metadata.get('_score', 'N/A'))
            if isinstance(score, float):
                # Convert ChromaDB distance to similarity score (lower distance = higher similarity)
                # Use exponential decay for better score distribution: e^(-distance/scale)
                # Scale factor chosen to give reasonable scores for typical L2 distances
                import math
                scale_factor = 100.0  # Adjust this to tune score sensitivity
                similarity_score = math.exp(-score / scale_factor)
                result_content = f"[bold]Relevance Score:[/bold] {similarity_score:.4f}\n"
            else:
                result_content = f"[bold]Relevance Score:[/bold] {score}\n"
        result_content += f"[bold]Document ID:[/bold] {doc.id or 'Not set'}\n"
        
        # Better source handling
        source_display = "Unknown"
        if doc.source:
            from pathlib import Path
            source_display = Path(doc.source).name
        elif doc.metadata and doc.metadata.get('file_name'):
            source_display = doc.metadata['file_name']
        elif doc.metadata and doc.metadata.get('file_path'):
            from pathlib import Path
            source_display = Path(doc.metadata['file_path']).name
            
        result_content += f"[bold]Source:[/bold] {source_display}\n"
        
        # Add key metadata
        if doc.metadata:
            important_fields = ["author", "date", "type", "category", "tags", "priority", "status"]
            for field in important_fields:
                if field in doc.metadata:
                    result_content += f"[bold]{field.title()}:[/bold] {doc.metadata[field]}\n"
        
        result_content += f"\n[bold]Content:[/bold]\n{doc.content[:300]}..."
        
        panel = Panel(
            result_content,
            title=f"[bold cyan]Result #{i + 1}[/bold cyan]",
            border_style="blue"
        )
        console.print(panel)


def add_processing_timestamp(documents: List[Document]) -> List[Document]:
    """Add processing timestamp to all documents."""
    timestamp = datetime.now().isoformat()
    for doc in documents:
        if not doc.metadata:
            doc.metadata = {}
        doc.metadata["processing_timestamp"] = timestamp
        doc.metadata["processing_date"] = datetime.now().strftime("%Y-%m-%d")
    return documents


def generate_document_id(content: str, source: str = None) -> str:
    """Generate a deterministic document ID based on content and source."""
    id_string = f"{source or 'unknown'}:{content[:100]}"
    return hashlib.sha256(id_string.encode()).hexdigest()[:16]


def display_demo_separator() -> None:
    """Display a separator between demo sections."""
    console.print("\n" + "─" * 80 + "\n", style="dim")