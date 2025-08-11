#!/usr/bin/env python3
"""
Enhanced RAG Demo: Real Embedding and Retrieval Showcase
Demonstrates the complete RAG pipeline with detailed output showing what's being embedded and retrieved.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.text import Text

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

# Import RAG components
from core.base import Document, Pipeline
from components.parsers.text_parser import PlainTextParser
from components.parsers.markdown_parser import MarkdownParser
from components.embedders.ollama_embedder import OllamaEmbedder
from components.stores.chroma_store import ChromaStore

# Setup rich console for beautiful output
console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def print_section_header(title: str, emoji: str = "🔥"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def print_document_content(doc: Document, max_chars: int = 500):
    """Print document content with rich formatting."""
    content_preview = doc.content[:max_chars]
    if len(doc.content) > max_chars:
        content_preview += "..."
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    table.add_row("Document ID", doc.id)
    table.add_row("Source", doc.source)
    table.add_row("Content Length", f"{len(doc.content):,} characters")
    table.add_row("Word Count", f"{len(doc.content.split()):,} words")
    
    console.print(table)
    console.print(Panel(content_preview, title="Content Preview", expand=False))


def print_embedding_info(embeddings: List[float]):
    """Print embedding information."""
    if not embeddings:
        console.print("❌ No embeddings found", style="red")
        return
    
    table = Table(show_header=True, header_style="bold green")
    table.add_column("Embedding Property", style="cyan")
    table.add_column("Value", style="white")
    
    table.add_row("Dimension", str(len(embeddings)))
    table.add_row("First 5 values", str([round(x, 4) for x in embeddings[:5]]))
    table.add_row("Last 5 values", str([round(x, 4) for x in embeddings[-5:]]))
    table.add_row("Min value", f"{min(embeddings):.4f}")
    table.add_row("Max value", f"{max(embeddings):.4f}")
    table.add_row("Mean value", f"{sum(embeddings)/len(embeddings):.4f}")
    
    console.print(table)


def print_search_results(query: str, results: List[Document], max_results: int = 3):
    """Print search results with detailed information."""
    console.print(f"\n🔍 Query: [bold yellow]'{query}'[/bold yellow]")
    console.print(f"📊 Found {len(results)} results (showing top {min(max_results, len(results))})")
    
    for i, doc in enumerate(results[:max_results], 1):
        # Create a result panel
        score = doc.metadata.get('search_score', 'N/A')
        score_text = f"Score: {score:.4f}" if isinstance(score, float) else f"Score: {score}"
        
        # Show content preview
        content_preview = doc.content[:300] + "..." if len(doc.content) > 300 else doc.content
        
        result_text = f"[bold]Source:[/bold] {doc.source}\n[bold]{score_text}[/bold]\n\n{content_preview}"
        
        console.print(Panel(
            result_text,
            title=f"Result #{i}",
            title_align="left",
            border_style="green" if i == 1 else "blue",
            expand=False
        ))


def demonstrate_rag_pipeline():
    """Demonstrate the complete RAG pipeline with detailed output."""
    
    print_section_header("🦙 Enhanced RAG Demo: Real Embedding & Retrieval", "🚀")
    
    console.print("\n[bold green]This demo will show you:[/bold green]")
    console.print("• What documents are being parsed and embedded")
    console.print("• The actual embedding vectors generated")
    console.print("• Real retrieval results with similarity scores")
    console.print("• Complete transparency in the RAG process")
    
    # Initialize components
    print_section_header("Component Initialization", "⚙️")
    
    console.print("🔧 Initializing parsers...")
    text_parser = PlainTextParser(
        name="DemoTextParser",
        config={
            "preserve_line_breaks": True,
            "detect_structure": True
        }
    )
    markdown_parser = MarkdownParser(
        name="DemoMarkdownParser", 
        config={
            "chunk_by_headings": True,
            "extract_frontmatter": True
        }
    )
    
    console.print("🧠 Initializing embedder (Ollama)...")
    embedder = OllamaEmbedder(name="enhanced_demo_embedder", config={
        "model": "nomic-embed-text",
        "batch_size": 4
    })
    
    console.print("🗄️ Initializing vector store (ChromaDB)...")
    vector_store = ChromaStore(name="enhanced_demo_store", config={
        "collection_name": "enhanced_demo",
        "persist_directory": "./demos/vectordb/enhanced_demo"
    })
    
    console.print("✅ All components initialized successfully!")
    
    # Parse documents
    print_section_header("Document Parsing", "📚")
    
    sample_files = [
        "samples/documents/ai_overview.txt",
        "samples/documents/machine_learning_guide.md", 
        "samples/documents/data_science_workflow.txt",
        "samples/documents/python_programming_basics.md"
    ]
    
    all_documents = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        parse_task = progress.add_task("Parsing documents...", total=len(sample_files))
        
        for file_path in sample_files:
            if not Path(file_path).exists():
                console.print(f"⚠️ File not found: {file_path}", style="yellow")
                progress.advance(parse_task)
                continue
            
            # Choose parser based on file extension
            if file_path.endswith('.md'):
                parser = markdown_parser
            else:
                parser = text_parser
            
            result = parser.parse(file_path)
            docs = result.documents
            all_documents.extend(docs)
            
            console.print(f"📄 Parsed [bold]{file_path}[/bold]: {len(docs)} document(s)")
            progress.advance(parse_task)
    
    console.print(f"\n✅ Total documents parsed: [bold green]{len(all_documents)}[/bold green]")
    
    # Show document details
    print_section_header("Document Analysis", "🔍")
    
    for i, doc in enumerate(all_documents[:2], 1):  # Show first 2 documents in detail
        console.print(f"\n📖 Document #{i} Details:")
        print_document_content(doc)
    
    if len(all_documents) > 2:
        console.print(f"\n... and {len(all_documents) - 2} more documents")
    
    # Generate embeddings
    print_section_header("Embedding Generation", "🧠")
    
    console.print("🔄 Generating embeddings for all documents...")
    console.print("This process converts text into high-dimensional vectors that capture semantic meaning.")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        embed_task = progress.add_task("Generating embeddings...", total=len(all_documents))
        
        for doc in all_documents:
            if not doc.embeddings:
                embeddings = embedder.embed([doc.content])
                doc.embeddings = embeddings[0] if embeddings else []
            progress.advance(embed_task)
    
    console.print(f"\n✅ Generated embeddings for {len(all_documents)} documents")
    
    # Show embedding details for first document
    if all_documents and all_documents[0].embeddings:
        console.print(f"\n🔬 Embedding Analysis for: [bold]{all_documents[0].source}[/bold]")
        print_embedding_info(all_documents[0].embeddings)
    
    # Store documents
    print_section_header("Vector Storage", "🗄️")
    
    console.print("💾 Storing documents in vector database...")
    success = vector_store.add_documents(all_documents)
    if success:
        console.print(f"✅ Stored {len(all_documents)} documents in vector database")
    else:
        console.print("❌ Failed to store documents in vector database")
        return
    
    # Demonstrate retrieval
    print_section_header("Real Retrieval Demonstration", "🔍")
    
    queries = [
        "What is machine learning and how does it work?",
        "Explain Python programming basics and syntax",
        "How do you implement a data science project workflow?",
        "What are the applications of artificial intelligence?",
        "Tell me about neural networks and deep learning"
    ]
    
    console.print("🎯 Running multiple retrieval queries to show the RAG system in action:")
    
    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold cyan]Query #{i}:[/bold cyan]")
        
        # Generate query embedding
        console.print("🧠 Generating query embedding...")
        query_embeddings = embedder.embed([query])
        query_embedding = query_embeddings[0] if query_embeddings else []
        
        console.print(f"📐 Query embedding dimension: {len(query_embedding)}")
        console.print(f"🔢 Query embedding sample: {[round(x, 4) for x in query_embedding[:5]]}...")
        
        # Search for similar documents
        console.print("🔍 Searching for similar documents...")
        results = vector_store.search(query_embedding=query_embedding, top_k=3)
        
        # Show results
        print_search_results(query, results)
        
        if i < len(queries):
            console.print("\n" + "─" * 50)
            time.sleep(1)  # Brief pause for readability
    
    # Show collection statistics
    print_section_header("Vector Database Statistics", "📊")
    
    info = vector_store.get_collection_info()
    if info:
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Collection Name", info.get("name", "N/A"))
        stats_table.add_row("Total Documents", str(info.get("document_count", "N/A")))
        embedding_dim = len(all_documents[0].embeddings) if all_documents and all_documents[0].embeddings else "N/A"
        stats_table.add_row("Embedding Dimension", str(embedding_dim))
        stats_table.add_row("Model Used", embedder.model)
        
        console.print(stats_table)
    
    # Final summary
    print_section_header("Demo Complete!", "🎉")
    
    console.print("🚀 [bold green]RAG Pipeline Successfully Demonstrated![/bold green]")
    console.print("\n[bold]What you just saw:[/bold]")
    console.print("✅ Real documents parsed from multiple file formats")
    console.print("✅ Actual embedding vectors generated using Ollama")
    console.print("✅ Documents stored in ChromaDB vector database")
    console.print("✅ Semantic search queries with similarity scores")
    console.print("✅ Complete transparency in embeddings and retrieval")
    
    console.print(f"\n📁 Vector database saved to: [bold]./demos/vectordb/enhanced_demo[/bold]")
    console.print("🔄 You can now query this database using the CLI:")
    console.print("[dim]uv run python cli.py search 'your query here'[/dim]")


if __name__ == "__main__":
    try:
        demonstrate_rag_pipeline()
    except KeyboardInterrupt:
        console.print("\n\n👋 Demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Demo failed: {str(e)}", style="red")
        console.print("Check that Ollama is running with the nomic-embed-text model")
        sys.exit(1)