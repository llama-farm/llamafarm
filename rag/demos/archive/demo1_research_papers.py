#!/usr/bin/env python3
"""
Demo 1: Research Paper Analysis System
Demonstrates advanced RAG capabilities for academic research papers using:
- Strategy-first configuration approach (no hardcoded parameters!)
- Plain text parser for research paper content
- Academic-focused extractors (statistics, citations, summaries)
- Semantic search optimized for research queries
- Entity extraction for authors, institutions, methodologies

This demo showcases the power of the strategy system:
- All configuration is in demo_strategies.yaml
- Easy to modify behavior without code changes
- Clean separation of configuration from implementation
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
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import strategy-based demo system
from demos.strategy_demo_utils import create_demo_system
from core.base import Document

# Import demo utilities for metadata display
from demos.utils import (
    display_document_with_metadata,
    display_embedding_process,
    display_search_results_with_metadata,
    add_processing_timestamp,
    generate_document_id,
    display_demo_separator
)

# Setup rich console for beautiful output
console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def print_section_header(title: str, emoji: str = "🔬"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def print_document_analysis(doc: Document, show_extractors: bool = True):
    """Print detailed document analysis with extractor results."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    table.add_row("Document ID", doc.id)
    table.add_row("Source", doc.source)
    table.add_row("Content Length", f"{len(doc.content):,} characters")
    table.add_row("Word Count", f"{len(doc.content.split()):,} words")
    
    if show_extractors and doc.metadata:
        # Show extractor results
        if 'extractors' in doc.metadata and 'statistics' in doc.metadata['extractors']:
            stats = doc.metadata['extractors']['statistics']
            table.add_row("Statistics Found", f"Content analysis completed")
        
        if 'entities' in doc.metadata:
            entities = doc.metadata['entities']
            table.add_row("Entities Extracted", f"{len(entities)} named entities")
        
        if 'summary' in doc.metadata:
            summary = doc.metadata['summary'][:100] + "..." if len(doc.metadata['summary']) > 100 else doc.metadata['summary']
            table.add_row("Auto Summary", summary)
    
    console.print(table)


def print_extractor_details(documents: List[Document]):
    """Print detailed extractor analysis results."""
    console.print("\n📊 [bold green]Extractor Analysis Results[/bold green]")
    
    for i, doc in enumerate(documents[:2], 1):  # Show first 2 docs in detail
        source_name = Path(doc.source).name if doc.source else f"Document {i}"
        console.print(f"\n📄 Document #{i}: {source_name}")
        
        # Statistics Extractor Results
        if 'extractors' in doc.metadata and 'statistics' in doc.metadata['extractors']:
            stats_data = doc.metadata['extractors']['statistics']
            stats_table = Table(title="📈 Content Analysis Results", show_header=True, header_style="bold yellow")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")
            stats_table.add_column("Description", style="dim")
            
            # Show key metrics from content analysis
            if 'basic' in stats_data:
                basic = stats_data['basic']
                stats_table.add_row("Word Count", str(basic.get('word_count', 'N/A')), "Total words in document")
                stats_table.add_row("Sentences", str(basic.get('sentence_count', 'N/A')), "Number of sentences")
                stats_table.add_row("Paragraphs", str(basic.get('paragraph_count', 'N/A')), "Document structure")
            
            if 'readability' in stats_data:
                readability = stats_data['readability']
                stats_table.add_row("Reading Level", readability.get('reading_level', 'N/A'), "Text complexity")
                stats_table.add_row("Reading Time", f"{readability.get('reading_time_minutes', 0):.1f} min", "Estimated reading time")
            
            console.print(stats_table)
        
        # Entity Extractor Results
        if 'extractors' in doc.metadata and 'entities' in doc.metadata['extractors']:
            entities_data = doc.metadata['extractors']['entities']
            entities_table = Table(title="👥 Named Entities Identified", show_header=True, header_style="bold green")
            entities_table.add_column("Entity", style="cyan")
            entities_table.add_column("Type", style="yellow")
            entities_table.add_column("Context", style="white")
            
            # Show first 8 entities
            entities_list = entities_data.get('entities', []) if isinstance(entities_data, dict) else entities_data
            for entity in entities_list[:8]:
                entities_table.add_row(
                    entity.get('text', 'N/A'),
                    entity.get('label', 'N/A'),
                    entity.get('context', 'N/A')[:30] + "..." if len(entity.get('context', '')) > 30 else entity.get('context', 'N/A')
                )
            
            console.print(entities_table)
        
        # Summary Extractor Results
        if 'extractors' in doc.metadata and 'summary' in doc.metadata['extractors']:
            summary_data = doc.metadata['extractors']['summary']
            summary_text = summary_data.get('summary', 'No summary available') if isinstance(summary_data, dict) else str(summary_data)
            summary_panel = Panel(
                summary_text,
                title="📝 Generated Summary",
                border_style="blue",
                expand=False
            )
            console.print(summary_panel)


def print_search_results_detailed(query: str, results: List[Document]):
    """Print detailed search results with research focus."""
    console.print(f"\n🔍 Query: [bold yellow]'{query}'[/bold yellow]")
    console.print(f"📊 Found {len(results)} relevant research documents")
    
    for i, doc in enumerate(results[:3], 1):  # Show top 3 results
        score = doc.metadata.get('search_score', 'N/A')
        score_text = f"Relevance: {score:.4f}" if isinstance(score, float) else f"Relevance: {score}"
        
        # Show research-specific metadata
        research_info = []
        if 'extractors' in doc.metadata and isinstance(doc.metadata['extractors'], dict):
            extractors = doc.metadata['extractors']
            if 'statistics' in extractors:
                research_info.append("📈 Content analysis available")
            if 'entities' in extractors:
                entities_data = extractors['entities']
                entity_count = len(entities_data.get('entities', [])) if isinstance(entities_data, dict) else len(entities_data) if isinstance(entities_data, list) else 0
                research_info.append(f"👥 {entity_count} entities")
            if 'summary' in extractors:
                research_info.append("📝 Summary available")
        
        research_metadata = " | ".join(research_info) if research_info else "No analysis data"
        
        # Content preview focusing on key research elements
        content_preview = doc.content[:400] + "..." if len(doc.content) > 400 else doc.content
        
        source_name = Path(doc.source).name if doc.source else f"Document {i}"
        result_text = f"""[bold]Source:[/bold] {source_name}
[bold]{score_text}[/bold]
[bold]Research Elements:[/bold] {research_metadata}

{content_preview}"""
        
        console.print(Panel(
            result_text,
            title=f"Research Result #{i}",
            title_align="left",
            border_style="green" if i == 1 else "blue",
            expand=False
        ))


def demonstrate_research_paper_rag():
    """Demonstrate RAG system optimized for research paper analysis using strategy-first approach."""
    
    print_section_header("🦙 Demo 1: Research Paper Analysis System", "🔬")
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• [bold cyan]Strategy-first configuration approach[/bold cyan] (no hardcoded parameters!)")
    console.print("• Advanced parsing of academic research papers")
    console.print("• Statistical data extraction from research content")
    console.print("• Named entity recognition for authors, institutions, methods")
    console.print("• Automatic summarization of research findings")
    console.print("• Semantic search optimized for research queries")
    console.print("• Comprehensive analysis of research methodologies")
    
    # Initialize system using strategy
    print_section_header("Strategy-Based System Initialization", "⚙️")
    
    console.print("🚀 [bold cyan]Initializing system with 'research_papers_demo' strategy...[/bold cyan]")
    
    try:
        # Create demo system using strategy - all configuration comes from demo_strategies.yaml!
        rag_system = create_demo_system("research_papers_demo")
        
        # Show strategy information
        rag_system.print_strategy_info()
        
        console.print("\n✅ [bold green]Research analysis system initialized using strategy![/bold green]")
        console.print("[dim]💡 All component configurations loaded from demo_strategies.yaml[/dim]")
        console.print("[dim]🔧 No hardcoded parameters in demo code![/dim]")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to initialize strategy system: {e}[/red]")
        console.print("[dim]💡 Make sure demo_strategies.yaml exists and contains research_papers_demo[/dim]")
        return
    
    # Process research papers using strategy-configured components
    print_section_header("Research Paper Processing", "📚")
    
    paper_files = [
        "demos/static_samples/research_papers/transformer_architecture.txt",
        "demos/static_samples/research_papers/llm_scaling_laws.txt"
    ]
    
    console.print("[bold cyan]🔄 Processing papers using strategy-configured parser and extractors...[/bold cyan]")
    
    # Process documents using strategy system - all parsing and extraction configured in strategy!
    all_documents = rag_system.process_documents(paper_files, show_progress=True)
    
    if not all_documents:
        console.print("❌ [red]No documents were processed. Check file paths.[/red]")
        return
    
    console.print(f"\n✅ Total research sections processed: [bold green]{len(all_documents)}[/bold green]")
    console.print("[dim]💡 Documents processed with strategy-configured parser and extractors[/dim]")
    
    # Add processing timestamps and IDs to documents (demo utility)
    all_documents = add_processing_timestamp(all_documents)
    for doc in all_documents:
        if not doc.id:
            doc.id = generate_document_id(doc.content, doc.source)
    
    display_demo_separator()
    
    # Academic content analysis already done by strategy system!
    print_section_header("Academic Content Analysis", "🔬")
    
    console.print("✅ [bold green]Research content analysis complete![/bold green]")
    console.print("[dim]💡 All extractors were applied automatically by the strategy system[/dim]")
    console.print("[dim]🔧 Extractor configurations loaded from demo_strategies.yaml[/dim]")
    
    # Show detailed extractor results
    print_extractor_details(all_documents)
    
    # Show sample documents with full metadata
    console.print("\n📊 [bold green]Sample Documents with Full Metadata[/bold green]")
    for i, doc in enumerate(all_documents[:2]):
        display_document_with_metadata(doc, i, "Research Paper Section")
    
    display_demo_separator()
    
    # Generate embeddings using strategy-configured embedder
    print_section_header("Research Embedding Generation", "🧠")
    
    console.print("🔄 [bold cyan]Generating embeddings using strategy-configured embedder...[/bold cyan]")
    
    # Show what will be embedded
    display_embedding_process(all_documents, f"Strategy-configured: {rag_system.embedder.__class__.__name__}")
    
    # Generate embeddings using strategy system
    all_documents = rag_system.generate_embeddings(all_documents, show_progress=True)
    
    console.print("[dim]💡 Embedder model and configuration loaded from demo_strategies.yaml[/dim]")
    
    # Store in vector database using strategy-configured store
    print_section_header("Research Database Storage", "🗄️")
    
    console.print("💾 [bold cyan]Storing documents using strategy-configured vector store...[/bold cyan]")
    
    # Store documents using strategy system
    success = rag_system.store_documents(all_documents, show_progress=True)
    if not success:
        console.print("❌ Failed to store research papers")
        return
    
    console.print("[dim]💡 Vector store type and configuration loaded from demo_strategies.yaml[/dim]")
    
    # Demonstrate research queries using strategy-configured search
    print_section_header("Research Query Demonstration", "🔍")
    
    research_queries = [
        "What are the key findings about transformer architecture performance?",
        "How do scaling laws affect neural language model performance?",
        "What statistical evidence supports the scaling law relationships?",
        "Which research methodologies were used to evaluate model performance?",
        "What are the implications of these findings for future AI development?",
        "How do parameter count and dataset size impact model capabilities?"
    ]
    
    console.print("🎯 [bold cyan]Running research-focused queries using strategy-configured search:[/bold cyan]")
    console.print("[dim]💡 Search parameters (top_k, etc.) configured in demo_strategies.yaml[/dim]")
    
    for i, query in enumerate(research_queries, 1):
        console.print(f"\n[bold cyan]Research Query #{i}:[/bold cyan]")
        
        # Search using strategy system - all search configuration from strategy!
        results = rag_system.search(query, show_progress=True)
        
        # Show research-focused results with full metadata
        display_search_results_with_metadata(results, query)
        
        if i < len(research_queries):
            display_demo_separator()
            time.sleep(1.5)  # Pause for readability
    
    # Show research database statistics
    print_section_header("Research Database Analytics", "📊")
    
    info = rag_system.vector_store.get_collection_info()
    if info:
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Research Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Research Collection", info.get("name", "N/A"))
        stats_table.add_row("Total Research Sections", str(info.get("document_count", "N/A")))
        
        # Calculate research-specific metrics
        total_statistics = sum(1 for doc in all_documents if 'extractors' in doc.metadata and 'statistics' in doc.metadata['extractors'])
        total_entities = 0
        papers_with_summaries = 0
        
        for doc in all_documents:
            if 'extractors' in doc.metadata:
                extractors = doc.metadata['extractors']
                if 'entities' in extractors:
                    entities_data = extractors['entities']
                    if isinstance(entities_data, dict) and 'entities' in entities_data:
                        total_entities += len(entities_data['entities'])
                    elif isinstance(entities_data, list):
                        total_entities += len(entities_data)
                if 'summary' in extractors:
                    papers_with_summaries += 1
        
        stats_table.add_row("Statistical Data Points", str(total_statistics))
        stats_table.add_row("Named Entities", str(total_entities))
        stats_table.add_row("Auto-Generated Summaries", str(papers_with_summaries))
        stats_table.add_row("Strategy Used", rag_system.strategy_name)
        stats_table.add_row("Embedding Model", getattr(rag_system.embedder, 'model', 'N/A'))
        
        console.print(stats_table)
    
    # Research insights summary
    print_section_header("Research System Summary", "🎓")
    
    console.print("🚀 [bold green]Research Paper Analysis Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ [bold cyan]Strategy-first configuration approach[/bold cyan] - no hardcoded parameters!")
    console.print("✅ Advanced parsing of complex academic content")
    console.print("✅ Statistical data extraction from research papers")
    console.print("✅ Named entity recognition for research elements") 
    console.print("✅ Automatic summarization of research findings")
    console.print("✅ Semantic search optimized for academic queries")
    console.print("✅ Comprehensive analysis suitable for literature reviews")
    
    console.print(f"\n[bold]Why the strategy-first approach is powerful:[/bold]")
    console.print("🔧 [bold cyan]All configuration in demo_strategies.yaml[/bold cyan] - easy to modify")
    console.print("⚡ Quick setup with predefined, optimized configurations")
    console.print("🔄 Easy to experiment with different settings")
    console.print("📋 Clean separation of configuration from implementation")
    console.print("🎯 Domain-specific optimizations built into strategies")
    
    console.print(f"\n[bold]Traditional research benefits:[/bold]")
    console.print("📈 Extracts quantitative data automatically")
    console.print("👥 Identifies key researchers and institutions")
    console.print("📝 Generates concise summaries of complex papers")
    console.print("🔍 Enables semantic search across research corpus")
    
    console.print(f"\n📁 Research database saved to: [bold]./demos/vectordb/research_papers[/bold]")
    console.print("🔄 You can now query this research database using the CLI with strategies:")
    console.print("[dim]uv run python cli.py search 'scaling laws neural networks' --strategy research_papers_demo[/dim]")
    
    # Clean up database to prevent duplicate accumulation
    print_section_header("Database Cleanup", "🧹")
    rag_system.cleanup(show_progress=True)


if __name__ == "__main__":
    try:
        demonstrate_research_paper_rag()
    except KeyboardInterrupt:
        console.print("\n\n👋 Research demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Research demo failed: {str(e)}", style="red")
        console.print("Check that Ollama is running with the nomic-embed-text model")
        sys.exit(1)