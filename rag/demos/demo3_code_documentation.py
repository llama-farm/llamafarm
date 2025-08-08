#!/usr/bin/env python3
"""
Demo 3: Code Documentation Analysis System
Demonstrates RAG capabilities for software documentation using:
- Strategy-first configuration approach (no hardcoded parameters!)
- Markdown parser for technical documentation
- Code extraction from documentation and examples
- Link extraction for API references and cross-references
- Heading-based chunking for structured content navigation
- Optimized search for developers and technical users

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


def print_section_header(title: str, emoji: str = "📚"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def extract_code_blocks(content: str) -> tuple:
    """Extract code blocks from markdown content."""
    import re
    
    # Pattern for fenced code blocks
    code_pattern = r'```(\w+)?\n(.*?)```'
    matches = re.findall(code_pattern, content, re.DOTALL)
    
    code_blocks = []
    for language, code in matches:
        code_blocks.append({
            'language': language if language else 'text',
            'code': code.strip(),
            'lines': len(code.strip().split('\n'))
        })
    
    # Pattern for inline code
    inline_pattern = r'`([^`]+)`'
    inline_matches = re.findall(inline_pattern, content)
    
    return code_blocks, inline_matches


def print_code_documentation_insights(documents: List[Document]):
    """Print insights specific to code documentation."""
    console.print("\n📚 [bold green]Code Documentation Intelligence[/bold green]")
    
    # Aggregate documentation metrics
    total_links = 0
    total_headings = 0
    total_code_blocks = 0
    languages_used = set()
    heading_levels = {}
    link_types = {}
    
    for doc in documents:
        # Count code blocks and extract languages
        code_blocks, inline_code = extract_code_blocks(doc.content)
        doc.metadata['code_blocks'] = len(code_blocks)
        doc.metadata['inline_code'] = len(inline_code)
        
        for block in code_blocks:
            if block['language']:
                languages_used.add(block['language'])
        
        total_code_blocks += len(code_blocks)
        
        # Process extractor results
        if 'extractors' in doc.metadata:
            extractors = doc.metadata['extractors']
            
            # Process heading extractor results
            if 'headings' in extractors:
                headings_data = extractors['headings']
                if isinstance(headings_data, dict) and 'headings' in headings_data:
                    for heading in headings_data['headings']:
                        level = heading.get('level', 1)
                        heading_levels[level] = heading_levels.get(level, 0) + 1
                        total_headings += 1
            
            # Process path/link extractor results
            if 'paths' in extractors:
                paths_data = extractors['paths']
                if isinstance(paths_data, dict):
                    if 'file_paths' in paths_data:
                        total_links += len(paths_data['file_paths'])
                    if 'urls' in paths_data:
                        total_links += len(paths_data['urls'])
            
            # Process keyword extractor results
            if 'yake_keywords' in extractors:
                keywords_data = extractors['yake_keywords']
                # Keywords are stored in metadata
    
    # Display aggregated insights
    insights_table = Table(show_header=True, header_style="bold magenta")
    insights_table.add_column("Documentation Metric", style="cyan")
    insights_table.add_column("Value", style="white")
    
    insights_table.add_row("Total Sections", str(len(documents)))
    insights_table.add_row("Code Blocks", str(total_code_blocks))
    insights_table.add_row("Programming Languages", ", ".join(sorted(languages_used)) if languages_used else "None")
    insights_table.add_row("Total Headings", str(total_headings))
    insights_table.add_row("Links & References", str(total_links))
    
    # Add heading distribution
    if heading_levels:
        heading_dist = " | ".join([f"H{level}: {count}" for level, count in sorted(heading_levels.items())])
        insights_table.add_row("Heading Distribution", heading_dist)
    
    console.print(insights_table)
    
    # Show sample code blocks
    if total_code_blocks > 0:
        console.print("\n🔍 [bold yellow]Sample Code Blocks Found:[/bold yellow]")
        shown = 0
        for doc in documents:
            code_blocks, _ = extract_code_blocks(doc.content)
            for block in code_blocks[:2]:  # Show first 2 blocks per doc
                if shown >= 3:  # Limit total shown
                    break
                console.print(f"\n[bold]Language:[/bold] {block['language']} ({block['lines']} lines)")
                # Show first 5 lines of code
                code_preview = '\n'.join(block['code'].split('\n')[:5])
                if block['lines'] > 5:
                    code_preview += '\n...'
                syntax = Syntax(code_preview, block['language'], theme="monokai", line_numbers=False)
                console.print(syntax)
                shown += 1
            if shown >= 3:
                break


def print_documentation_search_results(query: str, results: List[Document]):
    """Print search results with documentation-specific formatting."""
    console.print(f"\n🔍 Query: [bold yellow]'{query}'[/bold yellow]")
    console.print(f"📊 Found {len(results)} relevant documentation sections")
    
    for i, doc in enumerate(results[:3], 1):  # Show top 3 results
        score = doc.metadata.get('search_score', 'N/A')
        score_text = f"Relevance: {score:.4f}" if isinstance(score, float) else f"Relevance: {score}"
        
        # Extract documentation-specific metadata
        doc_info = []
        if 'extractors' in doc.metadata:
            extractors = doc.metadata['extractors']
            
            # Check for code blocks
            code_blocks, _ = extract_code_blocks(doc.content)
            if code_blocks:
                doc_info.append(f"💻 {len(code_blocks)} code examples")
            
            # Check for headings
            if 'headings' in extractors:
                headings_data = extractors['headings']
                if isinstance(headings_data, dict) and 'headings' in headings_data:
                    doc_info.append(f"📑 {len(headings_data['headings'])} sections")
            
            # Check for keywords (YAKE extractor stores as yake_keywords)
            if 'yake_keywords' in extractors:
                keywords_data = extractors['yake_keywords']
                if keywords_data and isinstance(keywords_data, list):
                    top_keywords = keywords_data[:3]
                    if top_keywords:
                        # Handle both dict and string formats
                        if isinstance(top_keywords[0], dict):
                            keywords_str = ", ".join([kw.get('phrase', str(kw)) for kw in top_keywords])
                        else:
                            keywords_str = ", ".join([str(kw) for kw in top_keywords])
                        doc_info.append(f"🏷️ {keywords_str}")
        
        doc_metadata = " | ".join(doc_info) if doc_info else "No analysis data"
        
        # Content preview with code highlighting
        content_preview = doc.content[:400]
        if '```' in content_preview:
            # Truncate at code block boundary if present
            code_start = content_preview.find('```')
            if code_start > 100:
                content_preview = content_preview[:code_start] + "\n[Code block follows...]"
        content_preview = content_preview + "..." if len(doc.content) > 400 else content_preview
        
        source_name = Path(doc.source).name if doc.source else f"Section {i}"
        result_text = f"""[bold]Source:[/bold] {source_name}
[bold]{score_text}[/bold]
[bold]Documentation Elements:[/bold] {doc_metadata}

{content_preview}"""
        
        console.print(Panel(
            result_text,
            title=f"Documentation Result #{i}",
            title_align="left",
            border_style="green" if i == 1 else "blue",
            expand=False
        ))


def demonstrate_code_documentation_rag():
    """Demonstrate RAG system optimized for code documentation using strategy-first approach."""
    
    print_section_header("🦙 Demo 3: Code Documentation Analysis System", "💻")
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• [bold cyan]Strategy-first configuration approach[/bold cyan] (no hardcoded parameters!)")
    console.print("• Advanced parsing of Markdown technical documentation")
    console.print("• Code block extraction and language identification")
    console.print("• Link extraction for API references and cross-references")
    console.print("• Heading-based content organization and navigation")
    console.print("• Pattern recognition for code examples and best practices")
    console.print("• Developer-optimized search for technical content")
    
    # Initialize system using strategy
    print_section_header("Strategy-Based System Initialization", "⚙️")
    
    console.print("🚀 [bold cyan]Initializing system with 'code_documentation_demo' strategy...[/bold cyan]")
    
    try:
        # Create demo system using strategy - all configuration comes from demo_strategies.yaml!
        rag_system = create_demo_system("code_documentation_demo")
        
        # Show strategy information
        rag_system.print_strategy_info()
        
        console.print("\n✅ [bold green]Code documentation system initialized using strategy![/bold green]")
        console.print("[dim]💡 All component configurations loaded from demo_strategies.yaml[/dim]")
        console.print("[dim]🔧 No hardcoded parameters in demo code![/dim]")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to initialize strategy system: {e}[/red]")
        console.print("[dim]💡 Make sure demo_strategies.yaml exists and contains code_documentation_demo[/dim]")
        return
    
    # Process documentation files using strategy-configured components
    print_section_header("Documentation Processing", "📂")
    
    doc_files = [
        "demos/static_samples/code_documentation/README.md",
        "demos/static_samples/code_documentation/API_REFERENCE.md"
    ]
    
    console.print("[bold cyan]🔄 Processing documentation using strategy-configured parser and extractors...[/bold cyan]")
    
    # Process documents using strategy system - all parsing and extraction configured in strategy!
    all_documents = rag_system.process_documents(doc_files, show_progress=True)
    
    if not all_documents:
        console.print("❌ [red]No documents were processed. Check file paths.[/red]")
        return
    
    console.print(f"\n✅ Total documentation sections processed: [bold green]{len(all_documents)}[/bold green]")
    console.print("[dim]💡 Documents processed with strategy-configured parser and extractors[/dim]")
    
    # Add processing timestamps and IDs to documents (demo utility)
    all_documents = add_processing_timestamp(all_documents)
    for doc in all_documents:
        if not doc.id:
            doc.id = generate_document_id(doc.content, doc.source)
    
    display_demo_separator()
    
    # Documentation intelligence already extracted by strategy system!
    print_section_header("Documentation Intelligence", "🔍")
    
    console.print("✅ [bold green]Documentation analysis complete![/bold green]")
    console.print("[dim]💡 All extractors were applied automatically by the strategy system[/dim]")
    console.print("[dim]🔧 Extractor configurations loaded from demo_strategies.yaml[/dim]")
    
    # Show documentation-specific insights
    print_code_documentation_insights(all_documents)
    
    # Show sample documents with full metadata
    console.print("\n📊 [bold green]Sample Documentation Sections[/bold green]")
    for i, doc in enumerate(all_documents[:2]):
        display_document_with_metadata(doc, i, "Documentation Section")
    
    display_demo_separator()
    
    # Generate embeddings using strategy-configured embedder
    print_section_header("Documentation Embedding Generation", "🧠")
    
    console.print("🔄 [bold cyan]Generating embeddings using strategy-configured embedder...[/bold cyan]")
    
    # Show what will be embedded
    display_embedding_process(all_documents, f"Strategy-configured: {rag_system.embedder.__class__.__name__}")
    
    # Generate embeddings using strategy system
    all_documents = rag_system.generate_embeddings(all_documents, show_progress=True)
    
    console.print("[dim]💡 Embedder model and configuration loaded from demo_strategies.yaml[/dim]")
    
    # Store in vector database using strategy-configured store
    print_section_header("Documentation Database Storage", "🗄️")
    
    console.print("💾 [bold cyan]Storing documentation using strategy-configured vector store...[/bold cyan]")
    
    # Store documents using strategy system
    success = rag_system.store_documents(all_documents, show_progress=True)
    if not success:
        console.print("❌ Failed to store documentation")
        return
    
    console.print("[dim]💡 Vector store type and configuration loaded from demo_strategies.yaml[/dim]")
    
    # Demonstrate developer-focused queries using strategy-configured search
    print_section_header("Developer Query Demonstration", "🔍")
    
    dev_queries = [
        "How do I authenticate with the API?",
        "What are the rate limits for API requests?",
        "Show me examples of error handling",
        "How to configure logging in the system?",
        "What are the best practices for performance optimization?",
        "How do I contribute to this project?"
    ]
    
    console.print("🎯 [bold cyan]Running developer-focused queries using strategy-configured search:[/bold cyan]")
    console.print("[dim]💡 Search parameters (top_k, etc.) configured in demo_strategies.yaml[/dim]")
    
    for i, query in enumerate(dev_queries, 1):
        console.print(f"\n[bold cyan]Developer Query #{i}:[/bold cyan]")
        
        # Search using strategy system - all search configuration from strategy!
        results = rag_system.search(query, show_progress=True)
        
        # Show documentation-focused results
        print_documentation_search_results(query, results)
        
        if i < len(dev_queries):
            display_demo_separator()
            time.sleep(1.5)  # Pause for readability
    
    # Show documentation database statistics
    print_section_header("Documentation Database Analytics", "📊")
    
    info = rag_system.vector_store.get_collection_info()
    if info:
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Documentation Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Documentation Collection", info.get("name", "N/A"))
        stats_table.add_row("Total Documentation Sections", str(info.get("document_count", "N/A")))
        
        # Calculate documentation-specific metrics
        total_code_blocks = sum(doc.metadata.get('code_blocks', 0) for doc in all_documents)
        total_with_headings = sum(1 for doc in all_documents if 'extractors' in doc.metadata and 'headings' in doc.metadata['extractors'])
        total_with_keywords = sum(1 for doc in all_documents if 'extractors' in doc.metadata and 'keywords' in doc.metadata['extractors'])
        
        stats_table.add_row("Total Code Examples", str(total_code_blocks))
        stats_table.add_row("Sections with Headings", str(total_with_headings))
        stats_table.add_row("Sections with Keywords", str(total_with_keywords))
        stats_table.add_row("Strategy Used", rag_system.strategy_name)
        stats_table.add_row("Embedding Model", getattr(rag_system.embedder, 'model', 'N/A'))
        
        console.print(stats_table)
    
    # Documentation system summary
    print_section_header("Documentation System Summary", "🎓")
    
    console.print("🚀 [bold green]Code Documentation Analysis Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ [bold cyan]Strategy-first configuration approach[/bold cyan] - no hardcoded parameters!")
    console.print("✅ Advanced Markdown documentation parsing")
    console.print("✅ Code block extraction and language detection")
    console.print("✅ Link and reference extraction for API docs")
    console.print("✅ Heading-based document organization")
    console.print("✅ Developer-optimized search capabilities")
    console.print("✅ Comprehensive documentation indexing")
    
    console.print(f"\n[bold]Why the strategy-first approach is powerful:[/bold]")
    console.print("🔧 [bold cyan]All configuration in demo_strategies.yaml[/bold cyan] - easy to modify")
    console.print("⚡ Quick setup with predefined, optimized configurations")
    console.print("🔄 Easy to experiment with different settings")
    console.print("📋 Clean separation of configuration from implementation")
    console.print("🎯 Domain-specific optimizations built into strategies")
    
    console.print(f"\n[bold]Documentation benefits:[/bold]")
    console.print("📚 Structured navigation through documentation")
    console.print("💻 Code examples readily searchable")
    console.print("🔗 Cross-references and links preserved")
    console.print("🏷️ Automatic keyword and topic extraction")
    
    console.print(f"\n📁 Documentation database saved to: [bold]./demos/vectordb/code_documentation[/bold]")
    console.print("🔄 You can now query this documentation database using the CLI with strategies:")
    console.print("[dim]uv run python cli.py --strategy-file demos/demo_strategies.yaml search 'API authentication' --strategy code_documentation_demo[/dim]")
    
    # Clean up database to prevent duplicate accumulation
    print_section_header("Database Cleanup", "🧹")
    rag_system.cleanup(show_progress=True)


if __name__ == "__main__":
    try:
        demonstrate_code_documentation_rag()
    except KeyboardInterrupt:
        console.print("\n\n👋 Documentation demo interrupted by user", style="yellow")
    except Exception as e:
        console.print(f"\n\n❌ Documentation demo failed: {str(e)}", style="red")
        console.print("Check that Ollama is running with the nomic-embed-text model")
        sys.exit(1)