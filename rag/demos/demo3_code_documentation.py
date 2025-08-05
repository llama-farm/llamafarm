#!/usr/bin/env python3
"""
Demo 3: Code Documentation Analysis System
Demonstrates RAG capabilities for software documentation using:
- Markdown parser for technical documentation
- Code extraction from documentation and examples
- Link extraction for API references and cross-references
- Heading-based chunking for structured content navigation
- Optimized search for developers and technical users
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

# Import RAG components
from core.base import Document, Pipeline
from components.parsers.markdown_parser.markdown_parser import MarkdownParser
from components.extractors.link_extractor.link_extractor import LinkExtractor
from components.extractors.heading_extractor.heading_extractor import HeadingExtractor
from components.extractors.pattern_extractor.pattern_extractor import PatternExtractor
from components.extractors.path_extractor.path_extractor import PathExtractor
from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder
from components.stores.chroma_store.chroma_store import ChromaStore

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


def print_section_header(title: str, emoji: str = "💻"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def print_code_doc_analysis(doc: Document):
    """Print analysis of a code documentation section."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Documentation Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    # Basic document info
    table.add_row("Document ID", doc.id)
    table.add_row("Source", Path(doc.source).name if doc.source else "Code Document")
    table.add_row("Content Length", f"{len(doc.content):,} characters")
    table.add_row("Word Count", f"{len(doc.content.split()):,} words")
    
    # Documentation-specific metadata
    if doc.metadata:
        if 'heading' in doc.metadata:
            table.add_row("Section Heading", doc.metadata['heading'])
        if 'heading_level' in doc.metadata:
            table.add_row("Heading Level", f"H{doc.metadata['heading_level']}")
        if 'section_index' in doc.metadata:
            table.add_row("Section Index", str(doc.metadata['section_index']))
        
        # Extractor results
        if 'links' in doc.metadata:
            links = len(doc.metadata['links'])
            table.add_row("Links Found", f"{links} references")
        
        if 'headings' in doc.metadata:
            headings = len(doc.metadata['headings'])
            table.add_row("Sub-headings", f"{headings} nested sections")
        
        if 'code_blocks' in doc.metadata:
            code_blocks = doc.metadata['code_blocks']
            table.add_row("Code Examples", f"{code_blocks} code blocks")
        
        if 'patterns' in doc.metadata:
            patterns = len(doc.metadata['patterns'])
            table.add_row("Code Patterns", f"{patterns} identified patterns")
    
    console.print(table)


def extract_code_blocks(content: str) -> List[Dict[str, Any]]:
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
        if 'links' in doc.metadata:
            total_links += len(doc.metadata['links'])
            for link in doc.metadata['links']:
                # Handle both string links and dict links
                if isinstance(link, dict):
                    link_url = link.get('url', '')
                else:
                    link_url = str(link)
                link_type = 'external' if link_url.startswith('http') else 'internal'
                link_types[link_type] = link_types.get(link_type, 0) + 1
        
        if 'headings' in doc.metadata:
            total_headings += len(doc.metadata['headings'])
            for heading in doc.metadata['headings']:
                level = heading.get('level', 1)
                heading_levels[f"H{level}"] = heading_levels.get(f"H{level}", 0) + 1
    
    # Display documentation structure metrics
    structure_table = Table(title="📖 Documentation Structure", show_header=True, header_style="bold yellow")
    structure_table.add_column("Metric", style="cyan")
    structure_table.add_column("Count", style="white")
    structure_table.add_column("Details", style="dim")
    
    structure_table.add_row("Total Sections", str(len(documents)), "Parsed documentation sections")
    structure_table.add_row("Code Examples", str(total_code_blocks), f"Across {len(languages_used)} languages")
    structure_table.add_row("Cross References", str(total_links), f"Internal: {link_types.get('internal', 0)}, External: {link_types.get('external', 0)}")
    structure_table.add_row("Nested Headings", str(total_headings), "Sub-sections and organization")
    
    console.print(structure_table)
    
    # Display programming languages found
    if languages_used:
        languages_table = Table(title="⌨️ Programming Languages in Documentation", show_header=True, header_style="bold green")
        languages_table.add_column("Language", style="cyan")
        languages_table.add_column("Usage Context", style="white")
        
        language_contexts = {
            'python': 'API examples, implementation code',
            'javascript': 'Frontend examples, client code',
            'bash': 'CLI commands, shell scripts',
            'json': 'Configuration, API responses',
            'yaml': 'Configuration files, CI/CD',
            'sql': 'Database queries, schema',
            'html': 'Web examples, templates',
            'css': 'Styling examples',
            'dockerfile': 'Container configuration',
            'typescript': 'Type definitions, advanced examples'
        }
        
        for lang in sorted(languages_used):
            context = language_contexts.get(lang.lower(), 'Code examples and snippets')
            languages_table.add_row(lang.capitalize(), context)
        
        console.print(languages_table)
    
    # Display heading hierarchy
    if heading_levels:
        hierarchy_table = Table(title="🗂️ Content Hierarchy", show_header=True, header_style="bold blue")
        hierarchy_table.add_column("Heading Level", style="cyan")
        hierarchy_table.add_column("Count", style="white")
        hierarchy_table.add_column("Typical Use", style="dim")
        
        level_descriptions = {
            'H1': 'Main sections, major topics',
            'H2': 'Subsections, key concepts',
            'H3': 'Detailed topics, methods',
            'H4': 'Specific implementations, parameters',
            'H5': 'Fine details, edge cases',
            'H6': 'Notes, additional information'
        }
        
        for level in ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']:
            if level in heading_levels:
                count = heading_levels[level]
                description = level_descriptions.get(level, 'Documentation structure')
                hierarchy_table.add_row(level, str(count), description)
        
        console.print(hierarchy_table)


def print_code_search_results(query: str, results: List[Document]):
    """Print search results optimized for code documentation."""
    console.print(f"\n🔍 Documentation Query: [bold yellow]'{query}'[/bold yellow]")
    console.print(f"📚 Found {len(results)} relevant documentation sections")
    
    for i, doc in enumerate(results[:3], 1):
        score = doc.metadata.get('search_score', 'N/A')
        score_text = f"Relevance: {score:.4f}" if isinstance(score, float) else f"Relevance: {score}"
        
        # Show documentation-specific metadata
        doc_info = []
        if 'heading' in doc.metadata:
            doc_info.append(f"📂 {doc.metadata['heading']}")
        
        if 'heading_level' in doc.metadata:
            level = doc.metadata['heading_level']
            doc_info.append(f"📊 Level H{level}")
        
        if 'code_blocks' in doc.metadata and doc.metadata['code_blocks'] > 0:
            doc_info.append(f"💻 {doc.metadata['code_blocks']} code examples")
        
        if 'links' in doc.metadata and len(doc.metadata['links']) > 0:
            doc_info.append(f"🔗 {len(doc.metadata['links'])} references")
        
        doc_metadata = " | ".join(doc_info) if doc_info else "No metadata"
        
        # Content preview focusing on code and technical content
        content_preview = doc.content[:400] + "..." if len(doc.content) > 400 else doc.content
        
        # Highlight code blocks in preview
        if '```' in content_preview:
            content_preview = content_preview.replace('```', '[bold cyan]```[/bold cyan]')
        
        result_text = f"""[bold]Source:[/bold] {Path(doc.source).name if doc.source else "Code Document"}
[bold]{score_text}[/bold]
[bold]Documentation:[/bold] {doc_metadata}

{content_preview}"""
        
        console.print(Panel(
            result_text,
            title=f"Documentation Result #{i}",
            title_align="left",
            border_style="green" if i == 1 else "blue",
            expand=False
        ))


def demonstrate_code_documentation_rag():
    """Demonstrate RAG system optimized for code documentation."""
    
    print_section_header("🦙 Demo 3: Code Documentation Analysis System", "💻")
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• Advanced parsing of Markdown technical documentation")
    console.print("• Code block extraction and language identification")
    console.print("• Link extraction for API references and cross-references")
    console.print("• Heading-based content organization and navigation")
    console.print("• Pattern recognition for code examples and best practices")
    console.print("• Developer-optimized search for technical content")
    
    # Initialize components
    print_section_header("Documentation System Initialization", "⚙️")
    
    console.print("🔧 Initializing Markdown documentation parser...")
    parser = MarkdownParser(config={
        "extract_headers": True,        # Extract headers
        "extract_links": True,          # Extract links
        "extract_code_blocks": True,    # Extract code blocks
        "preserve_structure": True      # Preserve document structure
    })
    
    console.print("🔗 Setting up documentation extractors...")
    # Path extractor to preserve source information
    path_extractor = PathExtractor("path_extractor", {"store_full_path": True, "store_filename": True, "store_directory": True, "store_extension": True})
    
    # Link extractor for cross-references and API links
    link_extractor = LinkExtractor({"include_external": True, "include_internal": True, "include_anchors": True})
    
    # Heading extractor for content structure
    heading_extractor = HeadingExtractor({"extract_hierarchy": True, "include_level": True, "min_level": 1, "max_level": 6})
    
    # Pattern extractor for code patterns and examples
    pattern_extractor = PatternExtractor({"patterns": ["email", "phone", "url", "ip_address"], "include_context": True})
    
    console.print("🧠 Initializing technical embedder...")
    embedder = OllamaEmbedder("docs_embedder", {
        "model": "nomic-embed-text",
        "batch_size": 3  # Smaller batches for larger technical documents
    })
    
    console.print("🗄️ Setting up documentation knowledge store...")
    vector_store = ChromaStore("code_documentation_store", {
        "collection_name": "code_documentation",
        "persist_directory": "./demos/vectordb/code_documentation"
    })
    
    console.print("✅ Code documentation system initialized!")
    
    # Process documentation files
    print_section_header("Documentation Processing", "📚")
    
    doc_files = [
        "demos/static_samples/code_documentation/api_reference.md",
        "demos/static_samples/code_documentation/implementation_guide.md",
        "demos/static_samples/code_documentation/best_practices.md"
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
        
        parse_task = progress.add_task("Processing documentation...", total=len(doc_files))
        
        for file_path in doc_files:
            if not Path(file_path).exists():
                console.print(f"⚠️ Documentation file not found: {file_path}", style="yellow")
                progress.advance(parse_task)
                continue
            
            result = parser.parse(file_path)
            # Handle both ProcessingResult and List[Document] return types
            if hasattr(result, 'documents'):
                documents = result.documents
            else:
                documents = result
            
            # Set source file path for each document
            for doc in documents:
                doc.source = file_path
            
            all_documents.extend(documents)
            
            console.print(f"📄 Processed [bold]{Path(file_path).name}[/bold]: {len(documents)} section(s)")
            progress.advance(parse_task)
    
    console.print(f"\n✅ Total documentation sections processed: [bold green]{len(all_documents)}[/bold green]")
    
    # Add processing timestamps and IDs to documents
    all_documents = add_processing_timestamp(all_documents)
    for doc in all_documents:
        if not doc.id:
            doc.id = generate_document_id(doc.content, doc.source)
    
    display_demo_separator()
    
    # Apply documentation extractors
    print_section_header("Technical Content Analysis", "🔍")
    
    console.print("🔬 Analyzing documentation with specialized extractors...")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        extract_task = progress.add_task("Extracting technical elements...", total=len(all_documents) * 4)
        
        # Apply path extractor first to preserve source information
        all_documents = path_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply link extractor
        all_documents = link_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply heading extractor
        all_documents = heading_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
        
        # Apply pattern extractor
        all_documents = pattern_extractor.extract(all_documents)
        progress.advance(extract_task, len(all_documents))
    
    console.print("✅ Technical content analysis complete!")
    
    # Show documentation insights
    print_code_documentation_insights(all_documents)
    
    # Show sample documents with full metadata
    console.print("\n📊 [bold green]Sample Documents with Full Metadata[/bold green]")
    for i, doc in enumerate(all_documents[:2]):
        display_document_with_metadata(doc, i, "Documentation Section")
    
    display_demo_separator()
    
    # Show detailed analysis of first few sections
    console.print("\n📖 [bold green]Sample Documentation Analysis[/bold green]")
    for i, doc in enumerate(all_documents[:3], 1):
        console.print(f"\n📄 Documentation Section #{i}:")
        print_code_doc_analysis(doc)
    
    # Generate embeddings
    print_section_header("Documentation Embedding Generation", "🧠")
    
    console.print("🔄 Generating embeddings for technical documentation...")
    
    # Show what will be embedded
    display_embedding_process(all_documents, "Ollama (nomic-embed-text)")
    
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
    
    console.print(f"✅ Generated embeddings for {len(all_documents)} documentation sections")
    
    # Store in vector database
    print_section_header("Documentation Knowledge Base Storage", "🗄️")
    
    console.print("💾 Building searchable documentation knowledge base...")
    success = vector_store.add_documents(all_documents)
    if success:
        console.print(f"✅ Stored {len(all_documents)} documentation sections in knowledge base")
    else:
        console.print("❌ Failed to store documentation")
        return
    
    # Demonstrate documentation queries
    print_section_header("Documentation Query Demonstration", "🔍")
    
    dev_queries = [
        "How to implement a custom parser class with error handling?",
        "What are the best practices for RAG system architecture?",
        "Show me examples of embedder configuration and usage",
        "How to set up vector store with ChromaDB persistence?",
        "What testing strategies are recommended for RAG components?",
        "How to implement retry logic with exponential backoff?",
        "What are the security considerations for RAG systems?",
        "How to use factories for component initialization?",
        "Performance optimization techniques for large documents"
    ]
    
    console.print("🎯 Running developer queries to demonstrate technical documentation search:")
    
    for i, query in enumerate(dev_queries, 1):
        console.print(f"\n[bold cyan]Developer Query #{i}:[/bold cyan]")
        
        # Generate query embedding
        console.print("🧠 Analyzing technical query...")
        query_embeddings = embedder.embed([query])
        query_embedding = query_embeddings[0] if query_embeddings else []
        
        # Search for relevant documentation
        console.print("🔍 Searching documentation knowledge base...")
        results = vector_store.search(query_embedding=query_embedding, top_k=3)
        
        # Show documentation-focused results with full metadata
        display_search_results_with_metadata(results, query)
        
        if i < len(dev_queries):
            display_demo_separator()
            time.sleep(1.3)  # Pause for readability
    
    # Show documentation system statistics
    print_section_header("Documentation System Analytics", "📊")
    
    info = vector_store.get_collection_info()
    if info:
        # Calculate documentation-specific metrics
        total_code_blocks = sum(doc.metadata.get('code_blocks', 0) for doc in all_documents)
        total_links = sum(len(doc.metadata.get('links', [])) for doc in all_documents)
        total_headings = sum(len(doc.metadata.get('headings', [])) for doc in all_documents)
        total_patterns = sum(len(doc.metadata.get('patterns', [])) for doc in all_documents)
        
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Documentation Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Documentation Collection", info.get("name", "N/A"))
        stats_table.add_row("Total Sections", str(info.get("document_count", "N/A")))
        stats_table.add_row("Code Examples", str(total_code_blocks))
        stats_table.add_row("Cross References", str(total_links))
        stats_table.add_row("Content Headings", str(total_headings))
        stats_table.add_row("Code Patterns", str(total_patterns))
        stats_table.add_row("Embedding Model", embedder.model)
        
        console.print(stats_table)
    
    # Documentation system summary
    print_section_header("Documentation System Summary", "🎉")
    
    console.print("🚀 [bold green]Code Documentation Analysis Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ Advanced Markdown parsing with structure preservation")
    console.print("✅ Code block extraction and language identification")
    console.print("✅ Link extraction for API references and navigation")
    console.print("✅ Heading-based content organization")
    console.print("✅ Pattern recognition for code examples and best practices")
    console.print("✅ Developer-optimized semantic search")
    
    console.print(f"\n[bold]Why this approach is powerful for developers:[/bold]")
    console.print("💻 Preserves code formatting and examples")
    console.print("🔗 Maintains cross-reference relationships")
    console.print("📂 Respects documentation hierarchy")
    console.print("🎯 Optimized for technical terminology")
    console.print("🔍 Context-aware code pattern matching")
    
    console.print(f"\n📁 Documentation knowledge base saved to: [bold]./demos/vectordb/code_documentation[/bold]")
    console.print("🔄 You can now query this documentation using the CLI:")
    console.print("[dim]uv run python cli.py search 'parser implementation examples' --collection code_documentation[/dim]")

    # Clean up database to prevent duplicate accumulation
    print_section_header("Database Cleanup", "🧹")
    console.print("🗑️  Cleaning up vector database to prevent duplicate accumulation...")
    try:
        # Delete the collection to clean up
        vector_store.delete_collection()
        console.print("✅ [green]Database cleaned successfully![/green]")
        console.print("[dim]The database has been reset to prevent duplicate data accumulation in future runs.[/dim]")
    except Exception as e:
        console.print(f"⚠️  [yellow]Note: Could not clean database: {e}[/yellow]")
        console.print("[dim]You may want to manually clean the vector database directory.[/dim]")



if __name__ == "__main__":
    try:
        demonstrate_code_documentation_rag()
    except KeyboardInterrupt:
        console.print("\n\n👋 Documentation demo interrupted by user", style="yellow")
    except Exception as e:
        import traceback
        console.print(f"\n\n❌ Documentation demo failed: {str(e)}", style="red")
        console.print("Check that Ollama is running with the nomic-embed-text model")
        console.print("\nDebug traceback:")
        traceback.print_exc()
        sys.exit(1)