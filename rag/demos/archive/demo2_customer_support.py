#!/usr/bin/env python3
"""
Demo 2: Customer Support Knowledge System
Demonstrates RAG capabilities for customer support scenarios using:
- Strategy-first configuration approach (no hardcoded parameters!)
- CSV parser for structured support ticket data
- Text parser for knowledge base articles
- Entity extraction for customer names, issue types, products
- Pattern extraction for common issue resolution workflows
- Optimized search for support case similarity and knowledge retrieval

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

# Setup rich console for beautiful output
console = Console()
logging.basicConfig(level=logging.WARNING)  # Reduce noise


def print_section_header(title: str, emoji: str = "🎧"):
    """Print a beautiful section header."""
    console.print(f"\n{emoji} {title} {emoji}", style="bold cyan", justify="center")
    console.print("=" * 80, style="cyan")


def print_support_ticket_analysis(doc: Document):
    """Print analysis of a support ticket with extracted information."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Ticket Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")
    
    # Basic document info
    table.add_row("Ticket ID", doc.id)
    table.add_row("Source", Path(doc.source).name if doc.source else "Support Document")
    table.add_row("Content Length", f"{len(doc.content):,} characters")
    
    # Support-specific metadata
    if doc.metadata:
        if 'customer_name' in doc.metadata:
            table.add_row("Customer", doc.metadata['customer_name'])
        if 'priority' in doc.metadata:
            table.add_row("Priority", doc.metadata['priority'])
        if 'category' in doc.metadata:
            table.add_row("Category", doc.metadata['category'])
        if 'status' in doc.metadata:
            table.add_row("Status", doc.metadata['status'])
        
        # Extractor results
        if 'entities' in doc.metadata:
            entities = len(doc.metadata.get('extractors', {}).get('entities', {}))
            table.add_row("Entities Found", f"{entities} named entities")
        
        if 'patterns' in doc.metadata:
            patterns = len(doc.metadata['patterns'])
            table.add_row("Patterns Detected", f"{patterns} resolution patterns")
        
        if 'summary' in doc.metadata:
            summary = doc.metadata.get('extractors', {}).get('summary', {})[:80] + "..." if len(doc.metadata.get('extractors', {}).get('summary', {})) > 80 else doc.metadata.get('extractors', {}).get('summary', {})
            table.add_row("Issue Summary", summary)
    
    console.print(table)


def print_extractor_insights(documents: List[Document]):
    """Print detailed insights from support ticket analysis."""
    console.print("\n🎯 [bold green]Customer Support Intelligence[/bold green]")
    
    # Aggregate insights across all tickets
    all_entities = {}
    all_patterns = []
    priority_counts = {}
    category_counts = {}
    status_counts = {}
    
    for doc in documents:
        if 'entities' in doc.metadata:
            for entity in doc.metadata.get('extractors', {}).get('entities', {}):
                entity_type = entity.get('label', 'Unknown')
                entity_text = entity.get('text', '').lower()
                if entity_type not in all_entities:
                    all_entities[entity_type] = {}
                all_entities[entity_type][entity_text] = all_entities[entity_type].get(entity_text, 0) + 1
        
        if 'patterns' in doc.metadata:
            all_patterns.extend(doc.metadata['patterns'])
        
        # Aggregate metadata
        if 'priority' in doc.metadata:
            priority = doc.metadata['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        if 'category' in doc.metadata:
            category = doc.metadata['category']
            category_counts[category] = category_counts.get(category, 0) + 1
        
        if 'status' in doc.metadata:
            status = doc.metadata['status']
            status_counts[status] = status_counts.get(status, 0) + 1
    
    # Display ticket distribution
    if priority_counts or category_counts:
        distribution_table = Table(title="🎫 Support Ticket Distribution", show_header=True, header_style="bold yellow")
        distribution_table.add_column("Metric", style="cyan")
        distribution_table.add_column("Value", style="white")
        distribution_table.add_column("Count", style="green")
        
        for priority, count in sorted(priority_counts.items(), key=lambda x: x[1], reverse=True):
            distribution_table.add_row("Priority", priority, str(count))
        
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            distribution_table.add_row("Category", category, str(count))
        
        for status, count in sorted(status_counts.items(), key=lambda x: x[1], reverse=True):
            distribution_table.add_row("Status", status, str(count))
        
        console.print(distribution_table)
    
    # Display most common entities (customers, products, etc.)
    if all_entities:
        entities_table = Table(title="👥 Key Entities in Support Tickets", show_header=True, header_style="bold green")
        entities_table.add_column("Entity Type", style="cyan")
        entities_table.add_column("Most Common", style="white")
        entities_table.add_column("Frequency", style="yellow")
        
        for entity_type, entities in all_entities.items():
            if entities:
                most_common = max(entities.items(), key=lambda x: x[1])
                entities_table.add_row(entity_type, most_common[0].title(), str(most_common[1]))
        
        console.print(entities_table)
    
    # Display common resolution patterns
    if all_patterns:
        patterns_table = Table(title="🔧 Common Resolution Patterns", show_header=True, header_style="bold blue")
        patterns_table.add_column("Pattern Type", style="cyan")
        patterns_table.add_column("Description", style="white")
        patterns_table.add_column("Frequency", style="yellow")
        
        pattern_counts = {}
        for pattern in all_patterns:
            pattern_type = pattern.get('type', 'Unknown')
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
        
        for pattern_type, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            description = f"Resolution steps for {pattern_type.lower()} issues"
            patterns_table.add_row(pattern_type, description, str(count))
        
        console.print(patterns_table)


def print_support_search_results(query: str, results: List[Document]):
    """Print search results optimized for support scenarios."""
    console.print(f"\n🔍 Support Query: [bold yellow]'{query}'[/bold yellow]")
    console.print(f"🎫 Found {len(results)} relevant support cases/articles")
    
    for i, doc in enumerate(results[:3], 1):
        score = doc.metadata.get('search_score', 'N/A')
        score_text = f"Relevance: {score:.4f}" if isinstance(score, float) else f"Relevance: {score}"
        
        # Show support-specific metadata
        support_info = []
        if 'priority' in doc.metadata:
            priority = doc.metadata['priority']
            if priority in ['High', 'Critical']:
                support_info.append(f"🔥 {priority} Priority")
            else:
                support_info.append(f"📋 {priority} Priority")
        
        if 'category' in doc.metadata:
            support_info.append(f"📂 {doc.metadata['category']}")
        
        if 'status' in doc.metadata:
            status = doc.metadata['status']
            if status == 'Resolved':
                support_info.append("✅ Resolved")
            elif status == 'In Progress':
                support_info.append("🔄 In Progress")
            else:
                support_info.append(f"📌 {status}")
        
        support_metadata = " | ".join(support_info) if support_info else "No metadata"
        
        # Content preview focusing on issue description
        content_preview = doc.content[:350] + "..." if len(doc.content) > 350 else doc.content
        
        result_text = f"""[bold]Source:[/bold] {Path(doc.source).name if doc.source else "Support Document"}
[bold]{score_text}[/bold]
[bold]Ticket Info:[/bold] {support_metadata}

{content_preview}"""
        
        console.print(Panel(
            result_text,
            title=f"Support Result #{i}",
            title_align="left",
            border_style="green" if i == 1 else "blue",
            expand=False
        ))


def demonstrate_customer_support_rag():
    """Demonstrate RAG system optimized for customer support scenarios using strategy-first approach."""
    
    print_section_header("🦙 Demo 2: Customer Support Knowledge System", "🎧")
    
    console.print("\n[bold green]This demo showcases:[/bold green]")
    console.print("• [bold cyan]Strategy-first configuration approach[/bold cyan] (no hardcoded parameters!)")
    console.print("• Processing structured support ticket data (CSV)")
    console.print("• Analyzing unstructured knowledge base articles")
    console.print("• Entity extraction for customers, products, and issues")
    console.print("• Pattern recognition for common resolution workflows")
    console.print("• Intelligent support case matching and recommendations")
    console.print("• Knowledge base search for agent assistance")
    
    # Initialize system using strategy
    print_section_header("Strategy-Based System Initialization", "⚙️")
    
    console.print("🚀 [bold cyan]Initializing system with 'customer_support_demo' strategy...[/bold cyan]")
    
    try:
        # Create demo system using strategy - all configuration comes from demo_strategies.yaml!
        rag_system = create_demo_system("customer_support_demo")
        
        # Show strategy information
        rag_system.print_strategy_info()
        
        console.print("\n✅ [bold green]Customer support system initialized using strategy![/bold green]")
        console.print("[dim]💡 All component configurations loaded from demo_strategies.yaml[/dim]")
        console.print("[dim]🔧 No hardcoded parameters in demo code![/dim]")
        
    except Exception as e:
        console.print(f"❌ [red]Failed to initialize strategy system: {e}[/red]")
        console.print("[dim]💡 Make sure demo_strategies.yaml exists and contains customer_support_demo[/dim]")
        return
    
    # Process support data
    print_section_header("Support Data Processing", "📊")
    
    console.print("[bold cyan]🔄 Processing support data using strategy-configured parser and extractors...[/bold cyan]")
    
    support_files = [
        "demos/static_samples/customer_support/support_tickets.csv",
        "demos/static_samples/customer_support/knowledge_base.txt"
    ]
    
    # Process documents using strategy system - all parsing and extraction configured in strategy!
    all_documents = rag_system.process_documents(support_files, show_progress=True)
    
    if not all_documents:
        console.print("❌ [red]No documents were processed. Check file paths.[/red]")
        return
    
    console.print(f"\n✅ Total support items processed: [bold green]{len(all_documents)}[/bold green]")
    console.print("[dim]💡 Documents processed with strategy-configured parser and extractors[/dim]")
    
    # Apply support extractors - already done by strategy system!
    print_section_header("Support Intelligence Extraction", "🧠")
    
    console.print("✅ [bold green]Support content analysis complete![/bold green]")
    console.print("[dim]💡 All extractors were applied automatically by the strategy system[/dim]")
    console.print("[dim]🔧 Extractor configurations loaded from demo_strategies.yaml[/dim]")
    
    console.print("✅ Support intelligence extraction complete!")
    
    # Show support insights
    print_extractor_insights(all_documents)
    
    # Show detailed analysis of first few tickets
    console.print("\n📋 [bold green]Sample Ticket Analysis[/bold green]")
    ticket_docs = [doc for doc in all_documents if doc.metadata.get('data_type') == 'tickets'][:3]
    for i, doc in enumerate(ticket_docs, 1):
        console.print(f"\n🎫 Support Ticket #{i}:")
        print_support_ticket_analysis(doc)
    
    # Generate embeddings using strategy system
    print_section_header("Support Content Embedding", "🧠")
    console.print("[bold cyan]🔄 Generating embeddings using strategy-configured embedder...[/bold cyan]")
    
    # Use strategy system for embeddings - all configuration from demo_strategies.yaml!
    all_documents = rag_system.generate_embeddings(all_documents, show_progress=True)
    console.print("[dim]💡 Embedder model and configuration loaded from demo_strategies.yaml[/dim]")
    
    # Store in vector database using strategy system  
    print_section_header("Support Knowledge Base Storage", "🗄️")
    console.print("[bold cyan]💾 Storing support data using strategy-configured vector store...[/bold cyan]")
    
    # Use strategy system for storage - all configuration from demo_strategies.yaml!
    success = rag_system.store_documents(all_documents, show_progress=True)
    console.print("[dim]💡 Vector store type and configuration loaded from demo_strategies.yaml[/dim]")
    
    # Demonstrate support queries
    print_section_header("Support Query Demonstration", "🔍")
    
    support_queries = [
        "Customer cannot log into their account after password reset",
        "Payment declined for premium subscription with valid card",
        "System running slowly and taking too long to load pages",
        "How to export data in CSV format for reporting",
        "Account suddenly locked without user action",
        "Files corrupted after system maintenance",
        "Unable to reset password, not receiving emails"
    ]
    
    console.print("🎯 Running support queries to demonstrate case matching and knowledge retrieval:")
    
    for i, query in enumerate(support_queries, 1):
        console.print(f"\n[bold cyan]Support Query #{i}:[/bold cyan]")
        console.print(f"🔍 Searching with query: '{query}'")
        
        # Use strategy system for search - all configured in demo_strategies.yaml!
        results = rag_system.search(query, show_progress=False)
        
        # Show support-focused results  
        from demos.utils import display_search_results_with_metadata
        display_search_results_with_metadata(results, query)
        
        if i < len(support_queries):
            console.print("\n" + "─" * 50)
            time.sleep(1.2)  # Pause for readability
    
    # Show support system statistics
    print_section_header("Support System Analytics", "📊")
    
    # Use strategy system to get collection info
    info = rag_system.vector_store.get_collection_info() if hasattr(rag_system.vector_store, 'get_collection_info') else None
    if info:
        # Calculate support-specific metrics
        ticket_count = sum(1 for doc in all_documents if doc.metadata.get('data_type') == 'tickets')
        knowledge_count = sum(1 for doc in all_documents if doc.metadata.get('data_type') == 'knowledge')
        total_entities = sum(len(doc.metadata.get('entities', [])) for doc in all_documents)
        total_patterns = sum(len(doc.metadata.get('patterns', [])) for doc in all_documents)
        
        stats_table = Table(show_header=True, header_style="bold magenta")
        stats_table.add_column("Support Metric", style="cyan")
        stats_table.add_column("Value", style="white")
        
        stats_table.add_row("Support Collection", info.get("name", "N/A"))
        stats_table.add_row("Total Support Items", str(info.get("document_count", "N/A")))
        stats_table.add_row("Support Tickets", str(ticket_count))
        stats_table.add_row("Knowledge Articles", str(knowledge_count))
        stats_table.add_row("Extracted Entities", str(total_entities))
        stats_table.add_row("Resolution Patterns", str(total_patterns))
        stats_table.add_row("Embedding Model", "OllamaEmbedder (nomic-embed-text)")
        
        console.print(stats_table)
    
    # Support system summary
    print_section_header("Support System Summary", "🎉")
    
    console.print("🚀 [bold green]Customer Support System Complete![/bold green]")
    console.print("\n[bold]What this demo demonstrated:[/bold]")
    console.print("✅ Processing structured support ticket data (CSV)")
    console.print("✅ Analyzing unstructured knowledge base content")
    console.print("✅ Entity extraction for customers, products, and issues")
    console.print("✅ Pattern recognition for common resolution workflows")
    console.print("✅ Intelligent case matching and similarity search")
    console.print("✅ Knowledge base search for agent assistance")
    
    console.print(f"\n[bold]Why this approach is powerful for support:[/bold]")
    console.print("🎫 Automatic similar case detection")
    console.print("📚 Instant knowledge base search")
    console.print("🎯 Entity-aware issue categorization")
    console.print("🔧 Resolution pattern recognition")
    console.print("📊 Support analytics and insights")
    
    console.print(f"\n📁 Support knowledge base saved to: [bold]./demos/vectordb/customer_support[/bold]")
    console.print("🔄 You can now query this support system using the CLI:")
    console.print("[dim]uv run python cli.py search 'login issues password reset' --collection customer_support[/dim]")

    # Clean up database to prevent duplicate accumulation
    print_section_header("Database Cleanup", "🧹")
    console.print("🗑️  Cleaning up vector database to prevent duplicate accumulation...")
    try:
        # Delete the collection to clean up
        if hasattr(rag_system.vector_store, 'delete_collection'):
            rag_system.vector_store.delete_collection()
        console.print("✅ [green]Database cleaned successfully![/green]")
        console.print("[dim]The database has been reset to prevent duplicate data accumulation in future runs.[/dim]")
    except Exception as e:
        console.print(f"⚠️  [yellow]Note: Could not clean database: {e}[/yellow]")
        console.print("[dim]You may want to manually clean the vector database directory.[/dim]")



if __name__ == "__main__":
    try:
        demonstrate_customer_support_rag()
    except KeyboardInterrupt:
        console.print("\n\n👋 Support demo interrupted by user", style="yellow")
    except Exception as e:
        import traceback
        console.print(f"\n\n❌ Support demo failed: {str(e)}", style="red")
        console.print("Check that Ollama is running with the nomic-embed-text model")
        console.print("\nDebug traceback:")
        traceback.print_exc()
        sys.exit(1)