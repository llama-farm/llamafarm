#!/usr/bin/env python3
"""
Demo 7: AFI Document Processing
================================
Demonstrates advanced PDF processing for Air Force Instructions (AFI) documents
with specialized parsing, extraction, and retrieval capabilities.

Features:
- Advanced PDF parsing with page-aware chunking
- AFI-specific pattern extraction (references, paragraph numbers, forms)
- Page number and section tracking for precise citations
- Hybrid retrieval strategy combining semantic and metadata search
- Technical terminology extraction and analysis
- Interactive user prompts for exploration
- Automatic database cleanup
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import json
from datetime import datetime
import shutil

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.strategies.manager import StrategyManager
from core.enhanced_pipeline import EnhancedPipeline
from utils.progress import LlamaProgressTracker
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup rich console for beautiful output
console = Console()


class AFIDocumentDemo:
    """Demonstration of AFI document processing capabilities."""
    
    def __init__(self):
        self.strategy_file = Path(__file__).parent / "demo_strategies.yaml"
        self.strategy_name = "afi_document_demo"
        self.afi_document = Path(__file__).parent / "static_samples" / "dafi21-101.pdf"
        self.vector_db_path = Path(__file__).parent / "vectordb" / "afi_documents"
        
    def run(self):
        """Run the complete AFI document processing demo."""
        console.print(Panel.fit(
            "[bold cyan]🚁 AFI DOCUMENT PROCESSING DEMO[/bold cyan]\n"
            "[yellow]Advanced military document analysis with interactive exploration[/yellow]",
            border_style="cyan"
        ))
        
        console.print(f"\n[bold]Document:[/bold] {self.afi_document.name}")
        console.print("[bold]Purpose:[/bold] Demonstrate advanced processing of military technical documentation")
        console.print("-"*80)
        
        # Interactive entry point 1: Ask if user wants to proceed
        if not Confirm.ask("\n[bold cyan]Ready to begin processing the AFI document?[/bold cyan]"):
            console.print("[yellow]Demo cancelled by user[/yellow]")
            return
        
        # Step 1: Initialize strategy
        print("\n📋 Step 1: Loading AFI-specific processing strategy...")
        strategy_manager = StrategyManager()
        strategy = strategy_manager.load_strategy(
            str(self.strategy_file),
            self.strategy_name
        )
        print(f"✅ Strategy loaded: {strategy.name}")
        print(f"   Description: {strategy.description}")
        
        # Step 2: Create pipeline
        print("\n🔧 Step 2: Creating enhanced processing pipeline...")
        pipeline = EnhancedPipeline(strategy=strategy)
        print("✅ Pipeline initialized with components:")
        print(f"   - Parser: {strategy.components['parser']['type']}")
        print(f"   - Extractors: {len(strategy.components.get('extractors', []))} extractors")
        print(f"   - Embedder: {strategy.components['embedder']['type']}")
        print(f"   - Vector Store: {strategy.components['vector_store']['type']}")
        
        # Step 3: Process the AFI document
        console.print("\n[bold cyan]📄 Step 3: Processing AFI document...[/bold cyan]")
        console.print(f"[dim]File: {self.afi_document}[/dim]")
        
        # Interactive entry point for processing details
        show_details = Confirm.ask("Would you like to see detailed processing steps?", default=True)
        
        # Create progress bar for processing
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Processing AFI document...", total=5)
            
            # Parse document
            progress.update(task, description="Parsing PDF...")
            if show_details:
                console.print("[dim]→ Extracting text and structure from PDF[/dim]")
            results = pipeline.process_documents(str(self.afi_document))
            progress.advance(task)
            
            # Extract metadata
            progress.update(task, description="Extracting AFI references and patterns...")
            if show_details:
                console.print("[dim]→ Finding AFI/TO references, form numbers, paragraph refs[/dim]")
            progress.advance(task)
            
            # Generate embeddings
            progress.update(task, description="Generating embeddings...")
            if show_details:
                console.print("[dim]→ Creating semantic embeddings for search[/dim]")
            progress.advance(task)
            
            # Store in vector database
            progress.update(task, description="Storing in vector database...")
            if show_details:
                console.print("[dim]→ Indexing documents for fast retrieval[/dim]")
            progress.advance(task)
            
            # Complete
            progress.update(task, description="Processing complete!")
            progress.advance(task)
            if show_details:
                console.print("[green]✓ All processing steps completed successfully![/green]")
        
        # Display processing results
        print("\n📊 Processing Results:")
        print(f"   Documents processed: {len(results.documents)}")
        print(f"   Total chunks created: {len(results.documents)}")
        
        if results.documents:
            # Show sample of extracted metadata
            print("\n📑 Sample Extracted Metadata (First Chunk):")
            first_doc = results.documents[0]
            metadata = first_doc.metadata
            
            # Display key metadata fields
            if 'page_number' in metadata:
                print(f"   - Page Number: {metadata['page_number']}")
            if 'section' in metadata:
                print(f"   - Section: {metadata['section']}")
            if 'afi_references' in metadata:
                print(f"   - AFI References Found: {metadata['afi_references'][:3]}...")
            if 'paragraph_refs' in metadata:
                print(f"   - Paragraph References: {metadata['paragraph_refs'][:3]}...")
            if 'entities' in metadata:
                print(f"   - Entities Extracted: {list(metadata['entities'].keys())}")
            if 'keywords' in metadata:
                print(f"   - Keywords: {metadata['keywords'][:5]}...")
        
        # Step 4: Demonstrate search capabilities
        print("\n🔍 Step 4: Demonstrating Search Capabilities...")
        self._demonstrate_searches(pipeline)
        
        # Step 5: Show advanced features
        # Interactive entry point 4: Ask about advanced features
        if Confirm.ask("\n[bold cyan]Would you like to see advanced AFI analysis features?[/bold cyan]"):
            print("\n⚡ Step 5: Advanced Features Demonstration...")
            self._demonstrate_advanced_features(pipeline, results)
        
        # Summary
        console.print("\n" + "="*80)
        console.print("[bold green]✅ AFI DOCUMENT PROCESSING DEMO COMPLETE![/bold green]")
        console.print("="*80)
        console.print("\n[bold]💡 Key Capabilities Demonstrated:[/bold]")
        console.print("   1. Page-aware PDF parsing with structure preservation")
        console.print("   2. AFI-specific pattern extraction (references, forms, codes)")
        console.print("   3. Precise citation tracking with page numbers")
        console.print("   4. Hybrid search combining semantic and metadata filtering")
        console.print("   5. Technical terminology and entity extraction")
        console.print("\n[bold]📚 Use Cases:[/bold]")
        console.print("   - Maintenance instruction lookup")
        console.print("   - Regulatory compliance checking")
        console.print("   - Cross-reference validation")
        console.print("   - Technical procedure search")
        
        # Database cleanup
        self._cleanup_database()
        
    def _demonstrate_searches(self, pipeline):
        """Demonstrate various search capabilities with interactive prompts."""
        
        # Interactive entry point 2: Ask if user wants to see predefined searches
        console.print("\n[bold cyan]📚 Ready to demonstrate search capabilities[/bold cyan]")
        if not Confirm.ask("Would you like to see example searches first?"):
            # Skip to custom search
            self._custom_search(pipeline)
            return
            
        # Example searches relevant to AFI documents
        searches = [
            {
                "query": "aircraft maintenance procedures",
                "description": "General maintenance search"
            },
            {
                "query": "safety requirements and precautions",
                "description": "Safety-focused search"
            },
            {
                "query": "inspection intervals and schedules",
                "description": "Scheduling information"
            },
            {
                "query": "technical order compliance",
                "description": "Compliance requirements"
            }
        ]
        
        console.print("\n[bold yellow]Example Searches:[/bold yellow]")
        for i, search in enumerate(searches, 1):
            console.print(f"\n[bold cyan]Search {i}: {search['description']}[/bold cyan]")
            console.print(f"[dim]Query: \"{search['query']}\"[/dim]")
            
            # Perform search
            results = pipeline.search(
                query=search['query'],
                top_k=5  # Get more results for better context
            )
            
            if results:
                console.print(f"[green]Found {len(results)} relevant chunks:[/green]")
                
                # Show more extensive content (dozen lines of context)
                for j, result in enumerate(results[:3], 1):  # Show top 3 results
                    console.print(f"\n[bold]Result {j}:[/bold]")
                    
                    # Show extensive snippet (500 chars for more context)
                    content_lines = result.content.split('\n')
                    content_preview = ""
                    char_count = 0
                    
                    # Build preview with multiple lines
                    for line in content_lines:
                        if char_count + len(line) < 500:
                            content_preview += line + "\n"
                            char_count += len(line)
                        else:
                            break
                    
                    # Display content in a nice panel
                    console.print(Panel(
                        content_preview.strip() + "...",
                        title=f"[cyan]Content Preview[/cyan]",
                        border_style="dim"
                    ))
                    
                    # Show detailed metadata
                    if hasattr(result, 'metadata'):
                        metadata_info = []
                        if 'page_number' in result.metadata:
                            metadata_info.append(f"📄 Page: {result.metadata['page_number']}")
                        if 'section' in result.metadata:
                            metadata_info.append(f"📑 Section: {result.metadata['section']}")
                        if 'paragraph_refs' in result.metadata:
                            refs = result.metadata['paragraph_refs'][:3]
                            metadata_info.append(f"📝 Paragraphs: {', '.join(refs)}")
                        if 'afi_references' in result.metadata:
                            refs = result.metadata['afi_references'][:2]
                            metadata_info.append(f"📚 AFI Refs: {', '.join(refs)}")
                        if 'score' in result.metadata:
                            metadata_info.append(f"📊 Relevance: {result.metadata['score']:.3f}")
                        
                        if metadata_info:
                            console.print("[dim]" + " | ".join(metadata_info) + "[/dim]")
                
                # Interactive entry point 3: Ask to continue
                if i < len(searches):
                    if not Confirm.ask("\n[cyan]Continue to next search?[/cyan]", default=True):
                        break
        
        # Offer custom search
        self._custom_search(pipeline)
    
    def _custom_search(self, pipeline):
        """Allow user to perform custom searches."""
        console.print("\n[bold cyan]💬 Custom Search Interface[/bold cyan]")
        console.print("[dim]Enter your own queries to search the AFI document[/dim]")
        
        while True:
            query = Prompt.ask("\n[bold]Enter search query (or 'done' to finish)[/bold]")
            
            if query.lower() == 'done':
                break
            
            # Perform search
            console.print(f"\n[yellow]Searching for: {query}...[/yellow]")
            results = pipeline.search(query=query, top_k=5)
            
            if results:
                console.print(f"[green]Found {len(results)} relevant chunks![/green]")
                
                # Show results with extensive context
                for j, result in enumerate(results[:3], 1):
                    console.print(f"\n[bold]Result {j}:[/bold]")
                    
                    # Show extensive snippet
                    content_preview = result.content[:600].replace('\n\n', '\n')
                    console.print(Panel(
                        content_preview + "...",
                        title=f"[cyan]Content (first 600 chars)[/cyan]",
                        border_style="dim"
                    ))
                    
                    # Show metadata
                    if hasattr(result, 'metadata') and 'page_number' in result.metadata:
                        console.print(f"[dim]📄 From page {result.metadata['page_number']}[/dim]")
            else:
                console.print("[red]No results found for this query[/red]")
    
    def _demonstrate_advanced_features(self, pipeline, processing_results):
        """Demonstrate advanced AFI-specific features."""
        
        print("\n🎯 Advanced AFI Features:")
        
        # 1. Reference extraction summary
        print("\n   1. AFI Reference Extraction:")
        all_references = set()
        for doc in processing_results.documents[:10]:  # Sample first 10 chunks
            if 'afi_references' in doc.metadata:
                all_references.update(doc.metadata['afi_references'])
        
        if all_references:
            print(f"      Found {len(all_references)} unique AFI/TO references:")
            for ref in list(all_references)[:5]:
                print(f"      - {ref}")
        
        # 2. Form identification
        print("\n   2. Military Form Identification:")
        all_forms = set()
        for doc in processing_results.documents[:10]:
            if 'form_numbers' in doc.metadata:
                all_forms.update(doc.metadata['form_numbers'])
        
        if all_forms:
            print(f"      Found {len(all_forms)} unique form references:")
            for form in list(all_forms)[:5]:
                print(f"      - {form}")
        
        # 3. Technical complexity analysis
        print("\n   3. Document Complexity Analysis:")
        if processing_results.documents:
            avg_readability = sum(
                doc.metadata.get('readability_score', 0) 
                for doc in processing_results.documents[:10]
            ) / min(10, len(processing_results.documents))
            
            print(f"      Average readability score: {avg_readability:.2f}")
            print(f"      Technical complexity: {'High' if avg_readability < 50 else 'Medium'}")
        
        # 4. Page distribution
        print("\n   4. Content Distribution:")
        page_counts = {}
        for doc in processing_results.documents:
            page = doc.metadata.get('page_number', 'unknown')
            page_counts[page] = page_counts.get(page, 0) + 1
        
        if page_counts:
            total_pages = len(page_counts)
            print(f"      Content spans {total_pages} pages")
            print(f"      Average chunks per page: {len(processing_results.documents) / total_pages:.1f}")
        
        # 5. Hybrid search demonstration
        print("\n   5. Hybrid Search Capabilities:")
        print("      ✅ Semantic search for conceptual queries")
        print("      ✅ Metadata filtering for precise location")
        print("      ✅ Multi-query expansion for comprehensive results")
        print("      ✅ Re-ranking based on multiple factors")
    
    def _cleanup_database(self):
        """Clean up the demo database after completion."""
        console.print("\n[bold cyan]🧹 Database Cleanup[/bold cyan]")
        console.print("[dim]Cleaning up demo data to keep your system tidy[/dim]")
        
        # Ask user if they want to keep the data
        if Confirm.ask("\n[yellow]Would you like to keep the AFI database for future queries?[/yellow]"):
            console.print("[green]✓ Database preserved at:[/green]", self.vector_db_path)
            console.print("[dim]You can query it later using the same strategy[/dim]")
            return
        
        # Clean up the database
        try:
            if self.vector_db_path.exists():
                console.print("[yellow]Removing AFI document database...[/yellow]")
                shutil.rmtree(self.vector_db_path)
                console.print("[green]✅ Database cleaned up successfully![/green]")
            else:
                console.print("[blue]ℹ️ No database found to clean up[/blue]")
                
            # Also try to clean up any cache files
            cache_path = Path(__file__).parent / ".cache" / "afi_documents"
            if cache_path.exists():
                shutil.rmtree(cache_path)
                console.print("[green]✅ Cache cleaned up successfully![/green]")
                
        except Exception as e:
            console.print(f"[yellow]⚠️ Cleanup encountered an issue: {e}[/yellow]")
            console.print("[dim]You can manually delete:", self.vector_db_path, "[/dim]")


def main():
    """Run the AFI document processing demo."""
    demo = AFIDocumentDemo()
    
    try:
        # Check if AFI document exists
        if not demo.afi_document.exists():
            print(f"\n⚠️  AFI document not found: {demo.afi_document}")
            print("Please ensure the DAFI 21-101 PDF is in the static_samples directory.")
            return
        
        # Check if strategy file exists
        if not demo.strategy_file.exists():
            print(f"\n⚠️  Strategy file not found: {demo.strategy_file}")
            return
        
        # Run the demo
        demo.run()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()