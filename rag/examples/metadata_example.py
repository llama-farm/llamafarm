#!/usr/bin/env python3
"""
Example demonstrating comprehensive metadata management in RAG system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.pretty import Pretty

# Import RAG components
from core.base import Document, Pipeline
from components.parsers.text_parser import PlainTextParser
from components.metadata import (
    MetadataEnricher,
    MetadataSchema,
    MetadataPresets,
    MetadataFilter,
    MetadataLevel
)
from components.embedders.ollama_embedder import OllamaEmbedder
from components.stores.chroma_store import ChromaStore

console = Console()


def demonstrate_metadata_enrichment():
    """Demonstrate metadata enrichment capabilities."""
    console.print("\n[bold cyan]🏷️  Metadata Management Example[/bold cyan]")
    console.print("=" * 80)
    
    # Create sample documents
    documents = [
        Document(
            content="Artificial Intelligence is transforming how we work and live.",
            source="blog/ai_transformation.txt"
        ),
        Document(
            content="Machine learning models require careful evaluation and testing.",
            source="research/ml_evaluation.pdf"
        ),
        Document(
            content="Customer reported issue with login functionality. Priority: High",
            source="tickets/SUPPORT-12345.txt"
        )
    ]
    
    console.print("\n[bold]1. Basic Metadata Enrichment[/bold]")
    console.print("-" * 40)
    
    # Create basic enricher
    basic_enricher = MetadataEnricher(schema=MetadataPresets.minimal())
    enriched_docs = basic_enricher.enrich_documents(documents[:1])
    
    # Display basic metadata
    doc = enriched_docs[0]
    basic_table = Table(title="Basic Metadata", show_header=True)
    basic_table.add_column("Field", style="cyan")
    basic_table.add_column("Value", style="white")
    
    for key, value in sorted(doc.metadata.items()):
        basic_table.add_row(key, str(value))
    
    console.print(basic_table)
    
    # Standard enrichment
    console.print("\n[bold]2. Standard Metadata Enrichment[/bold]")
    console.print("-" * 40)
    
    standard_enricher = MetadataEnricher()  # Uses standard schema by default
    enriched_docs = standard_enricher.enrich_documents(documents[:2])
    
    for i, doc in enumerate(enriched_docs):
        console.print(f"\n[yellow]Document {i+1}: {doc.source}[/yellow]")
        metadata_panel = Panel(
            Pretty(doc.metadata, expand_all=True),
            title="Metadata",
            border_style="green"
        )
        console.print(metadata_panel)
    
    # Domain-specific enrichment
    console.print("\n[bold]3. Domain-Specific Metadata Enrichment[/bold]")
    console.print("-" * 40)
    
    # Research paper enricher
    research_enricher = MetadataEnricher(schema=MetadataPresets.research_papers())
    research_doc = Document(
        content="This paper explores transformer architectures...",
        source="papers/transformers_2024.pdf",
        metadata={
            "authors": ["Smith, J.", "Doe, A."],
            "doi": "10.1234/ai.2024.001",
            "journal": "AI Research Quarterly"
        }
    )
    
    enriched_research = research_enricher.enrich_document(research_doc)
    console.print("\n[yellow]Research Paper Metadata:[/yellow]")
    console.print(Pretty(enriched_research.metadata, expand_all=True))
    
    # Customer support enricher
    support_enricher = MetadataEnricher(schema=MetadataPresets.customer_support())
    support_doc = Document(
        content=documents[2].content,
        source=documents[2].source,
        metadata={
            "ticket_id": "SUPPORT-12345",
            "priority": "high",
            "status": "open",
            "customer_id": "CUST-789"
        }
    )
    
    enriched_support = support_enricher.enrich_document(support_doc)
    console.print("\n[yellow]Support Ticket Metadata:[/yellow]")
    console.print(Pretty(enriched_support.metadata, expand_all=True))
    
    # Custom schema
    console.print("\n[bold]4. Custom Metadata Schema[/bold]")
    console.print("-" * 40)
    
    custom_schema = MetadataSchema(
        level=MetadataLevel.COMPREHENSIVE,
        required_fields=["id", "source", "processing_timestamp", "security_level"],
        optional_fields=["department", "reviewer", "approved"],
        custom_fields={
            "organization": "Acme Corp",
            "retention_period": "7 years",
            "compliance": ["GDPR", "HIPAA"]
        }
    )
    
    custom_enricher = MetadataEnricher(schema=custom_schema)
    secure_doc = Document(
        content="Confidential financial report for Q4 2024.",
        source="reports/q4_2024_financial.pdf",
        metadata={
            "security_level": "confidential",
            "department": "finance",
            "approved": True
        }
    )
    
    enriched_secure = custom_enricher.enrich_document(secure_doc)
    console.print("\n[yellow]Custom Schema Metadata:[/yellow]")
    console.print(Pretty(enriched_secure.metadata, expand_all=True))
    
    # Metadata filtering
    console.print("\n[bold]5. Metadata Filtering[/bold]")
    console.print("-" * 40)
    
    all_docs = [enriched_research, enriched_support, enriched_secure]
    
    # Filter by priority
    high_priority_filter = MetadataFilter({"priority": "high"})
    high_priority_docs = high_priority_filter.filter_documents(all_docs)
    console.print(f"\nHigh priority documents: {len(high_priority_docs)}")
    
    # Filter by date range
    date_filter = MetadataFilter({
        "processing_date": {"$gte": "2024-01-01"}
    })
    recent_docs = date_filter.filter_documents(all_docs)
    console.print(f"Recent documents: {len(recent_docs)}")
    
    # Complex filter
    complex_filter = MetadataFilter({
        "$or": [
            {"security_level": "confidential"},
            {"priority": {"$in": ["high", "critical"]}}
        ]
    })
    sensitive_docs = complex_filter.filter_documents(all_docs)
    console.print(f"Sensitive documents: {len(sensitive_docs)}")
    
    # ChromaDB compatibility
    console.print("\n[bold]6. ChromaDB Compatibility[/bold]")
    console.print("-" * 40)
    
    console.print("\n[yellow]Original metadata (with complex types):[/yellow]")
    complex_metadata = {
        "id": "doc_123",
        "tags": ["ai", "ml", "research"],
        "authors": {"primary": "Smith, J.", "contributors": ["Doe, A.", "Johnson, B."]},
        "metrics": {"accuracy": 0.95, "f1_score": 0.92},
        "reviewed": True,
        "review_date": "2024-01-15"
    }
    console.print(Pretty(complex_metadata))
    
    console.print("\n[yellow]ChromaDB-compatible metadata:[/yellow]")
    chroma_metadata = standard_enricher._ensure_chroma_compatibility(complex_metadata)
    
    chroma_table = Table(title="ChromaDB Compatible", show_header=True)
    chroma_table.add_column("Field", style="cyan")
    chroma_table.add_column("Type", style="yellow")
    chroma_table.add_column("Value", style="white")
    
    for key, value in sorted(chroma_metadata.items()):
        chroma_table.add_row(key, type(value).__name__, str(value)[:50])
    
    console.print(chroma_table)
    
    # Integration with pipeline
    console.print("\n[bold]7. Pipeline Integration[/bold]")
    console.print("-" * 40)
    
    # Create a pipeline with metadata enrichment
    pipeline = Pipeline("metadata_pipeline")
    
    # Add parser
    parser = PlainTextParser(name="PlainTextParser", config={"chunk_size": 500})
    pipeline.add_component(parser)
    
    # Add metadata enricher as a custom component
    class MetadataEnricherComponent:
        def __init__(self, enricher):
            self.enricher = enricher
            self.name = "MetadataEnricher"
        
        def process(self, documents):
            from core.base import ProcessingResult
            enriched = self.enricher.enrich_documents(documents)
            return ProcessingResult(documents=enriched, errors=[])
    
    enricher_component = MetadataEnricherComponent(
        MetadataEnricher(schema=MetadataPresets.research_papers())
    )
    pipeline.add_component(enricher_component)
    
    # Process a document
    result = pipeline.run(source="samples/test_document.txt")
    
    if result.documents:
        console.print(f"\nProcessed {len(result.documents)} documents with metadata")
        doc = result.documents[0]
        console.print(f"Document ID: {doc.id}")
        console.print(f"Metadata fields: {list(doc.metadata.keys())}")
    
    # Best practices summary
    console.print("\n[bold cyan]📚 Metadata Best Practices Summary[/bold cyan]")
    console.print("=" * 80)
    
    practices = [
        "1. Always include core metadata (ID, source, timestamp)",
        "2. Use domain-specific schemas for specialized content",
        "3. Ensure ChromaDB compatibility for vector storage",
        "4. Validate required fields before processing",
        "5. Use metadata filters for targeted retrieval",
        "6. Include content statistics for better filtering",
        "7. Preserve original metadata while adding enrichments"
    ]
    
    for practice in practices:
        console.print(f"✅ {practice}")
    
    console.print("\n[dim]See METADATA_BEST_PRACTICES.md for comprehensive guidelines[/dim]")


def demonstrate_metadata_search():
    """Demonstrate metadata-aware search capabilities."""
    console.print("\n[bold cyan]🔍 Metadata-Aware Search Example[/bold cyan]")
    console.print("=" * 80)
    
    # Create sample documents with rich metadata
    documents = [
        Document(
            content="Advanced transformer architectures for NLP.",
            source="papers/transformers_nlp.pdf",
            metadata={
                "category": "research",
                "field": "nlp",
                "year": 2024,
                "citations": 45,
                "peer_reviewed": True
            }
        ),
        Document(
            content="Computer vision using convolutional networks.",
            source="papers/cnn_vision.pdf",
            metadata={
                "category": "research",
                "field": "computer_vision",
                "year": 2023,
                "citations": 78,
                "peer_reviewed": True
            }
        ),
        Document(
            content="Customer guide for using our AI products.",
            source="docs/ai_user_guide.md",
            metadata={
                "category": "documentation",
                "audience": "customers",
                "version": "2.1.0",
                "last_updated": "2024-02-15"
            }
        )
    ]
    
    # Enrich with standard metadata
    enricher = MetadataEnricher()
    enriched_docs = enricher.enrich_documents(documents)
    
    # Add embeddings (mock for demonstration)
    for doc in enriched_docs:
        doc.embeddings = [0.1] * 768  # Mock embeddings
    
    # Create vector store with metadata support
    vector_store = ChromaStore("metadata_demo", {
        "collection_name": "metadata_search_demo",
        "persist_directory": "./demos/vectordb/metadata_demo"
    })
    
    # Store documents
    vector_store.add_documents(enriched_docs)
    
    # Demonstrate different search scenarios
    console.print("\n[bold]Search Scenarios:[/bold]")
    
    # 1. Search with category filter
    console.print("\n1. Search research papers only:")
    results = vector_store.search(
        query_embedding=[0.1] * 768,  # Mock query embedding
        filter={"category": "research"},
        top_k=2
    )
    console.print(f"   Found {len(results)} research papers")
    
    # 2. Search with multiple filters
    console.print("\n2. Search recent, highly-cited papers:")
    results = vector_store.search(
        query_embedding=[0.1] * 768,
        filter={
            "$and": [
                {"category": "research"},
                {"year": {"$gte": 2023}},
                {"citations": {"$gte": 50}}
            ]
        },
        top_k=2
    )
    console.print(f"   Found {len(results)} matching papers")
    
    # 3. Search documentation
    console.print("\n3. Search customer documentation:")
    results = vector_store.search(
        query_embedding=[0.1] * 768,
        filter={
            "$and": [
                {"category": "documentation"},
                {"audience": "customers"}
            ]
        },
        top_k=2
    )
    console.print(f"   Found {len(results)} customer docs")
    
    # Clean up
    vector_store.delete_collection()
    
    console.print("\n[dim]Metadata enables precise, filtered search across your knowledge base[/dim]")


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_metadata_enrichment()
    demonstrate_metadata_search()
    
    console.print("\n[bold green]✨ Metadata management demonstration complete![/bold green]")