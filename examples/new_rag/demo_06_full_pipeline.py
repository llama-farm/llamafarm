#!/usr/bin/env python3
"""
Demo 06: Full Pipeline Integration

This demo shows the complete RAG pipeline with new parsers:
1. Parse documents with Docling/MarkItDown
2. Chunk content using HybridChunker or SimpleChunker
3. Verify metadata preservation
4. Demonstrate query-ready document structure

Note: This demo runs locally without requiring the server.
For API-based demos, see demo_07_api_workflow.py.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add rag directory to path for imports
RAG_DIR = Path(__file__).parent.parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))


def check_dependencies():
    """Check if required dependencies are installed."""
    deps = {}

    try:
        from docling.document_converter import DocumentConverter
        deps["docling"] = True
    except ImportError:
        deps["docling"] = False

    try:
        from markitdown import MarkItDown
        deps["markitdown"] = True
    except ImportError:
        deps["markitdown"] = False

    try:
        from components.chunkers import SimpleChunker
        deps["simple_chunker"] = True
    except ImportError:
        deps["simple_chunker"] = False

    return deps


def find_sample_files():
    """Find sample files for demo."""
    examples_dir = Path(__file__).parent.parent
    files = {"pdf": None, "docx": None, "txt": None}

    # Look for PDFs
    for subdir in ["fda_rag/files", "rag_legal_filings/files", "rag_pipeline/sample_files"]:
        pdf_dir = examples_dir / subdir
        if pdf_dir.exists():
            pdfs = list(pdf_dir.glob("*.pdf"))
            if pdfs:
                # Sort by size (prefer smaller for faster demo)
                pdfs.sort(key=lambda p: p.stat().st_size)
                files["pdf"] = pdfs[0]
                break

    # Create a sample text file
    sample_txt = Path(__file__).parent / "files" / "sample.txt"
    sample_txt.parent.mkdir(parents=True, exist_ok=True)

    if not sample_txt.exists():
        sample_txt.write_text("""# Sample Document for RAG Demo

## Introduction

This is a sample document used to demonstrate the LlamaFarm RAG pipeline.
The new parsing system includes Docling for PDFs and MarkItDown for Office documents.

## Key Features

### 1. AI-Powered Parsing
Docling uses AI to understand document structure, achieving 97.9% accuracy on table extraction.

### 2. Smart Chunking
The HybridChunker is tokenizer-aware, ensuring chunks are optimally sized for LLM processing.

### 3. Metadata Preservation
All parsers preserve important metadata like page numbers, headings, and document structure.

## Conclusion

The new RAG system simplifies document processing while improving accuracy.
Users no longer need to understand LlamaIndex internals to get great results.
""")

    files["txt"] = sample_txt
    return files


def demo_parse_pdf(pdf_path: Path):
    """Demonstrate PDF parsing with Docling."""
    print("\n" + "=" * 60)
    print("STEP 1: Parse PDF with Docling")
    print("=" * 60)

    if not pdf_path:
        print("No PDF file available. Skipping PDF demo.")
        return None

    from components.parsers.docling import DoclingParser

    print(f"\nParsing: {pdf_path.name}")
    print(f"Size: {pdf_path.stat().st_size / 1024:.1f} KB")

    # Initialize parser with HybridChunker
    parser = DoclingParser(config={
        "chunk_size": 256,
        "chunk_strategy": "hybrid",
        "output_format": "markdown",
        "extract_metadata": True,
    })

    start = datetime.now()
    result = parser.parse(str(pdf_path))
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\nParsing completed in {elapsed:.2f} seconds")
    print(f"Total chunks: {len(result.documents)}")

    if result.documents:
        print(f"\nFirst chunk preview:")
        print("-" * 40)
        print(result.documents[0].content[:300])
        print("-" * 40)

        print(f"\nFirst chunk metadata:")
        for key, value in list(result.documents[0].metadata.items())[:5]:
            print(f"  {key}: {str(value)[:50]}")

    return result


def demo_parse_text(txt_path: Path):
    """Demonstrate text file parsing with MarkItDown."""
    print("\n" + "=" * 60)
    print("STEP 2: Parse Text with MarkItDown")
    print("=" * 60)

    from components.parsers.markitdown import MarkItDownParser

    print(f"\nParsing: {txt_path.name}")

    parser = MarkItDownParser(config={
        "output_format": "markdown",
        "extract_metadata": True,
    })

    start = datetime.now()
    result = parser.parse(str(txt_path))
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\nParsing completed in {elapsed:.3f} seconds")
    print(f"Documents: {len(result.documents)}")

    if result.documents:
        print(f"\nContent preview:")
        print("-" * 40)
        print(result.documents[0].content[:400])
        print("-" * 40)

    return result


def demo_simple_chunking(text: str):
    """Demonstrate SimpleChunker standalone usage."""
    print("\n" + "=" * 60)
    print("STEP 3: Chunk with SimpleChunker")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    strategies = [
        (ChunkingStrategy.SENTENCES, 100),
        (ChunkingStrategy.PARAGRAPHS, 200),
        (ChunkingStrategy.SECTIONS, 300),
    ]

    print(f"\nOriginal text length: {len(text)} characters")
    print("\nChunking with different strategies:")
    print("-" * 40)

    for strategy, chunk_size in strategies:
        chunker = SimpleChunker(
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=20,
        )

        chunks = chunker.chunk(text)
        avg_size = sum(len(c.content) for c in chunks) // max(len(chunks), 1)

        print(f"\n  {strategy.value}:")
        print(f"    Chunks: {len(chunks)}")
        print(f"    Avg size: {avg_size} chars")
        print(f"    Target: {chunk_size} chars")


def demo_metadata_flow():
    """Demonstrate metadata preservation through the pipeline."""
    print("\n" + "=" * 60)
    print("STEP 4: Metadata Preservation")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    # Simulate metadata from document upload
    input_metadata = {
        "filename": "quarterly_report.pdf",
        "dataset_name": "financial_docs",
        "namespace": "acme_corp",
        "project": "q4_analysis",
        "upload_timestamp": datetime.now().isoformat(),
        "file_hash": "sha256_abc123...",
        "custom_metadata": {
            "document_type": "Quarterly Report",
            "fiscal_year": "2024",
            "quarter": "Q4",
        },
    }

    print("\nInput metadata:")
    for key, value in input_metadata.items():
        print(f"  {key}: {value}")

    # Chunk with metadata preservation
    chunker = SimpleChunker(
        strategy=ChunkingStrategy.PARAGRAPHS,
        chunk_size=150,
    )

    sample_text = """First paragraph of the quarterly report.

Second paragraph with financial details.

Third paragraph with conclusions."""

    chunks = chunker.chunk(sample_text, metadata=input_metadata)

    print(f"\nOutput chunks: {len(chunks)}")
    print("\nChunk 0 metadata (preserved + added):")
    for key, value in sorted(chunks[0].metadata.items()):
        print(f"  {key}: {str(value)[:50]}")

    # Verify key fields preserved
    preserved = ["filename", "dataset_name", "namespace", "project"]
    print("\nPreservation check:")
    for field in preserved:
        status = "PRESERVED" if field in chunks[0].metadata else "MISSING"
        print(f"  {field}: {status}")


def demo_complete_pipeline():
    """Demonstrate complete parse -> chunk -> query-ready flow."""
    print("\n" + "=" * 60)
    print("STEP 5: Complete Pipeline")
    print("=" * 60)

    from components.parsers.markitdown import MarkItDownParser
    from components.chunkers import SimpleChunker, ChunkingStrategy

    # Create sample file
    sample_file = Path(__file__).parent / "files" / "pipeline_demo.txt"
    sample_file.parent.mkdir(parents=True, exist_ok=True)

    sample_file.write_text("""# Pipeline Demo Document

## Overview

This document demonstrates the complete RAG pipeline.

## Section 1: Parsing

Documents are parsed using Docling (for PDFs) or MarkItDown (for Office docs).
The parsers extract text and preserve structure.

## Section 2: Chunking

After parsing, content is chunked using smart strategies.
The HybridChunker respects document structure.

## Section 3: Embedding

Chunks are ready for embedding with any sentence transformer.
Metadata is preserved for filtering during retrieval.

## Section 4: Retrieval

Semantic search finds relevant chunks.
Metadata filters narrow results by document type, date, etc.
""")

    print("\n1. PARSE: Using MarkItDownParser")
    parser = MarkItDownParser(config={"output_format": "markdown"})
    parse_result = parser.parse(str(sample_file))
    print(f"   Parsed {len(parse_result.documents)} document(s)")

    print("\n2. CHUNK: Using SimpleChunker (sections strategy)")
    chunker = SimpleChunker(
        strategy=ChunkingStrategy.SECTIONS,
        chunk_size=200,
        overlap=0,
    )

    full_content = parse_result.documents[0].content
    chunks = chunker.chunk(
        full_content,
        metadata={
            "source": "pipeline_demo.txt",
            "processed_at": datetime.now().isoformat(),
        },
    )
    print(f"   Created {len(chunks)} chunks")

    print("\n3. CHUNKS READY FOR EMBEDDING:")
    print("-" * 40)
    for i, chunk in enumerate(chunks[:4]):  # Show first 4
        print(f"\n   Chunk {i}:")
        print(f"   Content: {chunk.content[:60]}...")
        print(f"   Metadata keys: {list(chunk.metadata.keys())}")
    print("-" * 40)

    print("\n4. METADATA PRESERVED:")
    print(f"   source: {chunks[0].metadata.get('source')}")
    print(f"   chunk_index: {chunks[0].metadata.get('chunk_index')}")
    print(f"   total_chunks: {chunks[0].metadata.get('total_chunks')}")


def print_summary():
    """Print demo summary."""
    print("\n" + "=" * 60)
    print("DEMO COMPLETE - Summary")
    print("=" * 60)

    print("""
Key Capabilities Demonstrated:

1. DOCLING PARSER
   - AI-powered PDF parsing
   - 97.9% table extraction accuracy
   - Built-in HybridChunker for smart chunking
   - Tokenizer-aware chunks for optimal LLM usage

2. MARKITDOWN PARSER
   - Fast document-to-markdown conversion
   - Supports DOCX, PPTX, XLSX, HTML
   - Clean, consistent output

3. SIMPLE CHUNKER
   - Standalone chunking module
   - Multiple strategies: sentences, paragraphs, sections
   - Configurable size and overlap
   - Full metadata preservation

4. PIPELINE INTEGRATION
   - Parse -> Chunk -> Embed ready flow
   - Metadata flows through entire pipeline
   - Ready for vector storage and retrieval

Next Steps:
- Run demo_07_api_workflow.py for API-based demos
- See llamafarm.yaml for production configuration
- Check docs/rag/parsers.md for migration guide
""")


def main():
    """Run all demos."""
    print("=" * 60)
    print("LLAMAFARM RAG - Full Pipeline Integration Demo")
    print("=" * 60)

    # Check dependencies
    print("\nChecking dependencies...")
    deps = check_dependencies()
    for name, available in deps.items():
        status = "Available" if available else "NOT AVAILABLE"
        print(f"  {name}: {status}")

    if not deps["docling"]:
        print("\nWarning: Docling not installed. PDF demos will be limited.")

    # Find sample files
    print("\nFinding sample files...")
    files = find_sample_files()
    for file_type, path in files.items():
        if path:
            print(f"  {file_type}: {path.name}")
        else:
            print(f"  {file_type}: Not found")

    # Run demos
    if deps["docling"] and files["pdf"]:
        demo_parse_pdf(files["pdf"])
    elif files["txt"]:
        print("\nSkipping PDF demo (Docling not available)")

    demo_parse_text(files["txt"])

    if files["txt"]:
        text = files["txt"].read_text()
        demo_simple_chunking(text)

    demo_metadata_flow()
    demo_complete_pipeline()
    print_summary()

    return 0


if __name__ == "__main__":
    sys.exit(main())
