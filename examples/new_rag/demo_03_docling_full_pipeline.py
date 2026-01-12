#!/usr/bin/env python3
"""
Demo 03: Docling Full Pipeline

This demo shows the complete Docling parsing pipeline:
- Loading and parsing a real PDF
- Smart chunking with HybridChunker
- Document structure extraction (headings, tables)
- Metadata preservation
- Output format options

Features demonstrated:
- AI-powered layout analysis
- Tokenizer-aware chunking
- Page-aware chunk metadata
- Comprehensive metadata extraction
"""

import sys
from pathlib import Path
from datetime import datetime

# Add rag directory to path for imports
RAG_DIR = Path(__file__).parent.parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))


def check_dependencies():
    """Check if Docling dependencies are installed."""
    missing = []

    try:
        from docling.document_converter import DocumentConverter
        print("Docling DocumentConverter: Available")
    except ImportError:
        missing.append("docling")
        print("Docling DocumentConverter: NOT AVAILABLE")

    try:
        from docling.chunking import HybridChunker
        print("Docling HybridChunker: Available")
    except ImportError:
        missing.append("docling-core[chunking]")
        print("Docling HybridChunker: NOT AVAILABLE (chunking disabled)")

    return missing


def find_sample_pdf():
    """Find a sample PDF for testing."""
    examples_dir = Path(__file__).parent.parent

    pdf_locations = [
        examples_dir / "fda_rag" / "files",
        examples_dir / "rag_legal_filings" / "files",
        examples_dir / "rag_pipeline" / "sample_files",
    ]

    for location in pdf_locations:
        if location.exists():
            pdfs = list(location.glob("*.pdf"))
            if pdfs:
                # Sort by size (prefer smaller for faster demo)
                pdfs.sort(key=lambda p: p.stat().st_size)
                return pdfs[0]

    return None


def demo_basic_parsing(pdf_path: Path):
    """Demonstrate basic Docling parsing without chunking."""
    print("\n" + "=" * 60)
    print("BASIC PARSING (No Chunking)")
    print("=" * 60)

    from components.parsers.docling import DoclingParser

    # Initialize parser without chunking
    parser = DoclingParser(config={
        "chunk_strategy": "none",
        "extract_metadata": True,
        "output_format": "markdown",
    })

    print(f"\nParsing: {pdf_path.name}")
    print(f"File size: {pdf_path.stat().st_size / 1024:.1f} KB")

    # Parse file
    start = datetime.now()
    result = parser.parse(str(pdf_path))
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\nParsing completed in {elapsed:.2f} seconds")
    print(f"Documents: {len(result.documents)}")
    print(f"Errors: {len(result.errors)}")

    if result.documents:
        doc = result.documents[0]
        print(f"\nDocument metadata:")
        for key, value in sorted(doc.metadata.items()):
            if key not in ["content", "custom_metadata", "headings"]:
                value_str = str(value)[:50]
                print(f"  {key}: {value_str}")

        # Show headings if extracted
        if "headings" in doc.metadata:
            print(f"\nExtracted headings ({len(doc.metadata['headings'])}):")
            for h in doc.metadata["headings"][:5]:
                print(f"  Level {h.get('level', '?')}: {h.get('text', '')[:50]}")

        print(f"\nContent preview (first 500 chars):")
        print("-" * 40)
        print(doc.content[:500])
        print("-" * 40)

    return result


def demo_chunked_parsing(pdf_path: Path):
    """Demonstrate Docling parsing with HybridChunker."""
    print("\n" + "=" * 60)
    print("CHUNKED PARSING (HybridChunker)")
    print("=" * 60)

    from components.parsers.docling import DoclingParser

    # Check if chunking is available
    try:
        from docling.chunking import HybridChunker
        chunking_available = True
    except ImportError:
        print("\nHybridChunker not available. Skipping chunked demo.")
        return None

    # Initialize parser with chunking
    parser = DoclingParser(config={
        "chunk_size": 256,  # Small chunks for demo
        "chunk_strategy": "hybrid",
        "extract_metadata": True,
        "output_format": "markdown",
    })

    print(f"\nParsing with HybridChunker (max_tokens=256)")
    print(f"File: {pdf_path.name}")

    # Parse file
    start = datetime.now()
    result = parser.parse(str(pdf_path))
    elapsed = (datetime.now() - start).total_seconds()

    print(f"\nParsing completed in {elapsed:.2f} seconds")
    print(f"Total chunks: {len(result.documents)}")

    if result.documents:
        print(f"\nChunk summary:")
        print("+-------+--------+--------------+------------------+")
        print("| Chunk | Chars  | Page(s)      | Has Headings     |")
        print("+-------+--------+--------------+------------------+")

        for i, doc in enumerate(result.documents[:10]):  # Show first 10
            meta = doc.metadata
            chars = len(doc.content)
            pages = meta.get("page_numbers", meta.get("page_start", "?"))
            has_heads = "Yes" if meta.get("headings") else "No"
            print(f"| {i:>5} | {chars:>6} | {str(pages):>12} | {has_heads:>16} |")

        print("+-------+--------+--------------+------------------+")

        if len(result.documents) > 10:
            print(f"... and {len(result.documents) - 10} more chunks")

        # Show a sample chunk
        print(f"\nSample chunk content (chunk 0):")
        print("-" * 40)
        print(result.documents[0].content[:400])
        print("-" * 40)

        # Show chunk metadata
        print(f"\nChunk 0 metadata:")
        for key, value in sorted(result.documents[0].metadata.items()):
            if key not in ["content", "custom_metadata", "headings"]:
                value_str = str(value)[:40]
                print(f"  {key}: {value_str}")

    return result


def demo_metadata_preservation():
    """Demonstrate metadata preservation through the pipeline."""
    print("\n" + "=" * 60)
    print("METADATA PRESERVATION")
    print("=" * 60)

    from components.parsers.docling import DoclingParser

    # Find PDF
    pdf_path = find_sample_pdf()
    if not pdf_path:
        print("No PDF found for metadata demo")
        return

    # Initialize parser
    parser = DoclingParser(config={"chunk_strategy": "none"})

    # Simulate metadata from upload/ingest process
    input_metadata = {
        "filename": pdf_path.name,
        "dataset_name": "fda_documents",
        "namespace": "regulatory",
        "project": "drug_approvals",
        "upload_timestamp": datetime.now().isoformat(),
        "file_hash": "sha256_abc123...",
        "custom_metadata": {
            "document_type": "FDA Warning Letter",
            "fiscal_year": "2024",
            "priority": "high",
        },
    }

    print(f"\nInput metadata:")
    for key, value in input_metadata.items():
        if key != "custom_metadata":
            print(f"  {key}: {value}")
        else:
            print(f"  custom_metadata: {value}")

    # Parse with metadata
    with open(pdf_path, "rb") as f:
        blob_data = f.read()

    documents = parser.parse_blob(blob_data, input_metadata)

    print(f"\nOutput document metadata:")
    if documents:
        doc = documents[0]
        for key, value in sorted(doc.metadata.items()):
            if key not in ["content", "headings"]:
                value_str = str(value)[:50]
                print(f"  {key}: {value_str}")

        print(f"\nPreserved fields:")
        preserved = ["dataset_name", "namespace", "project", "upload_timestamp", "file_hash"]
        for field in preserved:
            if field in doc.metadata:
                print(f"  {field}: PRESERVED")
            else:
                print(f"  {field}: MISSING")


def demo_output_formats():
    """Demonstrate different output formats."""
    print("\n" + "=" * 60)
    print("OUTPUT FORMATS")
    print("=" * 60)

    from components.parsers.docling import DoclingParser

    pdf_path = find_sample_pdf()
    if not pdf_path:
        print("No PDF found for format demo")
        return

    formats = ["markdown", "text", "json"]

    for fmt in formats:
        print(f"\n--- {fmt.upper()} Format ---")

        parser = DoclingParser(config={
            "output_format": fmt,
            "chunk_strategy": "none",
        })

        result = parser.parse(str(pdf_path))

        if result.documents:
            content = result.documents[0].content
            preview = content[:200].replace("\n", "\\n")
            print(f"Preview: {preview}...")


def compare_chunking_strategies():
    """Compare different chunking strategies."""
    print("\n" + "=" * 60)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 60)

    from components.parsers.docling import DoclingParser

    pdf_path = find_sample_pdf()
    if not pdf_path:
        print("No PDF found")
        return

    strategies = [
        ("none", "No chunking"),
        ("hybrid", "HybridChunker (default)"),
        ("hierarchical", "HierarchicalChunker"),
    ]

    print(f"\nFile: {pdf_path.name}")
    print(f"\n+------------------+--------+----------+----------+")
    print(f"| Strategy         | Chunks | Avg Size | Max Size |")
    print(f"+------------------+--------+----------+----------+")

    for strategy, desc in strategies:
        try:
            parser = DoclingParser(config={
                "chunk_size": 512,
                "chunk_strategy": strategy,
            })

            result = parser.parse(str(pdf_path))

            if result.documents:
                num_chunks = len(result.documents)
                avg_size = sum(len(d.content) for d in result.documents) // num_chunks
                max_size = max(len(d.content) for d in result.documents)
                print(f"| {desc[:16]:<16} | {num_chunks:>6} | {avg_size:>8} | {max_size:>8} |")
            else:
                print(f"| {desc[:16]:<16} | Failed |          |          |")

        except Exception as e:
            print(f"| {desc[:16]:<16} | Error: {str(e)[:30]} |")

    print(f"+------------------+--------+----------+----------+")


def main():
    """Run all demos."""
    print("=" * 60)
    print("LLAMAFARM RAG - Docling Full Pipeline Demo")
    print("=" * 60)

    # Check dependencies
    print("\n>>> Checking Dependencies...")
    missing = check_dependencies()

    if "docling" in missing:
        print("\nDocling is required for this demo.")
        print("Install with: pip install docling docling-core[chunking]")
        return 1

    # Find PDF
    print("\n>>> Finding Sample PDF...")
    pdf_path = find_sample_pdf()

    if not pdf_path:
        print("No sample PDF found!")
        print("Please add a PDF to one of:")
        print("  - examples/fda_rag/files/")
        print("  - examples/rag_legal_filings/files/")
        return 1

    print(f"Found: {pdf_path}")

    # Run demos
    print("\n>>> Running Demos...")

    demo_basic_parsing(pdf_path)
    demo_chunked_parsing(pdf_path)
    demo_metadata_preservation()
    compare_chunking_strategies()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
