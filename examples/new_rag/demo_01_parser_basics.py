#!/usr/bin/env python3
"""
Demo 01: Basic Parser Comparison

This demo shows how to use the new DoclingParser and MarkItDownParser
to parse documents, and compares their outputs.

Features demonstrated:
- Basic parser initialization
- PDF parsing with both parsers
- Metadata extraction comparison
- Output format differences
"""

import sys
from pathlib import Path

# Add rag directory to path for imports
RAG_DIR = Path(__file__).parent.parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))

from datetime import datetime


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        from docling.document_converter import DocumentConverter
        print("✅ Docling is installed")
    except ImportError:
        missing.append("docling (pip install docling docling-core[chunking])")
        print("❌ Docling is NOT installed")

    try:
        from markitdown import MarkItDown
        print("✅ MarkItDown is installed")
    except ImportError:
        missing.append("markitdown (pip install 'markitdown[all]')")
        print("❌ MarkItDown is NOT installed")

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
                # Prefer smaller files for faster demo
                pdfs.sort(key=lambda p: p.stat().st_size)
                return pdfs[0]

    return None


def demo_docling_parser(pdf_path: Path, metadata: dict):
    """Demonstrate DoclingParser capabilities."""
    print("\n" + "=" * 60)
    print("📘 DOCLING PARSER DEMO")
    print("=" * 60)

    try:
        from components.parsers.docling import DoclingParser

        # Initialize with smart chunking
        parser = DoclingParser(config={
            "chunk_size": 512,
            "chunk_strategy": "hybrid",
            "extract_metadata": True,
            "output_format": "markdown",
        })

        print(f"\n📄 Parsing: {pdf_path.name}")
        print(f"   File size: {pdf_path.stat().st_size / 1024:.1f} KB")

        # Parse blob (simulating API usage)
        with open(pdf_path, "rb") as f:
            blob_data = f.read()

        start_time = datetime.now()
        documents = parser.parse_blob(blob_data, metadata)
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ Parsing completed in {elapsed:.2f} seconds")
        print(f"   Generated {len(documents)} chunks")

        if documents:
            # Show first chunk details
            first_doc = documents[0]
            print(f"\n📋 First Chunk Metadata:")
            for key, value in sorted(first_doc.metadata.items()):
                if key not in ["content", "custom_metadata"]:
                    value_str = str(value)
                    if len(value_str) > 60:
                        value_str = value_str[:57] + "..."
                    print(f"   {key}: {value_str}")

            print(f"\n📝 First Chunk Content (first 500 chars):")
            print("-" * 40)
            print(first_doc.content[:500])
            print("-" * 40)

        return documents

    except ImportError as e:
        print(f"❌ Could not import DoclingParser: {e}")
        return []
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        return []


def demo_markitdown_parser(pdf_path: Path, metadata: dict):
    """Demonstrate MarkItDownParser capabilities."""
    print("\n" + "=" * 60)
    print("📗 MARKITDOWN PARSER DEMO")
    print("=" * 60)

    try:
        from components.parsers.markitdown import MarkItDownParser

        # Initialize (MarkItDown doesn't chunk by default)
        parser = MarkItDownParser(config={
            "output_format": "markdown",
            "extract_metadata": True,
        })

        print(f"\n📄 Parsing: {pdf_path.name}")
        print(f"   File size: {pdf_path.stat().st_size / 1024:.1f} KB")

        # Parse blob (simulating API usage)
        with open(pdf_path, "rb") as f:
            blob_data = f.read()

        start_time = datetime.now()
        documents = parser.parse_blob(blob_data, metadata)
        elapsed = (datetime.now() - start_time).total_seconds()

        print(f"\n✅ Parsing completed in {elapsed:.2f} seconds")
        print(f"   Generated {len(documents)} document(s)")

        if documents:
            # Show document details
            first_doc = documents[0]
            print(f"\n📋 Document Metadata:")
            for key, value in sorted(first_doc.metadata.items()):
                if key not in ["content", "custom_metadata"]:
                    value_str = str(value)
                    if len(value_str) > 60:
                        value_str = value_str[:57] + "..."
                    print(f"   {key}: {value_str}")

            print(f"\n📝 Document Content (first 500 chars):")
            print("-" * 40)
            print(first_doc.content[:500])
            print("-" * 40)

        return documents

    except ImportError as e:
        print(f"❌ Could not import MarkItDownParser: {e}")
        return []
    except Exception as e:
        print(f"❌ Error during parsing: {e}")
        return []


def compare_outputs(docling_docs, markitdown_docs):
    """Compare outputs from both parsers."""
    print("\n" + "=" * 60)
    print("📊 PARSER COMPARISON")
    print("=" * 60)

    print("\n┌─────────────────────┬────────────────┬────────────────┐")
    print("│ Metric              │ Docling        │ MarkItDown     │")
    print("├─────────────────────┼────────────────┼────────────────┤")

    # Number of chunks/documents
    docling_count = len(docling_docs) if docling_docs else 0
    markitdown_count = len(markitdown_docs) if markitdown_docs else 0
    print(f"│ Output count        │ {docling_count:>14} │ {markitdown_count:>14} │")

    # Total content length
    docling_len = sum(len(d.content) for d in docling_docs) if docling_docs else 0
    markitdown_len = sum(len(d.content) for d in markitdown_docs) if markitdown_docs else 0
    print(f"│ Total chars         │ {docling_len:>14,} │ {markitdown_len:>14,} │")

    # Chunking
    docling_chunked = "Yes" if docling_docs and docling_docs[0].metadata.get("is_chunked") else "No"
    markitdown_chunked = "Yes" if markitdown_docs and markitdown_docs[0].metadata.get("is_chunked") else "No"
    print(f"│ Chunked             │ {docling_chunked:>14} │ {markitdown_chunked:>14} │")

    print("└─────────────────────┴────────────────┴────────────────┘")

    print("\n💡 Key Differences:")
    print("   • Docling: AI-powered layout analysis, smart chunking, better for complex PDFs")
    print("   • MarkItDown: Fast conversion, no chunking, good for simple documents")


def main():
    """Run the demo."""
    print("=" * 60)
    print("🦙 LLAMAFARM RAG - New Parser Demo")
    print("=" * 60)

    # Check dependencies
    print("\n📦 Checking Dependencies...")
    missing = check_dependencies()

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Some features may not work. Install with the commands above.")

    # Find a sample PDF
    print("\n🔍 Finding Sample PDF...")
    pdf_path = find_sample_pdf()

    if not pdf_path:
        print("❌ No sample PDF found!")
        print("   Please copy a PDF to examples/new_rag/files/")
        return 1

    print(f"   Found: {pdf_path}")

    # Simulate metadata that would come from the upload/ingest process
    metadata = {
        "filename": pdf_path.name,
        "dataset_name": "demo_dataset",
        "upload_timestamp": datetime.now().isoformat(),
        "namespace": "demo",
        "project": "new_rag_demo",
        "content_type": "application/pdf",
    }

    # Run demos
    docling_docs = demo_docling_parser(pdf_path, metadata)
    markitdown_docs = demo_markitdown_parser(pdf_path, metadata)

    # Compare outputs
    compare_outputs(docling_docs, markitdown_docs)

    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
