#!/usr/bin/env python3
"""
Demo 04: Simple Configuration

This demo shows how the new parser and chunker system simplifies configuration:
- Minimal configuration examples
- Comparison with legacy approaches
- Mixed old/new parser configurations
- Backward compatibility demonstration

Features demonstrated:
- New Docling + SimpleChunker workflow
- Legacy parser compatibility
- Priority-based parser selection
- Migration path from old to new
"""

import sys
from pathlib import Path

# Add rag directory to path for imports
RAG_DIR = Path(__file__).parent.parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))


def demo_minimal_new_config():
    """Show minimal configuration for new parsers."""
    print("\n" + "=" * 60)
    print("MINIMAL NEW CONFIGURATION")
    print("=" * 60)

    print("\nWith the new Docling parser, configuration is simple:")
    print("-" * 40)

    minimal_config = """
# llamafarm.yaml - Minimal New Configuration

rag:
  data_processing_strategy:
    name: "simple_docling"
    parsers:
      - parser: DoclingParser
        # All defaults work out of the box!
        # chunk_size: 512 (default)
        # chunk_strategy: hybrid (default)
        # output_format: markdown (default)
"""
    print(minimal_config)

    print("\nCompare with the old LlamaIndex approach:")
    print("-" * 40)

    old_config = """
# Old approach - required understanding LlamaIndex internals

rag:
  data_processing_strategy:
    name: "llamaindex_pdf"
    parsers:
      - parser: PDFParser_LlamaIndex
        reader: LlamaParse  # Need to know LlamaIndex readers
        reader_options:
          result_type: markdown
          # Many LlamaIndex-specific options...
    node_parser:
      parser: SentenceSplitter  # LlamaIndex concept
      chunk_size: 512
      chunk_overlap: 50
    # Had to understand LlamaIndex's node/document model
"""
    print(old_config)

    print("\nBenefits of new approach:")
    print("  + No LlamaIndex internals to learn")
    print("  + Chunking built into parser")
    print("  + Sensible defaults work immediately")
    print("  + 97.9% table extraction accuracy with Docling")


def demo_full_featured_config():
    """Show full-featured configuration options."""
    print("\n" + "=" * 60)
    print("FULL-FEATURED CONFIGURATION")
    print("=" * 60)

    print("\nFor advanced use cases, all options are available:")
    print("-" * 40)

    full_config = """
# llamafarm.yaml - Full Configuration

rag:
  data_processing_strategy:
    name: "production_setup"

    # Multiple parsers with priority fallback
    parsers:
      # Primary: Docling for PDFs (AI-powered)
      - parser: DoclingParser
        extensions: [".pdf"]
        priority: 0  # Highest priority
        config:
          chunk_size: 512
          chunk_strategy: hybrid  # AI-aware chunking
          output_format: markdown
          extract_metadata: true
          extract_tables: true
          extract_images: false

      # Secondary: MarkItDown for Office docs
      - parser: MarkItDownParser
        extensions: [".docx", ".pptx", ".xlsx"]
        priority: 0
        config:
          output_format: markdown
          extract_metadata: true

      # Fallback: Legacy PyPDF2 (if Docling fails)
      - parser: PDFParser_PyPDF2
        extensions: [".pdf"]
        priority: 2  # Lower priority = fallback

    # Optional: Standalone chunker for post-processing
    chunker:
      type: SimpleChunker
      strategy: sentences  # or: characters, paragraphs, sections
      chunk_size: 256
      overlap: 50
"""
    print(full_config)


def demo_backward_compatibility():
    """Demonstrate backward compatibility with legacy configs."""
    print("\n" + "=" * 60)
    print("BACKWARD COMPATIBILITY")
    print("=" * 60)

    print("\nExisting configurations continue to work:")
    print("-" * 40)

    # Show the parsers that are available (from registry config)
    new_parsers = ["DoclingParser", "MarkItDownParser"]
    legacy_parsers = [
        "PDFParser_PyPDF2", "PDFParser_LlamaIndex",
        "CSVParser_Pandas", "CSVParser_Python", "CSVParser_LlamaIndex",
        "DocxParser_PythonDocx", "DocxParser_LlamaIndex",
        "ExcelParser_OpenPyXL", "ExcelParser_Pandas",
        "TextParser_Python", "TextParser_LlamaIndex",
        "MarkdownParser_Python", "MarkdownParser_LlamaIndex",
        "MsgParser_ExtractMsg", "HTMLParser_BeautifulSoup",
    ]

    print("\nAll registered parsers:")

    print(f"\n  NEW parsers ({len(new_parsers)}):")
    for p in sorted(new_parsers):
        print(f"    + {p}")

    print(f"\n  LEGACY parsers ({len(legacy_parsers)}) - still fully supported:")
    for p in sorted(legacy_parsers):
        print(f"    - {p}")


def demo_extension_lookup():
    """Show how parser selection works for file extensions."""
    print("\n" + "=" * 60)
    print("PARSER SELECTION BY EXTENSION")
    print("=" * 60)

    # Parser priority data (from registry config)
    extension_parsers = {
        ".pdf": [
            ("DoclingParser", 0, True),
            ("PDFParser_PyPDF2", 2, False),
            ("PDFParser_LlamaIndex", 3, False),
        ],
        ".docx": [
            ("MarkItDownParser", 0, True),
            ("DocxParser_PythonDocx", 2, False),
            ("DocxParser_LlamaIndex", 3, False),
        ],
        ".csv": [
            ("CSVParser_Pandas", 1, False),
            ("CSVParser_Python", 2, False),
            ("CSVParser_LlamaIndex", 3, False),
        ],
        ".xlsx": [
            ("MarkItDownParser", 0, True),
            ("ExcelParser_OpenPyXL", 2, False),
            ("ExcelParser_Pandas", 2, False),
        ],
        ".txt": [
            ("TextParser_Python", 1, False),
            ("TextParser_LlamaIndex", 3, False),
        ],
        ".md": [
            ("MarkdownParser_Python", 1, False),
            ("MarkdownParser_LlamaIndex", 3, False),
        ],
        ".msg": [
            ("MsgParser_ExtractMsg", 1, False),
        ],
    }

    print("\nParser priority for each file type:")
    print("-" * 40)

    for ext, parsers in extension_parsers.items():
        print(f"\n  {ext}:")
        for i, (parser_name, priority, is_new) in enumerate(parsers[:3]):
            marker = "[NEW]" if is_new else ""
            print(f"    {i+1}. {parser_name} (priority: {priority}) {marker}")
        if len(parsers) > 3:
            print(f"    ... and {len(parsers) - 3} more fallback options")


def demo_mixed_config():
    """Show mixing old and new parsers."""
    print("\n" + "=" * 60)
    print("MIXED CONFIGURATION")
    print("=" * 60)

    print("\nYou can mix new and legacy parsers freely:")
    print("-" * 40)

    mixed_config = """
# llamafarm.yaml - Mixed Configuration

rag:
  data_processing_strategy:
    name: "gradual_migration"

    parsers:
      # Use new Docling for PDFs
      - parser: DoclingParser
        extensions: [".pdf"]
        priority: 0
        config:
          chunk_strategy: hybrid

      # Keep using legacy Pandas for CSVs (works great)
      - parser: CSVParser_Pandas
        extensions: [".csv"]
        priority: 0
        config:
          delimiter: ","
          encoding: utf-8

      # Use new MarkItDown for Office docs
      - parser: MarkItDownParser
        extensions: [".docx", ".pptx"]
        priority: 0

      # Keep legacy for specialized formats
      - parser: MsgParser_ExtractMsg
        extensions: [".msg"]
        priority: 0
"""
    print(mixed_config)

    print("\nMigration strategy:")
    print("  1. Start with new parsers for PDF (biggest improvement)")
    print("  2. Gradually migrate other formats as needed")
    print("  3. Keep legacy parsers for specialized formats")
    print("  4. All configs remain backward compatible")


def demo_chunker_standalone():
    """Show standalone chunker usage."""
    print("\n" + "=" * 60)
    print("STANDALONE CHUNKER")
    print("=" * 60)

    print("\nSimpleChunker can be used independently:")
    print("-" * 40)

    try:
        from components.chunkers import SimpleChunker, ChunkingStrategy

        # Create chunker
        chunker = SimpleChunker(
            strategy=ChunkingStrategy.SENTENCES,
            chunk_size=100,
            overlap=20
        )

        sample_text = """
        LlamaFarm is a powerful platform for building AI applications.
        It provides tools for RAG, classifiers, and agents with tools.
        The new parsing system simplifies document processing significantly.
        Users can now focus on their use case instead of implementation details.
        """

        chunks = chunker.chunk(sample_text)

        print(f"\nInput text: {len(sample_text)} characters")
        print(f"Chunking: sentences, size=100, overlap=20")
        print(f"Output: {len(chunks)} chunks")
        print("-" * 40)

        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Content: {chunk.content[:60]}...")
            print(f"  Chars: {len(chunk.content)}")
            print(f"  Metadata: {chunk.metadata}")

    except ImportError:
        # Show example output if module not available
        print("\n(Showing example output - run from rag/ directory for live demo)")
        print("""
Input text: 312 characters
Chunking: sentences, size=100, overlap=20
Output: 4 chunks
----------------------------------------

Chunk 1:
  Content: LlamaFarm is a powerful platform for building AI appli...
  Chars: 67
  Metadata: {'chunk_num': 0, 'strategy': 'sentences'}

Chunk 2:
  Content: It provides tools for RAG, classifiers, and agents wi...
  Chars: 70
  Metadata: {'chunk_num': 1, 'strategy': 'sentences'}

Chunk 3:
  Content: The new parsing system simplifies document processing...
  Chars: 64
  Metadata: {'chunk_num': 2, 'strategy': 'sentences'}

Chunk 4:
  Content: Users can now focus on their use case instead of impl...
  Chars: 73
  Metadata: {'chunk_num': 3, 'strategy': 'sentences'}
""")


def demo_config_comparison_table():
    """Show side-by-side comparison table."""
    print("\n" + "=" * 60)
    print("CONFIGURATION COMPARISON")
    print("=" * 60)

    print("""
+------------------------+----------------------------+----------------------------+
| Feature                | Old (LlamaIndex)           | New (Docling/MarkItDown)   |
+------------------------+----------------------------+----------------------------+
| PDF Parsing            | PDFParser_LlamaIndex       | DoclingParser              |
|                        | + LlamaParse               | (built-in, no external)    |
+------------------------+----------------------------+----------------------------+
| Chunking               | Separate node_parser       | Built into parser          |
|                        | SentenceSplitter           | or use SimpleChunker       |
+------------------------+----------------------------+----------------------------+
| Table Extraction       | Limited (text-based)       | 97.9% accuracy (AI)        |
+------------------------+----------------------------+----------------------------+
| Config Complexity      | ~20 lines minimum          | ~5 lines minimum           |
+------------------------+----------------------------+----------------------------+
| Learning Curve         | Must learn LlamaIndex      | Just works                 |
+------------------------+----------------------------+----------------------------+
| Fallback Options       | Limited                    | Priority-based fallback    |
+------------------------+----------------------------+----------------------------+
| Office Docs            | Requires plugins           | MarkItDownParser built-in  |
+------------------------+----------------------------+----------------------------+
| Backward Compatible?   | N/A                        | Yes - all legacy works     |
+------------------------+----------------------------+----------------------------+
""")


def main():
    """Run all demos."""
    print("=" * 60)
    print("LLAMAFARM RAG - Simple Configuration Demo")
    print("=" * 60)
    print("\nThis demo shows the simplified configuration system.")

    demo_minimal_new_config()
    demo_full_featured_config()
    demo_backward_compatibility()
    demo_extension_lookup()
    demo_mixed_config()
    demo_chunker_standalone()
    demo_config_comparison_table()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("  1. New parsers (Docling, MarkItDown) are simpler to configure")
    print("  2. All legacy parsers continue to work unchanged")
    print("  3. You can mix new and old parsers freely")
    print("  4. Priority-based selection provides automatic fallback")
    print("  5. SimpleChunker works standalone or with any parser")

    return 0


if __name__ == "__main__":
    sys.exit(main())
