#!/usr/bin/env python
"""Demo script for MarkItDown parser in server."""

import sys
from pathlib import Path

# Add server to path
server_path = Path(__file__).parent.parent / "server"
sys.path.insert(0, str(server_path))

from services.rag.parsers.markitdown_parser import MarkItDownParser


def main():
    """Demo MarkItDown parser with real files."""
    print("=" * 80)
    print("MarkItDown Parser Demo")
    print("=" * 80)

    # Initialize parser
    print("\n1. Initializing MarkItDown parser...")
    parser = MarkItDownParser()
    print("✓ Parser initialized with OCR fallback support")

    # Test files from examples
    examples_dir = Path(__file__).parent.parent / "examples" / "rag_pipeline" / "sample_files"

    test_files = [
        (examples_dir / "Alpaca-Care-for-Beginners.pdf", "PDF Document"),
        (examples_dir / "news_articles" / "ai_breakthrough.html", "HTML Article"),
        (examples_dir / "fda" / "761248_2024_Orig1s000OtherActionLtrs.pdf", "FDA PDF"),
    ]

    for file_path, description in test_files:
        if not file_path.exists():
            print(f"\n❌ Skipping {description}: File not found at {file_path}")
            continue

        print(f"\n2. Parsing {description}...")
        print(f"   File: {file_path.name}")

        try:
            result = parser.parse_sync(str(file_path))

            # Show stats
            lines = result.split("\n")
            words = len(result.split())

            print(f"   ✓ Success!")
            print(f"   - Characters: {len(result):,}")
            print(f"   - Words: {words:,}")
            print(f"   - Lines: {len(lines):,}")

            # Show preview
            preview = result[:300].replace("\n", " ")
            print(f"\n   Preview: {preview}...")

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    print("\n" + "=" * 80)
    print("Demo Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
