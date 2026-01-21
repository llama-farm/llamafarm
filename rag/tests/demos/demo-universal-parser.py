#!/usr/bin/env python3
"""Demo script for UniversalParser.

This demo shows:
1. Parse a sample .txt and .md file with different chunk strategies
2. Output JSON with chunks, each having content + rich metadata
"""

import json
import sys
import tempfile
from pathlib import Path

# Add rag directory to path
rag_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rag_dir))

from components.parsers.universal import UniversalParser  # noqa: E402


def create_sample_files() -> tuple[Path, Path]:
    """Create sample text and markdown files for demo."""
    # Sample text content
    txt_content = """Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on the development
of algorithms and statistical models that enable computers to perform tasks without
explicit programming.

The field has seen tremendous growth in recent years, driven by advances in computing
power, the availability of large datasets, and improvements in algorithms. Today,
machine learning is used in a wide variety of applications, from image recognition
and natural language processing to recommendation systems and autonomous vehicles.

There are three main types of machine learning: supervised learning, unsupervised
learning, and reinforcement learning. Each approach has its own strengths and
weaknesses, and the choice of method depends on the specific problem being addressed.

Supervised learning involves training a model on labeled data, where the correct
output is known for each input. This is the most common type of machine learning
and is used for tasks like classification and regression.
"""

    # Sample markdown content
    md_content = """# Getting Started with Python

Python is a versatile programming language that's perfect for beginners and experts alike.

## Why Python?

Python's simple syntax makes it easy to learn and read. It has a vast ecosystem of
libraries and frameworks that make development faster and more efficient.

## Basic Concepts

### Variables and Data Types

Python supports various data types including integers, floats, strings, lists,
dictionaries, and more. Variables in Python are dynamically typed, meaning you
don't need to declare their type explicitly.

### Functions and Classes

Functions are reusable blocks of code that perform specific tasks. Classes provide
a way to bundle data and functionality together, enabling object-oriented programming.

## Conclusion

Python's versatility and ease of use have made it one of the most popular programming
languages in the world. Whether you're building web applications, analyzing data, or
developing machine learning models, Python has the tools you need.
"""

    # Create temp files using context managers
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, prefix="demo_"
    ) as txt_file:
        txt_file.write(txt_content)
        txt_path = txt_file.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix="demo_"
    ) as md_file:
        md_file.write(md_content)
        md_path = md_file.name

    return Path(txt_path), Path(md_path)


def demo_chunk_strategy(
    parser_config: dict,
    file_path: Path,
    strategy_name: str,
) -> None:
    """Demo a specific chunk strategy."""
    print(f"\n{'=' * 60}")
    print(f"Strategy: {strategy_name}")
    print(f"File: {file_path.name}")
    print("=" * 60)

    parser = UniversalParser(parser_config)
    result = parser.parse(str(file_path))

    print(f"\nChunks created: {len(result.documents)}")
    print(f"Errors: {len(result.errors)}")

    if result.metrics:
        print(f"Metrics: {json.dumps(result.metrics, indent=2)}")

    for i, doc in enumerate(result.documents):
        print(f"\n--- Chunk {i + 1} ---")
        print(f"Content preview: {doc.content[:100]}...")
        print("\nMetadata:")
        print(json.dumps(doc.metadata, indent=2, default=str))


def main():
    """Run the UniversalParser demo."""
    print("=" * 60)
    print("UniversalParser Demo")
    print("=" * 60)

    # Create sample files
    txt_file, md_file = create_sample_files()

    try:
        # Demo 1: Paragraphs strategy on text file
        demo_chunk_strategy(
            {"chunk_strategy": "paragraphs", "chunk_size": 500},
            txt_file,
            "Paragraphs (text file)",
        )

        # Demo 2: Sections strategy on markdown file
        demo_chunk_strategy(
            {"chunk_strategy": "sections", "chunk_size": 500},
            md_file,
            "Sections (markdown file)",
        )

        # Demo 3: Sentences strategy
        demo_chunk_strategy(
            {"chunk_strategy": "sentences", "chunk_size": 300, "min_chunk_size": 30},
            txt_file,
            "Sentences (text file)",
        )

        # Demo 4: Characters strategy (fallback)
        demo_chunk_strategy(
            {"chunk_strategy": "characters", "chunk_size": 200, "min_chunk_size": 30},
            txt_file,
            "Characters/Fixed (text file)",
        )

        # Demo 5: Semantic strategy (if available)
        print("\n" + "=" * 60)
        print("Testing Semantic Strategy (may fall back if semchunk unavailable)")
        print("=" * 60)
        demo_chunk_strategy(
            {"chunk_strategy": "semantic", "chunk_size": 500},
            txt_file,
            "Semantic (text file)",
        )

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    finally:
        # Cleanup
        txt_file.unlink(missing_ok=True)
        md_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
