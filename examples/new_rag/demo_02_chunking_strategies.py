#!/usr/bin/env python3
"""
Demo 02: Chunking Strategies Comparison

This demo shows how to use the SimpleChunker with different strategies
to split documents in various ways.

Features demonstrated:
- Character-based chunking
- Sentence-based chunking
- Paragraph-based chunking
- Section-based chunking (markdown headers)
- Page-based chunking
- Token-based chunking
- Overlap handling
"""

import sys
from pathlib import Path

# Add rag directory to path for imports
RAG_DIR = Path(__file__).parent.parent.parent / "rag"
sys.path.insert(0, str(RAG_DIR))


# Sample document for testing
SAMPLE_DOCUMENT = """# Introduction

This is the introduction section of our sample document. It contains multiple sentences that explain what this document is about. The purpose is to demonstrate different chunking strategies.

## Background

Here we provide some background information. This section has several paragraphs.

The first paragraph discusses the history of text processing. Natural language processing has evolved significantly over the years. Modern approaches use transformer architectures.

The second paragraph covers current challenges. Handling long documents remains difficult. Chunking strategies help manage this complexity.

## Methods

### Data Collection

We collected data from multiple sources. The dataset includes various document types. Each document was carefully reviewed.

### Processing Pipeline

The processing pipeline involves several steps:
1. Document ingestion
2. Text extraction
3. Chunking
4. Embedding generation
5. Storage

## Results

Our results show significant improvements. The new chunking strategies outperform baseline methods. Users reported higher satisfaction with search results.

### Quantitative Analysis

The quantitative metrics are impressive:
- Accuracy: 95%
- Recall: 92%
- F1 Score: 93.5%

### Qualitative Feedback

Users appreciated the semantic coherence of chunks. The overlap feature helped maintain context across boundaries.

## Conclusion

In conclusion, proper chunking is essential for RAG systems. The SimpleChunker provides flexible options for different use cases. Future work will explore more advanced strategies.
"""


def demo_character_chunking():
    """Demonstrate character-based chunking."""
    print("\n" + "=" * 60)
    print("STRATEGY 1: CHARACTER-BASED CHUNKING")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    # Without overlap
    chunker = SimpleChunker(
        chunk_size=200,
        overlap=0,
        strategy=ChunkingStrategy.CHARACTERS,
    )

    chunks = chunker.chunk(SAMPLE_DOCUMENT)

    print(f"\nSettings: chunk_size=200, overlap=0")
    print(f"Result: {len(chunks)} chunks")

    print("\nFirst 3 chunks:")
    for i, chunk in enumerate(chunks[:3]):
        preview = chunk.content[:60].replace("\n", " ") + "..."
        print(f"  [{i}] {len(chunk.content)} chars: {preview}")

    # With overlap
    chunker_overlap = SimpleChunker(
        chunk_size=200,
        overlap=30,
        strategy=ChunkingStrategy.CHARACTERS,
    )

    chunks_overlap = chunker_overlap.chunk(SAMPLE_DOCUMENT)
    print(f"\nWith overlap=30: {len(chunks_overlap)} chunks")
    print(f"  Overlap before (chunk 1): {chunks_overlap[1].overlap_before} chars")

    return chunks


def demo_sentence_chunking():
    """Demonstrate sentence-based chunking."""
    print("\n" + "=" * 60)
    print("STRATEGY 2: SENTENCE-BASED CHUNKING")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    chunker = SimpleChunker(
        chunk_size=300,
        overlap=0,
        strategy=ChunkingStrategy.SENTENCES,
    )

    chunks = chunker.chunk(SAMPLE_DOCUMENT)

    print(f"\nSettings: chunk_size=300, strategy=sentences")
    print(f"Result: {len(chunks)} chunks")

    print("\nFirst 3 chunks (showing sentence boundaries):")
    for i, chunk in enumerate(chunks[:3]):
        # Count sentences in chunk
        sentences = [s.strip() for s in chunk.content.split(". ") if s.strip()]
        print(f"  [{i}] {len(sentences)} sentences, {len(chunk.content)} chars")
        preview = chunk.content[:80].replace("\n", " ") + "..."
        print(f"       {preview}")

    return chunks


def demo_paragraph_chunking():
    """Demonstrate paragraph-based chunking."""
    print("\n" + "=" * 60)
    print("STRATEGY 3: PARAGRAPH-BASED CHUNKING")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    chunker = SimpleChunker(
        chunk_size=400,
        overlap=0,
        strategy=ChunkingStrategy.PARAGRAPHS,
    )

    chunks = chunker.chunk(SAMPLE_DOCUMENT)

    print(f"\nSettings: chunk_size=400, strategy=paragraphs")
    print(f"Result: {len(chunks)} chunks")

    print("\nFirst 3 chunks (showing paragraph groups):")
    for i, chunk in enumerate(chunks[:3]):
        # Count paragraphs
        paras = [p.strip() for p in chunk.content.split("\n\n") if p.strip()]
        print(f"  [{i}] {len(paras)} paragraphs, {len(chunk.content)} chars")
        first_line = chunk.content.split("\n")[0][:60]
        print(f"       Starts: {first_line}...")

    return chunks


def demo_section_chunking():
    """Demonstrate section-based chunking (markdown headers)."""
    print("\n" + "=" * 60)
    print("STRATEGY 4: SECTION-BASED CHUNKING (Markdown Headers)")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    chunker = SimpleChunker(
        strategy=ChunkingStrategy.SECTIONS,
    )

    chunks = chunker.chunk(SAMPLE_DOCUMENT)

    print(f"\nSettings: strategy=sections")
    print(f"Result: {len(chunks)} sections")

    print("\nAll sections:")
    for i, chunk in enumerate(chunks):
        # Extract header from chunk
        lines = chunk.content.split("\n")
        header = lines[0] if lines else "No header"
        print(f"  [{i}] {header} ({len(chunk.content)} chars)")

    return chunks


def demo_token_chunking():
    """Demonstrate token-based chunking."""
    print("\n" + "=" * 60)
    print("STRATEGY 5: TOKEN-BASED CHUNKING")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    chunker = SimpleChunker(
        chunk_size=50,  # 50 tokens (approximately words)
        overlap=10,
        strategy=ChunkingStrategy.TOKENS,
    )

    chunks = chunker.chunk(SAMPLE_DOCUMENT)

    print(f"\nSettings: chunk_size=50 tokens, overlap=10")
    print(f"Result: {len(chunks)} chunks")

    print("\nFirst 3 chunks (showing word counts):")
    for i, chunk in enumerate(chunks[:3]):
        words = len(chunk.content.split())
        print(f"  [{i}] ~{words} words, {len(chunk.content)} chars")
        preview = " ".join(chunk.content.split()[:8]) + "..."
        print(f"       {preview}")

    return chunks


def demo_metadata_preservation():
    """Demonstrate metadata preservation through chunking."""
    print("\n" + "=" * 60)
    print("METADATA PRESERVATION")
    print("=" * 60)

    from components.chunkers import SimpleChunker

    chunker = SimpleChunker(chunk_size=300)

    # Simulate metadata from document parsing
    input_metadata = {
        "source_filename": "sample_document.pdf",
        "dataset_name": "demo_dataset",
        "namespace": "demo_ns",
        "project": "chunking_demo",
        "upload_timestamp": "2024-01-15T10:30:00",
        "parser_type": "DoclingParser",
        "page_count": 5,
    }

    chunks = chunker.chunk(SAMPLE_DOCUMENT, input_metadata)

    print(f"\nInput metadata keys: {list(input_metadata.keys())}")
    print(f"\nChunk metadata (first chunk):")

    first_chunk = chunks[0]
    for key, value in sorted(first_chunk.metadata.items()):
        value_str = str(value)
        if len(value_str) > 40:
            value_str = value_str[:37] + "..."
        print(f"  {key}: {value_str}")


def compare_all_strategies():
    """Compare all strategies on the same document."""
    print("\n" + "=" * 60)
    print("STRATEGY COMPARISON SUMMARY")
    print("=" * 60)

    from components.chunkers import SimpleChunker, ChunkingStrategy

    strategies = [
        ("Characters (200)", ChunkingStrategy.CHARACTERS, 200),
        ("Sentences (300)", ChunkingStrategy.SENTENCES, 300),
        ("Paragraphs (400)", ChunkingStrategy.PARAGRAPHS, 400),
        ("Sections", ChunkingStrategy.SECTIONS, None),
        ("Tokens (50)", ChunkingStrategy.TOKENS, 50),
    ]

    print("\n+-----------------------+--------+----------+----------+")
    print("| Strategy              | Chunks | Avg Size | Max Size |")
    print("+-----------------------+--------+----------+----------+")

    for name, strategy, size in strategies:
        if size:
            chunker = SimpleChunker(chunk_size=size, strategy=strategy)
        else:
            chunker = SimpleChunker(strategy=strategy)

        chunks = chunker.chunk(SAMPLE_DOCUMENT)

        if chunks:
            avg_size = sum(len(c.content) for c in chunks) // len(chunks)
            max_size = max(len(c.content) for c in chunks)
        else:
            avg_size = 0
            max_size = 0

        print(f"| {name:<21} | {len(chunks):>6} | {avg_size:>8} | {max_size:>8} |")

    print("+-----------------------+--------+----------+----------+")

    print("\nKey Observations:")
    print("  - Character chunking: Predictable sizes, may break mid-word/sentence")
    print("  - Sentence chunking: Preserves sentence integrity")
    print("  - Paragraph chunking: Preserves semantic units")
    print("  - Section chunking: Preserves document structure")
    print("  - Token chunking: Good for LLM context limits")


def main():
    """Run all demos."""
    print("=" * 60)
    print("LLAMAFARM RAG - Chunking Strategies Demo")
    print("=" * 60)

    print(f"\nSample document length: {len(SAMPLE_DOCUMENT)} characters")
    print(f"Word count: {len(SAMPLE_DOCUMENT.split())} words")

    # Run each demo
    demo_character_chunking()
    demo_sentence_chunking()
    demo_paragraph_chunking()
    demo_section_chunking()
    demo_token_chunking()
    demo_metadata_preservation()

    # Compare all
    compare_all_strategies()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
