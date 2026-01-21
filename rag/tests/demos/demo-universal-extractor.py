#!/usr/bin/env python3
"""Demo script for UniversalExtractor.

This demo shows:
1. Extract comprehensive metadata from sample documents
2. Documents with all metadata fields populated
"""

import json
import sys
import tempfile
from pathlib import Path

# Add rag directory to path
rag_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rag_dir))

from components.extractors.universal_extractor import UniversalExtractor  # noqa: E402
from core.base import Document  # noqa: E402


def create_sample_documents() -> tuple[list[Document], Path]:
    """Create sample documents for demo."""
    # Create a temp file for source path testing
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, prefix="demo_"
    ) as tmp:
        tmp.write("# Sample Document\n\nThis is sample content.")
        source_path = Path(tmp.name)

    documents = [
        # Document with chunk metadata (simulating parser output)
        Document(
            content="""Machine learning is transforming industries worldwide.
            Companies are using AI to automate processes, analyze data, and
            improve customer experiences. Deep learning, a subset of machine
            learning, has enabled breakthroughs in computer vision and natural
            language processing.""",
            metadata={
                "chunk_index": 0,
                "total_chunks": 3,
                "author": "Dr. Jane Smith",
                "title": "AI Revolution",
            },
            source=str(source_path),
        ),
        # Document with markdown table
        Document(
            content="""# Performance Metrics

Here are the results:

| Model | Accuracy | Speed |
|-------|----------|-------|
| GPT-4 | 92%      | Fast  |
| BERT  | 88%      | Med   |

These results demonstrate significant improvements.""",
            metadata={
                "chunk_index": 1,
                "total_chunks": 3,
            },
            source=str(source_path),
        ),
        # Document with code
        Document(
            content="""# Code Example

Here's how to use the API:

```python
def train_model(data):
    model = MachineLearning()
    model.fit(data)
    return model.predict()
```

This function handles training and prediction.""",
            metadata={
                "chunk_index": 2,
                "total_chunks": 3,
            },
            source=str(source_path),
        ),
    ]

    return documents, source_path


def demo_extraction():
    """Demo UniversalExtractor capabilities."""
    print("=" * 60)
    print("UniversalExtractor Demo")
    print("=" * 60)

    # Create sample documents
    documents, source_path = create_sample_documents()

    try:
        # Initialize extractor with custom config
        extractor = UniversalExtractor(
            config={
                "keyword_count": 5,
                "generate_summary": True,
                "summary_sentences": 2,
            }
        )

        print(f"\nExtractor initialized: {extractor.name}")
        print(f"Keyword count: {extractor.keyword_count}")
        print(f"Summary sentences: {extractor.summary_sentences}")

        # Process documents
        print("\n" + "-" * 60)
        print("Processing documents...")
        print("-" * 60)

        enhanced_docs = extractor.extract(documents)

        for i, doc in enumerate(enhanced_docs):
            print(f"\n{'=' * 60}")
            print(f"Document {i + 1}")
            print("=" * 60)

            print(f"\nContent preview: {doc.content[:80]}...")

            print("\n--- Metadata ---")
            # Group metadata by category
            doc_level = {}
            chunk_level = {}
            features = {}
            other = {}

            for key, value in doc.metadata.items():
                if key.startswith("document_"):
                    doc_level[key] = value
                elif key.startswith("chunk_"):
                    chunk_level[key] = value
                elif key in [
                    "keywords",
                    "summary",
                    "language",
                    "has_tables",
                    "has_code",
                    "entities",
                ]:
                    features[key] = value
                else:
                    other[key] = value

            if doc_level:
                print("\nDocument-Level:")
                print(json.dumps(doc_level, indent=2, default=str))

            if chunk_level:
                print("\nChunk-Level:")
                print(json.dumps(chunk_level, indent=2, default=str))

            if features:
                print("\nExtracted Features:")
                print(json.dumps(features, indent=2, default=str))

            if other:
                print("\nOther Metadata:")
                print(json.dumps(other, indent=2, default=str))

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)

    finally:
        # Cleanup temp file
        source_path.unlink(missing_ok=True)


if __name__ == "__main__":
    demo_extraction()
