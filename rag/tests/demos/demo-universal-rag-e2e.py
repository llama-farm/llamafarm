#!/usr/bin/env python3
"""Demo script for Universal RAG End-to-End Integration.

This demo shows:
1. Create a minimal config with only database (no data_processing_strategy)
2. Verify universal_rag is used by default
3. Process actual files through the pipeline
4. Verify documents have proper metadata

NOTE: This demo doesn't require any services to be running - it tests
the configuration and processing logic.
"""

import sys
import tempfile
from pathlib import Path

# Add rag directory to path
rag_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rag_dir))

import yaml  # noqa: E402


def create_minimal_config(temp_dir):
    """Create a minimal llamafarm.yaml with only database config."""
    config = {
        "version": "v1",
        "name": "demo_minimal",
        "namespace": "demo",
        "models": [{"provider": "local", "model": "llama3.1:8b"}],
        "prompts": [
            {"name": "default", "messages": [{"role": "system", "content": "Test"}]}
        ],
        "runtime": {
            "provider": "openai",
            "model": "llama3.1:8b",
            "api_key": "demo",
            "base_url": "http://localhost:11434/v1",
        },
        "rag": {
            "databases": [
                {
                    "name": "demo_db",
                    "type": "ChromaStore",
                    "config": {
                        "persist_directory": f"{temp_dir}/vectordb",
                        "collection_name": "demo",
                    },
                    "default_embedding_strategy": "demo_embed",
                    "embedding_strategies": [
                        {
                            "name": "demo_embed",
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        }
                    ],
                }
            ]
            # NOTE: NO data_processing_strategies - should use universal_rag default
        },
    }

    config_path = Path(temp_dir) / "llamafarm.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return str(config_path)


def create_sample_files(temp_dir):
    """Create sample files for processing."""
    files = []

    # Markdown file
    md_path = Path(temp_dir) / "sample.md"
    md_path.write_text("""# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience without being explicitly
programmed.

## Types of Machine Learning

There are three main types of machine learning:

1. **Supervised Learning**: The algorithm learns from labeled training data
2. **Unsupervised Learning**: The algorithm finds patterns in unlabeled data
3. **Reinforcement Learning**: The algorithm learns through trial and error

## Applications

Machine learning has many real-world applications including:
- Image recognition
- Natural language processing
- Recommendation systems
- Fraud detection
""")
    files.append(str(md_path))

    # Text file
    txt_path = Path(temp_dir) / "notes.txt"
    txt_path.write_text("""Natural Language Processing Notes

NLP is a branch of AI that helps computers understand human language.

Key concepts:
- Tokenization: Breaking text into words or subwords
- Embeddings: Vector representations of words
- Transformers: Modern architecture for sequence modeling
- Attention: Mechanism for focusing on relevant parts of input

Popular NLP models include BERT, GPT, and T5.
""")
    files.append(str(txt_path))

    return files


def demo_minimal_config_uses_universal_rag():
    """Demo that minimal config uses universal_rag by default."""
    print("=" * 60)
    print("Demo: Minimal Config Uses Universal RAG")
    print("=" * 60)

    from core.strategies.handler import SchemaHandler

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = create_minimal_config(temp_dir)
        print(f"\nConfig file: {config_path}")

        handler = SchemaHandler(config_path)

        # Check available strategies
        strategies = handler.get_data_processing_strategy_names()
        print(f"\nAvailable strategies: {strategies}")

        # Verify universal_rag is available
        if "universal_rag" in strategies:
            print("[PASS] universal_rag is available by default")
        else:
            print("[FAIL] universal_rag not in strategies")
            return False

        # Get default strategy
        default = handler.get_default_processing_strategy()
        print(f"\nDefault strategy: {default.name}")

        if default.name == "universal_rag":
            print("[PASS] Default strategy is universal_rag")
        else:
            print("[FAIL] Default strategy is not universal_rag")
            return False

        # Get the universal_rag strategy config
        universal = handler.create_processing_config("universal_rag")
        print("\n--- Universal RAG Strategy Config ---")
        print(f"Name: {universal.name}")
        print(f"Parsers: {[p.type for p in universal.parsers or []]}")
        print(f"Extractors: {[e.type for e in universal.extractors or []]}")

        if universal.parsers and universal.parsers[0].type == "UniversalParser":
            print("[PASS] UniversalParser is configured")
        else:
            print("[FAIL] UniversalParser not configured")
            return False

        if universal.extractors and universal.extractors[0].type == "UniversalExtractor":
            print("[PASS] UniversalExtractor is configured")
        else:
            print("[FAIL] UniversalExtractor not configured")
            return False

    print("\n" + "=" * 60)
    print("Minimal Config Demo Complete")
    print("=" * 60)
    return True


def demo_process_files_with_universal_rag():
    """Demo processing files through the universal_rag pipeline."""
    print("\n\n" + "=" * 60)
    print("Demo: Process Files with Universal RAG")
    print("=" * 60)

    from core.blob_processor import BlobProcessor
    from core.strategies.handler import SchemaHandler

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = create_minimal_config(temp_dir)
        sample_files = create_sample_files(temp_dir)

        print(f"\nSample files created: {sample_files}")

        handler = SchemaHandler(config_path)
        strategy = handler.create_processing_config("universal_rag")
        processor = BlobProcessor(strategy)

        print("\n--- Processing Files ---")

        total_docs = 0
        for file_path in sample_files:
            path = Path(file_path)
            print(f"\nProcessing: {path.name}")

            with open(file_path, "rb") as f:
                blob_data = f.read()

            documents = processor.process_blob(
                blob_data=blob_data,
                metadata={"filename": path.name, "content_type": "text/plain"},
            )

            print(f"  Chunks created: {len(documents)}")
            total_docs += len(documents)

            if documents:
                # Show first chunk info
                first_doc = documents[0]
                print("  First chunk metadata:")
                print(f"    - chunk_index: {first_doc.metadata.get('chunk_index')}")
                print(f"    - chunk_label: {first_doc.metadata.get('chunk_label')}")
                print(f"    - total_chunks: {first_doc.metadata.get('total_chunks')}")
                print(f"    - word_count: {first_doc.metadata.get('word_count')}")
                print(f"    - document_name: {first_doc.metadata.get('document_name')}")

        print("\n--- Summary ---")
        print(f"Files processed: {len(sample_files)}")
        print(f"Total chunks created: {total_docs}")

        if total_docs > 0:
            print("[PASS] Files processed successfully")
        else:
            print("[FAIL] No documents created")
            return False

    print("\n" + "=" * 60)
    print("Process Files Demo Complete")
    print("=" * 60)
    return True


def demo_document_metadata():
    """Demo that documents have proper metadata."""
    print("\n\n" + "=" * 60)
    print("Demo: Document Metadata Verification")
    print("=" * 60)

    from core.blob_processor import BlobProcessor
    from core.strategies.handler import SchemaHandler

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = create_minimal_config(temp_dir)
        sample_files = create_sample_files(temp_dir)

        handler = SchemaHandler(config_path)
        strategy = handler.create_processing_config("universal_rag")
        processor = BlobProcessor(strategy)

        # Process the markdown file
        md_file = sample_files[0]
        with open(md_file, "rb") as f:
            blob_data = f.read()

        documents = processor.process_blob(
            blob_data=blob_data,
            metadata={"filename": Path(md_file).name, "content_type": "text/markdown"},
        )

        if not documents:
            print("[FAIL] No documents created")
            return False

        print(f"\nProcessed {len(documents)} chunks from {Path(md_file).name}")
        print("\n--- Checking Required Metadata ---")

        # Required metadata fields
        required_fields = [
            "chunk_index",
            "chunk_label",
            "total_chunks",
            "chunk_position",
            "document_name",
            "document_type",
            "processed_at",
            "character_count",
            "word_count",
        ]

        first_doc = documents[0]
        all_present = True

        for field in required_fields:
            value = first_doc.metadata.get(field)
            if value is not None:
                print(f"[PASS] {field}: {value}")
            else:
                print(f"[FAIL] {field}: missing")
                all_present = False

        if all_present:
            print("\n[PASS] All required metadata present")
        else:
            print("\n[FAIL] Some metadata missing")
            return False

    print("\n" + "=" * 60)
    print("Document Metadata Demo Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success1 = demo_minimal_config_uses_universal_rag()
    success2 = demo_process_files_with_universal_rag()
    success3 = demo_document_metadata()

    if success1 and success2 and success3:
        print("\n\nAll E2E demos passed!")
        sys.exit(0)
    else:
        print("\n\nSome demos failed!")
        sys.exit(1)
