#!/usr/bin/env python3
"""Simple CLI for the RAG system."""

import argparse
import logging
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any
from colorama import init, Fore, Style

# Initialize colorama
init(autoreset=True)

from core.base import Pipeline
from core.enhanced_pipeline import EnhancedPipeline
from core.factories import (
    create_embedder_from_config,
    create_parser_from_config,
    create_vector_store_from_config,
    EmbedderFactory,
    VectorStoreFactory,
)
from utils.progress import LlamaProgressTracker, create_enhanced_progress_bar
from utils.path_resolver import PathResolver, resolve_paths_in_config


def setup_logging(level: str = "INFO"):
    """Setup logging configuration.

    TODO: Replace with global logging module when available.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_config(config_path: str, base_dir: str = None) -> Dict[str, Any]:
    """Load configuration from JSON file with flexible path resolution.

    Args:
        config_path: Path to configuration file (can be relative or absolute)
        base_dir: Base directory for relative path resolution

    TODO: Replace with top-level config lib when available.
    """
    resolver = PathResolver(base_dir)

    try:
        resolved_config_path = resolver.resolve_config_path(config_path)
        with open(resolved_config_path, "r") as f:
            config = json.load(f)

        # Resolve any paths within the configuration
        config = resolve_paths_in_config(config, resolver)

        return config
    except FileNotFoundError as e:
        print(f"Config file error: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON in config file: {e}")
        sys.exit(1)


def create_pipeline_from_config(
    config: Dict[str, Any], enhanced: bool = False, base_dir: str = None
) -> Pipeline:
    """Create pipeline from configuration using factories.

    Args:
        config: Configuration dictionary
        enhanced: Whether to use enhanced pipeline with progress tracking
        base_dir: Base directory for resolving relative paths in config
    """
    # Create components using factories
    parser = create_parser_from_config(config.get("parser", {}))
    embedder = create_embedder_from_config(config.get("embedder", {}))
    store = create_vector_store_from_config(config.get("vector_store", {}))

    # Create pipeline
    if enhanced:
        pipeline = EnhancedPipeline("🦙 Enhanced RAG Pipeline")
    else:
        pipeline = Pipeline("RAG Pipeline")

    pipeline.add_component(parser)
    pipeline.add_component(embedder)
    pipeline.add_component(store)

    return pipeline


def ingest_command(args):
    """Handle the ingest command."""
    setup_logging(args.log_level)
    tracker = LlamaProgressTracker()

    # Resolve data source path
    resolver = PathResolver(args.base_dir if hasattr(args, "base_dir") else None)
    try:
        source_path = resolver.resolve_data_source(args.source)
        tracker.print_info(f"📂 Data source resolved to: {source_path}")
    except FileNotFoundError as e:
        tracker.print_error(f"Data source error: {e}")
        sys.exit(1)

    # Load configuration
    try:
        config = load_config(
            args.config, args.base_dir if hasattr(args, "base_dir") else None
        )
    except Exception as e:
        tracker.print_error(f"Failed to load configuration: {e}")
        sys.exit(1)

    # Create enhanced pipeline
    try:
        pipeline = create_pipeline_from_config(
            config,
            enhanced=True,
            base_dir=args.base_dir if hasattr(args, "base_dir") else None,
        )
    except Exception as e:
        tracker.print_error(f"Failed to create pipeline: {e}")
        sys.exit(1)

    # Run ingestion with enhanced progress tracking
    try:
        if hasattr(pipeline, "run_with_progress"):
            result = pipeline.run_with_progress(source=str(source_path))
        else:
            # Fallback to regular pipeline
            tracker.print_info(f"📂 Processing documents from: {source_path}")
            result = pipeline.run(source=str(source_path))
            tracker.print_success("Processing completed!")

        # Show final summary
        print(f"\n📊 Final Results:")
        tracker.print_success(f"Documents processed: {len(result.documents)}")

        if result.errors:
            tracker.print_warning(f"Errors encountered: {len(result.errors)}")
            if args.log_level.upper() in ["DEBUG", "INFO"]:
                print("\n🔍 Error Details:")
                for i, error in enumerate(result.errors[:5], 1):
                    print(f"   {i}. {error}")
                if len(result.errors) > 5:
                    print(f"   ... and {len(result.errors) - 5} more errors")
        else:
            tracker.print_success("Zero errors - perfect execution! 🎯")

    except KeyboardInterrupt:
        tracker.print_warning("\n⏸️  Processing interrupted by user. No prob-llama!")
        sys.exit(0)
    except Exception as e:
        tracker.print_error(f"Ingestion failed: {e}")
        sys.exit(1)


def search_command(args):
    """Handle the search command."""
    setup_logging(args.log_level)
    tracker = LlamaProgressTracker()

    # Load configuration
    base_dir = getattr(args, "base_dir", None)
    config = load_config(args.config, base_dir)

    # Create embedder and vector store using factories
    try:
        embedder = create_embedder_from_config(config.get("embedder", {}))
        store = create_vector_store_from_config(config.get("vector_store", {}))
    except Exception as e:
        tracker.print_error(f"Failed to create components: {e}")
        sys.exit(1)

    # Perform search with llama flair
    tracker.print_header(f"🔍 Searching the Llama-sphere! 🔍")
    tracker.print_info(f"🎯 Query: '{args.query}'")
    tracker.print_info(f"🦙 {tracker.get_random_pun()}")

    try:
        # Embed the query
        print("🧠 Converting your query into llama-friendly embeddings...")
        query_embedding = embedder.embed([args.query])[0]
        tracker.print_success("Query embedded successfully!")

        # Search using the query embedding
        print("🔍 Searching through the knowledge pasture...")
        results = store.search(query_embedding=query_embedding, top_k=args.top_k)

        if results:
            tracker.print_success(f"Found {len(results)} llama-nificent matches!")
            print(f"\n📋 Search Results:")

            for i, doc in enumerate(results, 1):
                print(f"\n{Fore.CYAN}{'='*50}")
                print(f"🏆 Result #{i} - Document: {doc.id}")
                print(f"{'='*50}{Style.RESET_ALL}")

                print(f"📝 Content Preview:")
                print(
                    f"   {doc.content[:300]}{'...' if len(doc.content) > 300 else ''}"
                )

                if "similarity_score" in doc.metadata:
                    score = doc.metadata["similarity_score"]
                    if score > -100:
                        print(f"🎯 Similarity: {score:.3f} (Excellent match!)")
                    elif score > -300:
                        print(f"🎯 Similarity: {score:.3f} (Good match)")
                    else:
                        print(f"🎯 Similarity: {score:.3f} (Partial match)")

                if "priority" in doc.metadata:
                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                        doc.metadata["priority"], "⚪"
                    )
                    print(f"{priority_emoji} Priority: {doc.metadata['priority']}")

                if "tags" in doc.metadata:
                    print(f"🏷️  Tags: {doc.metadata['tags']}")
        else:
            tracker.print_warning(
                "No results found. The llamas are still grazing in other pastures! 🌾"
            )
            print("\n💡 Tips for better results:")
            print("   • Try different keywords")
            print("   • Use more specific terms")
            print("   • Check if your data has been ingested")

    except Exception as e:
        tracker.print_error(f"Search failed: {e}")
        sys.exit(1)


def info_command(args):
    """Handle the info command."""
    setup_logging(args.log_level)

    # Load configuration
    base_dir = getattr(args, "base_dir", None)
    config = load_config(args.config, base_dir)

    # Create vector store using factory
    try:
        store = create_vector_store_from_config(config.get("vector_store", {}))
    except Exception as e:
        print(f"Failed to create vector store: {e}")
        sys.exit(1)

    # Get info
    try:
        info = store.get_collection_info()
        print("Vector Store Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Failed to get info: {e}")
        sys.exit(1)


def test_command(args):
    """Handle the test command."""
    setup_logging(args.log_level)

    print("Testing RAG system components...")

    # Test Ollama connection
    print("\n1. Testing Ollama connection...")
    try:
        embedder = EmbedderFactory.create("OllamaEmbedder")
        embedder.validate_config()
        print("   ✓ Ollama is available")

        # Test embedding
        test_texts = ["Hello world", "This is a test"]
        embeddings = embedder.embed(test_texts)
        if embeddings and all(emb for emb in embeddings):
            print(f"   ✓ Embeddings generated (dimension: {len(embeddings[0])})")
        else:
            print("   ✗ Failed to generate embeddings")
    except Exception as e:
        print(f"   ✗ Ollama test failed: {e}")

    # Test ChromaDB
    print("\n2. Testing ChromaDB...")
    try:
        store = VectorStoreFactory.create(
            "ChromaStore", {"collection_name": "test_collection"}
        )
        store.validate_config()
        print("   ✓ ChromaDB is available")

        # Clean up test collection
        store.delete_collection()
        print("   ✓ Test collection cleaned up")
    except Exception as e:
        print(f"   ✗ ChromaDB test failed: {e}")

    # Test file parsing
    print("\n3. Testing file parsing...")
    try:
        if args.test_file:
            from core.factories import ParserFactory

            # Resolve test file path
            resolver = PathResolver(
                args.base_dir if hasattr(args, "base_dir") else None
            )
            try:
                test_file_path = resolver.resolve_data_source(args.test_file)
                print(f"   📁 Test file resolved to: {test_file_path}")
            except FileNotFoundError as e:
                print(f"   ✗ Test file not found: {e}")
                return

            # Detect file type and use appropriate parser
            file_extension = test_file_path.suffix.lower()
            if file_extension == '.csv':
                parser = ParserFactory.create("CustomerSupportCSVParser")
                test_type = "CSV"
            elif file_extension == '.pdf':
                parser = ParserFactory.create("PDFParser")
                test_type = "PDF"
            else:
                print(f"   ⚠ Unsupported file type: {file_extension}")
                return
            
            print(f"   🔍 Testing {test_type} parsing...")
            result = parser.parse(str(test_file_path))
            print(f"   ✓ Parsed {len(result.documents)} documents")
            if result.errors:
                print(f"   ⚠ {len(result.errors)} parsing errors")
                # Show first error for debugging
                if result.errors:
                    print(f"      First error: {result.errors[0].get('error', 'Unknown error')}")
        else:
            print("   ⚠ No test file provided (use --test-file)")
    except Exception as e:
        print(f"   ✗ File parsing test failed: {e}")

    print("\nTest completed!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Simple RAG System CLI")
    parser.add_argument(
        "--config",
        "-c",
        default="rag_config.json",
        help="Configuration file path (supports relative and absolute paths)",
    )
    parser.add_argument(
        "--base-dir",
        "-b",
        help="Base directory for resolving relative paths (defaults to current directory)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents")
    ingest_parser.add_argument(
        "source", help="Source file or directory (supports relative and absolute paths)"
    )

    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--top-k", type=int, default=5, help="Number of results to return"
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show vector store info")

    # Test command
    test_parser = subparsers.add_parser("test", help="Test system components")
    test_parser.add_argument(
        "--test-file",
        help="CSV file to test parsing (supports relative and absolute paths)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "ingest":
        ingest_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "info":
        info_command(args)
    elif args.command == "test":
        test_command(args)


if __name__ == "__main__":
    main()
