#!/usr/bin/env python3
"""Demo script for Backward Compatibility.

This demo shows:
1. Legacy parsers are still available and functional
2. Explicit strategies override the default universal_rag
3. Mixed configurations work correctly
4. Priority system works as expected

NOTE: This demo doesn't require any services to be running.
"""

import sys
import tempfile
from pathlib import Path

# Add rag directory to path
rag_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rag_dir))

import yaml  # noqa: E402


def demo_legacy_parsers_available():
    """Demo that legacy parsers are still available."""
    print("=" * 60)
    print("Demo: Legacy Parsers Still Available")
    print("=" * 60)

    from components.parsers.parser_factory import ToolAwareParserFactory

    # Check for legacy parsers
    legacy_parsers = [
        "PDFParser_PyPDF2",
        "PDFParser_LlamaIndex",
        "TextParser_Python",
        "TextParser_LlamaIndex",
        "CSVParser_Pandas",
        "CSVParser_Python",
        "MarkdownParser_Python",
        "DocxParser_PythonDocx",
        "ExcelParser_OpenPyXL",
    ]

    print("\n--- Checking Legacy Parser Availability ---")
    available_count = 0

    for parser_name in legacy_parsers:
        info = ToolAwareParserFactory.get_parser_info(parser_name)
        if info:
            print(f"[PASS] {parser_name} is available")
            available_count += 1
        else:
            print(f"[SKIP] {parser_name} not found (may not be installed)")

    print(f"\n{available_count}/{len(legacy_parsers)} legacy parsers available")

    if available_count > 5:
        print("[PASS] Most legacy parsers remain available")
        return True
    else:
        print("[FAIL] Too few legacy parsers available")
        return False


def demo_explicit_strategy_overrides_default():
    """Demo that explicit strategies override the default."""
    print("\n\n" + "=" * 60)
    print("Demo: Explicit Strategy Overrides Default")
    print("=" * 60)

    from core.strategies.handler import SchemaHandler

    # Create config with explicit legacy strategy
    config = {
        "version": "v1",
        "name": "explicit_test",
        "namespace": "test",
        "models": [{"provider": "local", "model": "llama3.1:8b"}],
        "prompts": [
            {"name": "default", "messages": [{"role": "system", "content": "Test"}]}
        ],
        "runtime": {
            "provider": "openai",
            "model": "llama3.1:8b",
            "api_key": "test",
            "base_url": "http://localhost:11434/v1",
        },
        "rag": {
            "databases": [
                {
                    "name": "test_db",
                    "type": "ChromaStore",
                    "config": {
                        "persist_directory": "./vectordb/test",
                        "collection_name": "test",
                    },
                    "default_embedding_strategy": "default_embed",
                    "embedding_strategies": [
                        {
                            "name": "default_embed",
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        }
                    ],
                }
            ],
            "data_processing_strategies": [
                {
                    "name": "custom_text_strategy",
                    "description": "Custom text processing",
                    "parsers": [
                        {
                            "type": "TextParser_Python",
                            "config": {"encoding": "utf-8"},
                        }
                    ],
                    "extractors": [],
                }
            ],
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        handler = SchemaHandler(str(config_path))

        # Get available strategies
        strategies = handler.get_data_processing_strategy_names()
        print(f"\nAvailable strategies: {strategies}")

        # Both custom and universal_rag should be available
        if "custom_text_strategy" in strategies and "universal_rag" in strategies:
            print("[PASS] Both custom and universal_rag strategies available")
        else:
            print("[FAIL] Missing expected strategies")
            return False

        # Get custom strategy
        custom = handler.create_processing_config("custom_text_strategy")
        print("\n--- Custom Strategy ---")
        print(f"Name: {custom.name}")
        print(f"Parser: {custom.parsers[0].type}")

        if custom.parsers[0].type == "TextParser_Python":
            print("[PASS] Custom strategy uses TextParser_Python (not UniversalParser)")
        else:
            print("[FAIL] Custom strategy should use TextParser_Python")
            return False

        # Get universal_rag strategy
        universal = handler.create_processing_config("universal_rag")
        print("\n--- Universal RAG Strategy ---")
        print(f"Name: {universal.name}")
        print(f"Parser: {universal.parsers[0].type}")

        if universal.parsers[0].type == "UniversalParser":
            print("[PASS] Universal strategy uses UniversalParser")
        else:
            print("[FAIL] Universal strategy should use UniversalParser")
            return False

    print("\n" + "=" * 60)
    print("Explicit Strategy Demo Complete")
    print("=" * 60)
    return True


def demo_priority_system():
    """Demo the parser priority system."""
    print("\n\n" + "=" * 60)
    print("Demo: Parser Priority System")
    print("=" * 60)

    from components.parsers.parser_factory import ToolAwareParserFactory

    # Get parser priorities
    parsers = ToolAwareParserFactory.list_parsers()

    print("\n--- Parser Priorities ---")
    priorities = []
    for parser_name in sorted(parsers):
        info = ToolAwareParserFactory.get_parser_info(parser_name)
        if info:
            priority = info.get("priority", 100)
            priorities.append((priority, parser_name))

    # Sort by priority
    priorities.sort()

    print("\nPriority (lower = higher priority):")
    for priority, name in priorities[:10]:  # Show top 10
        marker = " <-- Highest" if priority == priorities[0][0] else ""
        print(f"  Priority {priority:3d}: {name}{marker}")

    if len(priorities) > 10:
        print(f"  ... and {len(priorities) - 10} more parsers")

    # Verify UniversalParser has highest priority
    universal_priority = next(
        (p for p, n in priorities if n == "UniversalParser"), None
    )

    if universal_priority is not None:
        if universal_priority == priorities[0][0]:
            print(f"\n[PASS] UniversalParser has highest priority ({universal_priority})")
        elif universal_priority <= 10:
            print(f"\n[PASS] UniversalParser has high priority ({universal_priority})")
        else:
            print(f"\n[INFO] UniversalParser priority: {universal_priority}")
    else:
        print("\n[FAIL] UniversalParser not found")
        return False

    # Verify legacy parsers have priority 100
    legacy_check = [n for p, n in priorities if "LlamaIndex" in n or "PyPDF2" in n]
    if legacy_check:
        legacy_priorities = [p for p, n in priorities if n in legacy_check]
        if all(p == 100 for p in legacy_priorities):
            print("[PASS] Legacy parsers have default priority (100)")
        else:
            print(f"[INFO] Legacy parser priorities vary: {set(legacy_priorities)}")

    print("\n" + "=" * 60)
    print("Priority System Demo Complete")
    print("=" * 60)
    return True


def demo_mixed_config_processing():
    """Demo processing with mixed configurations."""
    print("\n\n" + "=" * 60)
    print("Demo: Mixed Configuration Processing")
    print("=" * 60)

    from core.blob_processor import BlobProcessor
    from core.strategies.handler import SchemaHandler

    config = {
        "version": "v1",
        "name": "mixed_test",
        "namespace": "test",
        "models": [{"provider": "local", "model": "llama3.1:8b"}],
        "prompts": [
            {"name": "default", "messages": [{"role": "system", "content": "Test"}]}
        ],
        "runtime": {
            "provider": "openai",
            "model": "llama3.1:8b",
            "api_key": "test",
            "base_url": "http://localhost:11434/v1",
        },
        "rag": {
            "databases": [
                {
                    "name": "test_db",
                    "type": "ChromaStore",
                    "config": {
                        "persist_directory": "./vectordb/test",
                        "collection_name": "test",
                    },
                    "default_embedding_strategy": "default_embed",
                    "embedding_strategies": [
                        {
                            "name": "default_embed",
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        }
                    ],
                }
            ],
            "data_processing_strategies": [
                {
                    "name": "legacy_text",
                    "description": "Legacy text processing",
                    "parsers": [
                        {"type": "TextParser_Python", "config": {}},
                    ],
                    "extractors": [],
                }
            ],
        },
    }

    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        handler = SchemaHandler(str(config_path))

        # Process same text with both strategies
        sample_text = b"This is a sample document for backward compatibility testing."

        print("\n--- Processing with Legacy Strategy ---")
        legacy_strategy = handler.create_processing_config("legacy_text")
        legacy_processor = BlobProcessor(legacy_strategy)
        legacy_docs = legacy_processor.process_blob(
            blob_data=sample_text,
            metadata={"filename": "test.txt", "content_type": "text/plain"},
        )
        print(f"Documents created: {len(legacy_docs)}")
        if legacy_docs:
            print(f"Content sample: {legacy_docs[0].content[:50]}...")
            print("[PASS] Legacy parser processed file successfully")
        else:
            print("[FAIL] No documents created")
            return False

        print("\n--- Processing with Universal Strategy ---")
        universal_strategy = handler.create_processing_config("universal_rag")
        universal_processor = BlobProcessor(universal_strategy)
        universal_docs = universal_processor.process_blob(
            blob_data=sample_text,
            metadata={"filename": "test.txt", "content_type": "text/plain"},
        )
        print(f"Documents created: {len(universal_docs)}")
        if universal_docs:
            print(f"Content sample: {universal_docs[0].content[:50]}...")
            print("[PASS] Universal parser processed file successfully")
        else:
            print("[FAIL] No documents created")
            return False

    print("\n" + "=" * 60)
    print("Mixed Config Demo Complete")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success1 = demo_legacy_parsers_available()
    success2 = demo_explicit_strategy_overrides_default()
    success3 = demo_priority_system()
    success4 = demo_mixed_config_processing()

    if success1 and success2 and success3 and success4:
        print("\n\nAll backward compatibility demos passed!")
        sys.exit(0)
    else:
        print("\n\nSome demos failed!")
        sys.exit(1)
