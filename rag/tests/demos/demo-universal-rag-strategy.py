#!/usr/bin/env python3
"""Demo script for universal_rag default strategy integration.

This demo shows:
1. Load a minimal config with only database (no strategies defined)
2. Verify universal_rag is used as the default strategy
3. Confirm UniversalParser and UniversalExtractor are active
"""

import sys
import tempfile
from pathlib import Path

import yaml

# Add rag directory to path
rag_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(rag_dir))

from core.strategies.handler import SchemaHandler  # noqa: E402


def create_minimal_config(tmp_dir: Path) -> Path:
    """Create a minimal config file with only database, no strategies."""
    config = {
        "version": "v1",
        "name": "test_project",
        "namespace": "default",
        "runtime": {
            "models": [
                {
                    "name": "default",
                    "provider": "ollama",
                    "model": "llama3.2",
                    "base_url": "http://localhost:11434",
                }
            ]
        },
        "rag": {
            "databases": [
                {
                    "name": "test_db",
                    "type": "ChromaStore",
                    "config": {
                        "collection_name": "test",
                        "persist_directory": str(tmp_dir / "chroma"),
                    },
                    "embedding_strategies": [
                        {
                            "name": "default",
                            "type": "OllamaEmbedder",
                            "config": {"model": "nomic-embed-text"},
                        }
                    ],
                    "default_embedding_strategy": "default",
                }
            ]
            # NOTE: No data_processing_strategies - should use universal_rag default
        },
    }

    config_file = tmp_dir / "llamafarm.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return config_file


def demo_default_strategy():
    """Demo default universal_rag strategy."""
    print("=" * 60)
    print("Universal RAG Default Strategy Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create minimal config
        config_file = create_minimal_config(tmp_path)
        print(f"\nCreated minimal config at: {config_file}")

        # Load with SchemaHandler
        print("\n" + "-" * 60)
        print("Loading config with SchemaHandler...")
        print("-" * 60)

        handler = SchemaHandler(str(config_file))

        # Check available strategies
        strategies = handler.get_data_processing_strategy_names()
        print(f"\nAvailable strategies: {strategies}")

        # Verify universal_rag is available
        if "universal_rag" in strategies:
            print("\n[PASS] universal_rag is available as a strategy")
        else:
            print("\n[FAIL] universal_rag is NOT available")
            return False

        # Get the default strategy
        print("\n" + "-" * 60)
        print("Getting default processing strategy...")
        print("-" * 60)

        default_strategy = handler.get_default_processing_strategy()
        print(f"\nDefault strategy name: {default_strategy.name}")
        print(f"Default strategy description: {default_strategy.description or 'N/A'}")

        # Check parsers
        print("\n--- Parsers ---")
        for parser in default_strategy.parsers or []:
            print(f"  Type: {parser.type}")
            print(f"  Config: {parser.config}")

        # Check extractors
        print("\n--- Extractors ---")
        for extractor in default_strategy.extractors or []:
            print(f"  Type: {extractor.type}")
            print(f"  Config: {extractor.config}")

        # Verify UniversalParser is configured
        parser_types = [p.type for p in (default_strategy.parsers or [])]
        if "UniversalParser" in parser_types:
            print("\n[PASS] UniversalParser is configured in default strategy")
        else:
            print(f"\n[FAIL] UniversalParser NOT found. Found: {parser_types}")
            return False

        # Verify UniversalExtractor is configured
        extractor_types = [e.type for e in (default_strategy.extractors or [])]
        if "UniversalExtractor" in extractor_types:
            print("[PASS] UniversalExtractor is configured in default strategy")
        else:
            print(f"[FAIL] UniversalExtractor NOT found. Found: {extractor_types}")
            return False

        # Try to create the processing config for universal_rag
        print("\n" + "-" * 60)
        print("Creating processing config for 'universal_rag'...")
        print("-" * 60)

        try:
            proc_config = handler.create_processing_config("universal_rag")
            print(f"\nSuccessfully created config for: {proc_config.name}")
            print("[PASS] create_processing_config('universal_rag') works")
        except Exception as e:
            print(f"\n[FAIL] Failed to create processing config: {e}")
            return False

        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("=" * 60)
        return True


def demo_explicit_strategy_override():
    """Demo that explicit strategies override the default."""
    print("\n\n" + "=" * 60)
    print("Explicit Strategy Override Demo")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create config WITH explicit strategy
        config = {
            "version": "v1",
            "name": "test_project",
            "namespace": "default",
            "runtime": {
                "models": [
                    {
                        "name": "default",
                        "provider": "ollama",
                        "model": "llama3.2",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test",
                            "persist_directory": str(tmp_path / "chroma"),
                        },
                        "embedding_strategies": [
                            {
                                "name": "default",
                                "type": "OllamaEmbedder",
                                "config": {"model": "nomic-embed-text"},
                            }
                        ],
                        "default_embedding_strategy": "default",
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "custom_strategy",
                        "parsers": [
                            {
                                "type": "PDFParser_PyPDF2",
                                "config": {"chunk_size": 2000},
                            }
                        ],
                    }
                ],
            },
        }

        config_file = tmp_path / "llamafarm.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        print(f"\nCreated config with explicit strategy at: {config_file}")

        # Load with SchemaHandler
        handler = SchemaHandler(str(config_file))

        # Check available strategies
        strategies = handler.get_data_processing_strategy_names()
        print(f"\nAvailable strategies: {strategies}")

        # Should have both custom_strategy AND universal_rag
        if "custom_strategy" in strategies and "universal_rag" in strategies:
            print("[PASS] Both custom_strategy and universal_rag are available")
        else:
            print(f"[FAIL] Expected both strategies. Found: {strategies}")
            return False

        # Get explicit strategy
        custom_config = handler.create_processing_config("custom_strategy")
        print(f"\nExplicit strategy: {custom_config.name}")

        parser_types = [p.type for p in (custom_config.parsers or [])]
        if "PDFParser_PyPDF2" in parser_types:
            print("[PASS] Explicit strategy uses PDFParser_PyPDF2")
        else:
            print(f"[FAIL] Expected PDFParser_PyPDF2. Found: {parser_types}")
            return False

        # Can still get universal_rag
        universal_config = handler.create_processing_config("universal_rag")
        print(f"\nUniversal strategy: {universal_config.name}")

        u_parser_types = [p.type for p in (universal_config.parsers or [])]
        if "UniversalParser" in u_parser_types:
            print("[PASS] universal_rag uses UniversalParser")
        else:
            print(f"[FAIL] Expected UniversalParser. Found: {u_parser_types}")
            return False

        print("\n" + "=" * 60)
        print("Override demo completed successfully!")
        print("=" * 60)
        return True


if __name__ == "__main__":
    success1 = demo_default_strategy()
    success2 = demo_explicit_strategy_override()

    if success1 and success2:
        print("\n\nAll demos passed!")
        sys.exit(0)
    else:
        print("\n\nSome demos failed!")
        sys.exit(1)
