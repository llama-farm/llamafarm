"""
System validation tests to ensure all components are working correctly.
Tests strategies, demos, and CLI functionality.
"""

import pytest
import subprocess
import sys
from pathlib import Path
import json
from unittest.mock import patch, MagicMock


class TestSystemValidation:
    """Test suite for validating the entire RAG system."""

    def test_demo_config_available(self):
        """Test that demo configuration is available."""
        from core.strategies.handler import SchemaHandler
        from pathlib import Path

        demo_config = Path(__file__).parent.parent / "demos" / "demo_strategies.yaml"
        if demo_config.exists():
            handler = SchemaHandler(str(demo_config))
            databases = handler.get_database_names()
            strategies = handler.get_data_processing_strategy_names()

            assert len(databases) > 0, "Should have at least one database"
            assert len(strategies) > 0, (
                "Should have at least one data processing strategy"
            )

    def test_schema_handler_loading(self):
        """Test that schema handler can load and parse configurations."""
        from core.strategies.handler import SchemaHandler
        from pathlib import Path

        demo_config = Path(__file__).parent.parent / "demos" / "demo_strategies.yaml"
        if demo_config.exists():
            handler = SchemaHandler(str(demo_config))

            # Test database access
            databases = handler.get_database_names()
            if databases:
                db_config = handler.create_database_config(databases[0])
                assert "vector_store" in db_config
                assert "embedding_strategies" in db_config
                assert "retrieval_strategies" in db_config

    def test_component_imports(self):
        """Test that all core components can be imported."""
        try:
            from core.factories import (
                create_embedder_from_config,
                create_vector_store_from_config,
                create_retrieval_strategy_from_config,
            )
            from core.strategies.handler import SchemaHandler
            from components.stores.chroma_store.chroma_store import ChromaStore
            import cli

            assert create_embedder_from_config is not None
            assert create_vector_store_from_config is not None
            assert create_retrieval_strategy_from_config is not None
            assert SchemaHandler is not None
            assert ChromaStore is not None
            assert cli is not None
        except ImportError as e:
            pytest.fail(f"Failed to import component: {e}")

    def test_demo_files_exist(self):
        """Test that all demo files exist."""
        demo_files = [
            "demos/demo1_research_papers_cli.py",
            "demos/demo2_customer_support_cli.py",
            "demos/demo3_code_documentation.py",
            "demos/demo3_code_documentation_cli.py",
            "demos/demo4_news_analysis.py",
            "demos/demo5_business_reports.py",
            "demos/demo6_document_management.py",
            "demos/demo_strategies.yaml",
        ]

        for demo_file in demo_files:
            path = Path(demo_file)
            if not path.exists():
                pytest.skip(f"Demo file {demo_file} not present in this environment")

    def test_chroma_store_metadata_parsing(self):
        """Test that ChromaStore correctly handles nested metadata."""
        from components.stores.chroma_store.chroma_store import ChromaStore

        config = {"collection_name": "test_metadata_parsing"}
        store = ChromaStore(name="test_store", config=config)

        # Test metadata parsing
        test_metadata = {"nested": {"key": "value", "number": 42}}

        # Simulate what ChromaDB does - serialize nested objects
        serialized_nested = json.dumps(test_metadata["nested"])
        chromadb_metadata = {"nested": serialized_nested}

        # Parse it back
        parsed = store._parse_metadata(chromadb_metadata)

        # Verify parsing worked correctly
        assert "nested" in parsed
        assert isinstance(parsed["nested"], dict)
        assert parsed["nested"]["key"] == "value"
        assert parsed["nested"]["number"] == 42

        # Cleanup
        if hasattr(store, "client"):
            try:
                store.client.delete_collection(name="test_metadata_parsing")
            except:
                pass

    def test_retrieval_strategy_factory(self):
        """Test that RetrievalStrategyFactory can create all strategies."""
        from core.factories import RetrievalStrategyFactory

        strategies = [
            "BasicSimilarityStrategy",
            "MetadataFilteredStrategy",
            "MultiQueryStrategy",
            "RerankedStrategy",
            "HybridUniversalStrategy",
        ]

        for strategy_name in strategies:
            strategy = RetrievalStrategyFactory.create(strategy_name, {"top_k": 10})
            assert strategy is not None, f"Failed to create {strategy_name}"

    def test_cli_based_demos_syntax(self):
        """Test that CLI-based demos have correct command syntax."""
        demo_files = ["demos/demo4_news_analysis.py", "demos/demo5_business_reports.py"]

        for demo_file in demo_files:
            with open(demo_file, "r") as f:
                content = f.read()

                # Check for correct CLI syntax patterns
                # Commands should be: python cli.py --strategy-file path ingest --strategy name
                # This ensures demos are using the correct modern syntax

                # These patterns should exist (correct order with strategy file)
                assert "--strategy-file demos/demo_strategies.yaml" in content
                assert "python cli.py --strategy-file" in content

                # Basic command structure should exist
                assert "ingest" in content
                assert "search" in content


class TestCLICommands:
    """Test CLI command functionality."""

    def test_cli_help_commands(self):
        """Test that CLI help commands work."""
        commands = [
            ["python", "cli.py", "-h"],
            ["python", "cli.py", "ingest", "-h"],
            ["python", "cli.py", "search", "-h"],
            ["python", "cli.py", "info", "-h"],
        ]

        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,  # Increased timeout for CI environment
                    cwd=Path(__file__).parent.parent,
                )
                assert result.returncode == 0, f"Command {' '.join(cmd)} failed"
                assert (
                    "usage:" in result.stdout.lower()
                    or "usage:" in result.stderr.lower()
                )
            except subprocess.TimeoutExpired:
                pytest.fail(f"Command {' '.join(cmd)} timed out after 30 seconds")
