"""Backward Compatibility Tests for Universal RAG.

This test file verifies:
1. Existing parsers (PDFParser_PyPDF2, TextParser_LlamaIndex) still work
2. Explicit data_processing_strategy in config overrides default
3. Legacy parser configs continue to function correctly
4. Mixed strategy configs work (some files universal, some legacy)
"""

from pathlib import Path

import pytest
import yaml

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def config_with_legacy_parsers():
    """Config using legacy parsers explicitly."""
    return {
        "version": "v1",
        "name": "legacy_test",
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
                    "name": "legacy_db",
                    "type": "ChromaStore",
                    "config": {
                        "persist_directory": "./vectordb/legacy",
                        "collection_name": "legacy",
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
                    "name": "legacy_text_strategy",
                    "description": "Legacy text processing with TextParser_Python",
                    "parsers": [
                        {
                            "type": "TextParser_Python",
                            "config": {"encoding": "utf-8"},
                        }
                    ],
                    "extractors": [],
                },
                {
                    "name": "legacy_pdf_strategy",
                    "description": "Legacy PDF processing with PDFParser_PyPDF2",
                    "parsers": [
                        {
                            "type": "PDFParser_PyPDF2",
                            "config": {},
                        }
                    ],
                    "extractors": [],
                },
            ],
        },
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestLegacyParsersStillWork:
    """Test that legacy parsers still function correctly."""

    def test_pdf_parser_pypdf2_info_available(self):
        """Test: PDFParser_PyPDF2 info is available in factory."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("PDFParser_PyPDF2")

        assert info is not None
        assert info["name"] == "PDFParser_PyPDF2"

    def test_text_parser_python_info_available(self):
        """Test: TextParser_Python info is available in factory."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("TextParser_Python")

        assert info is not None
        assert info["name"] == "TextParser_Python"

    def test_csv_parser_pandas_info_available(self):
        """Test: CSVParser_Pandas info is available in factory."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        info = ToolAwareParserFactory.get_parser_info("CSVParser_Pandas")

        assert info is not None
        assert info["name"] == "CSVParser_Pandas"

    def test_legacy_parsers_listed_in_factory(self):
        """Test: All legacy parsers are listed in factory."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        parsers = ToolAwareParserFactory.list_parsers()

        legacy_parsers = [
            "PDFParser_PyPDF2",
            "PDFParser_LlamaIndex",
            "TextParser_Python",
            "TextParser_LlamaIndex",
            "CSVParser_Pandas",
            "CSVParser_Python",
        ]

        for parser in legacy_parsers:
            assert parser in parsers, f"{parser} should be in factory"


class TestExplicitStrategyOverridesDefault:
    """Test that explicit strategies override the default universal_rag."""

    def test_explicit_text_strategy_uses_text_parser(
        self, temp_dir, config_with_legacy_parsers
    ):
        """Test: Explicit text strategy uses TextParser_Python."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_legacy_parsers, f)

        handler = SchemaHandler(str(config_path))

        # Get the explicit legacy strategy
        strategy = handler.create_processing_config("legacy_text_strategy")

        assert strategy is not None
        assert strategy.name == "legacy_text_strategy"
        assert strategy.parsers[0].type == "TextParser_Python"

    def test_explicit_pdf_strategy_uses_pdf_parser(
        self, temp_dir, config_with_legacy_parsers
    ):
        """Test: Explicit PDF strategy uses PDFParser_PyPDF2."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_legacy_parsers, f)

        handler = SchemaHandler(str(config_path))

        # Get the explicit legacy strategy
        strategy = handler.create_processing_config("legacy_pdf_strategy")

        assert strategy is not None
        assert strategy.name == "legacy_pdf_strategy"
        assert strategy.parsers[0].type == "PDFParser_PyPDF2"

    def test_universal_rag_still_available_with_legacy_strategies(
        self, temp_dir, config_with_legacy_parsers
    ):
        """Test: universal_rag is still available when legacy strategies defined."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_legacy_parsers, f)

        handler = SchemaHandler(str(config_path))

        # All strategies should be available
        strategies = handler.get_data_processing_strategy_names()

        assert "universal_rag" in strategies
        assert "legacy_text_strategy" in strategies
        assert "legacy_pdf_strategy" in strategies


class TestMixedStrategyConfigs:
    """Test mixed configurations with both universal and legacy strategies."""

    def test_can_switch_between_strategies(
        self, temp_dir, config_with_legacy_parsers
    ):
        """Test: Can switch between universal and legacy strategies."""
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_legacy_parsers, f)

        handler = SchemaHandler(str(config_path))

        # Get universal strategy
        universal = handler.create_processing_config("universal_rag")
        assert universal.parsers[0].type == "UniversalParser"

        # Get legacy strategy
        legacy = handler.create_processing_config("legacy_text_strategy")
        assert legacy.parsers[0].type == "TextParser_Python"

        # Both work without conflicts
        assert universal.name != legacy.name


class TestParserRegistryBackwardCompat:
    """Test parser registry maintains backward compatibility."""

    def test_registry_contains_legacy_parsers(self):
        """Test: Parser registry contains all legacy parsers."""
        from components.parsers.parser_registry import ParserRegistry

        registry = ParserRegistry()
        all_parsers = registry.list_all_parsers()

        # Should have multiple legacy parsers
        assert len(all_parsers) > 0

        # Check for specific legacy parsers
        pdf_parsers = [p for p in all_parsers if "PDF" in p]
        text_parsers = [p for p in all_parsers if "Text" in p]

        assert len(pdf_parsers) > 0, "Should have PDF parsers"
        assert len(text_parsers) > 0, "Should have Text parsers"

    def test_registry_does_not_break_with_universal_parser(self):
        """Test: Registry works correctly with UniversalParser in the mix."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        # Discover all parsers - should not raise
        parsers = ToolAwareParserFactory.list_parsers()

        # Should have both universal and legacy
        assert "UniversalParser" in parsers

        # Legacy parsers should still be there
        legacy_found = any("PyPDF2" in p or "LlamaIndex" in p for p in parsers)
        assert legacy_found, "Legacy parsers should still be discoverable"


class TestBlobProcessorBackwardCompat:
    """Test BlobProcessor backward compatibility."""

    def test_blob_processor_initializes_with_legacy_strategy(
        self, temp_dir, config_with_legacy_parsers
    ):
        """Test: BlobProcessor initializes with legacy strategy."""
        from core.blob_processor import BlobProcessor
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_legacy_parsers, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("legacy_text_strategy")

        # Should not raise
        processor = BlobProcessor(strategy)

        assert processor is not None

    def test_blob_processor_processes_with_legacy_parser(
        self, temp_dir, config_with_legacy_parsers
    ):
        """Test: BlobProcessor processes files with legacy TextParser."""
        from core.blob_processor import BlobProcessor
        from core.strategies.handler import SchemaHandler

        config_path = Path(temp_dir) / "llamafarm.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_with_legacy_parsers, f)

        handler = SchemaHandler(str(config_path))
        strategy = handler.create_processing_config("legacy_text_strategy")
        processor = BlobProcessor(strategy)

        # Create sample text
        sample_text = "This is a simple test document for backward compatibility."

        # Process with legacy parser
        documents = processor.process_blob(
            blob_data=sample_text.encode("utf-8"),
            metadata={"filename": "test.txt", "content_type": "text/plain"},
        )

        assert documents is not None
        assert len(documents) > 0


class TestPrioritySystem:
    """Test parser priority system works correctly."""

    def test_universal_parser_has_higher_priority(self):
        """Test: UniversalParser has higher priority than legacy parsers."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        universal_info = ToolAwareParserFactory.get_parser_info("UniversalParser")
        legacy_info = ToolAwareParserFactory.get_parser_info("TextParser_Python")

        universal_priority = universal_info.get("priority", 100)
        legacy_priority = legacy_info.get("priority", 100)

        # Lower number = higher priority
        assert universal_priority < legacy_priority

    def test_legacy_parsers_have_default_priority(self):
        """Test: Legacy parsers have default priority of 100."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        legacy_parsers = ["PDFParser_PyPDF2", "TextParser_Python", "CSVParser_Pandas"]

        for parser_name in legacy_parsers:
            info = ToolAwareParserFactory.get_parser_info(parser_name)
            if info:
                priority = info.get("priority", 100)
                assert priority == 100, f"{parser_name} should have priority 100"
