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
    
    def test_demo_strategies_available(self):
        """Test that all demo strategies are available."""
        from core.strategies.manager import StrategyManager
        
        manager = StrategyManager(load_demos=True)
        strategies = manager.get_available_strategies()
        
        expected_strategies = [
            "research_papers_demo",
            "customer_support_demo",
            "code_documentation_demo",
            "news_analysis_demo",
            "business_reports_demo"
        ]
        
        for strategy in expected_strategies:
            assert strategy in strategies, f"Strategy {strategy} not found"
    
    def test_strategy_loading(self):
        """Test that strategies can be loaded and configured."""
        from core.strategies.manager import StrategyManager
        
        manager = StrategyManager(load_demos=True)
        
        # Test loading a demo strategy
        config = manager.convert_strategy_to_config("news_analysis_demo")
        assert config is not None
        assert "parser" in config
        assert "embedder" in config
        assert "vector_store" in config
        assert "retrieval_strategy" in config
    
    def test_component_imports(self):
        """Test that all core components can be imported."""
        try:
            from core.factories import ComponentFactory, RetrievalStrategyFactory
            from core.strategies.manager import StrategyManager
            from components.stores.chroma_store.chroma_store import ChromaStore
            import cli
            
            assert ComponentFactory is not None
            assert RetrievalStrategyFactory is not None
            assert StrategyManager is not None
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
            "demos/demo_strategies.yaml"
        ]
        
        for demo_file in demo_files:
            path = Path(demo_file)
            assert path.exists(), f"Demo file {demo_file} does not exist"
    
    def test_chroma_store_metadata_parsing(self):
        """Test that ChromaStore correctly handles nested metadata."""
        from components.stores.chroma_store.chroma_store import ChromaStore
        
        config = {
            "collection_name": "test_metadata_parsing"
        }
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
        if hasattr(store, 'client'):
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
            "HybridUniversalStrategy"
        ]
        
        for strategy_name in strategies:
            strategy = RetrievalStrategyFactory.create(
                strategy_name, 
                {"top_k": 10}
            )
            assert strategy is not None, f"Failed to create {strategy_name}"
    
    def test_strategy_manager_handles_optional_fields(self):
        """Test that StrategyManager correctly handles optional fields."""
        from core.strategies.manager import StrategyManager
        
        manager = StrategyManager(load_demos=True)
        
        # Get info for a strategy without all optional fields
        info = manager.get_strategy_info("news_analysis_demo")
        
        assert info is not None
        assert "name" in info
        assert "description" in info
        assert "use_cases" in info
        assert "components" in info
        
        # Optional fields may or may not be present
        # The print methods should handle this gracefully
        try:
            # This should not raise an error even if fields are missing
            import io
            from contextlib import redirect_stdout
            
            f = io.StringIO()
            with redirect_stdout(f):
                manager.print_strategy_summary("news_analysis_demo")
            
            output = f.getvalue()
            assert "news_analysis_demo" in output
        except KeyError as e:
            pytest.fail(f"print_strategy_summary failed with missing field: {e}")
    
    def test_cli_based_demos_syntax(self):
        """Test that CLI-based demos have correct command syntax."""
        demo_files = [
            "demos/demo4_news_analysis.py",
            "demos/demo5_business_reports.py"
        ]
        
        for demo_file in demo_files:
            with open(demo_file, 'r') as f:
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
    
    @pytest.mark.integration
    def test_end_to_end_strategy_pipeline(self):
        """Test an end-to-end pipeline using a strategy."""
        from core.strategies.manager import StrategyManager
        from core.factories import ComponentFactory
        from core.base import Document
        
        # Load strategy
        manager = StrategyManager(load_demos=True)
        config = manager.convert_strategy_to_config("news_analysis_demo")
        
        assert config is not None
        
        # Verify the strategy configuration has all required components
        # Allow DirectoryParser for news_analysis_demo when strategy aggregates directory inputs
        assert config["parser"]["type"] in ("HTMLParser", "DirectoryParser")
        assert config["embedder"]["type"] == "OllamaEmbedder"
        assert config["vector_store"]["type"] == "ChromaStore"
        assert config["retrieval_strategy"]["type"] == "RerankedStrategy"
        
        # Verify extractors are configured
        assert "extractors" in config
        assert len(config["extractors"]) > 0
        
        extractor_types = [e["type"] for e in config["extractors"]]
        assert "EntityExtractor" in extractor_types
        assert "DateTimeExtractor" in extractor_types
        assert "SummaryExtractor" in extractor_types


class TestCLICommands:
    """Test CLI command functionality."""
    
    def test_cli_help_commands(self):
        """Test that CLI help commands work."""
        commands = [
            ["python", "cli.py", "-h"],
            ["python", "cli.py", "ingest", "-h"],
            ["python", "cli.py", "search", "-h"],
            ["python", "cli.py", "strategies", "-h"],
        ]
        
        for cmd in commands:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,  # Increased timeout for CI environment
                    cwd=Path(__file__).parent.parent
                )
                assert result.returncode == 0, f"Command {' '.join(cmd)} failed"
                assert "usage:" in result.stdout.lower() or "usage:" in result.stderr.lower()
            except subprocess.TimeoutExpired:
                pytest.fail(f"Command {' '.join(cmd)} timed out after 30 seconds")
    
    def test_cli_strategies_list(self):
        """Test that CLI can list strategies."""
        result = subprocess.run(
            ["python", "cli.py", "strategies", "list"],
            capture_output=True,
            text=True,
            timeout=30,  # Increased timeout for CI environment
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        output = result.stdout
        
        # Check for demo strategies
        assert "news_analysis_demo" in output
        assert "business_reports_demo" in output
        assert "code_documentation_demo" in output