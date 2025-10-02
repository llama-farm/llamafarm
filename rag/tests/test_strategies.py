"""Essential strategy tests."""

from pathlib import Path

import pytest

from core.strategies.handler import SchemaHandler


class TestStrategies:
    """Core strategy functionality tests."""

    @pytest.fixture
    def test_config_path(self):
        """Get path to test config file."""
        return str(Path(__file__).parent / "test_data" / "test_strategies.yaml")

    def test_schema_loading(self, test_config_path):
        """Test loading schema from YAML file."""
        handler = SchemaHandler(test_config_path)

        assert handler.rag_config is not None
        assert (
            handler.rag_config.databases is not None
            or handler.rag_config.data_processing_strategies is not None
        )

    def test_get_available_strategies(self, test_config_path):
        """Test getting available strategies."""
        handler = SchemaHandler(test_config_path)

        available = handler.get_available_strategies()
        assert isinstance(available, list)
        assert len(available) > 0

    def test_strategy_name_parsing(self, test_config_path):
        """Test parsing strategy names."""
        handler = SchemaHandler(test_config_path)
        available = handler.get_available_strategies()

        if available:
            proc_name, db_name = handler.parse_strategy_name(available[0])
            assert proc_name is not None or db_name is not None

    def test_get_combined_config(self, test_config_path):
        """Test getting combined configuration."""
        handler = SchemaHandler(test_config_path)
        available = handler.get_available_strategies()

        if available:
            config = handler.get_combined_config(available[0])
            assert config is not None
            assert config.database and config.processing_strategy is not None

    def test_database_config(self, test_config_path):
        """Test database configuration."""
        handler = SchemaHandler(test_config_path)
        databases = handler.get_database_names()

        if databases:
            db_config = handler.create_database_config(databases[0])
            # New schema structure - check for 'config' instead of 'vector_store'
            assert db_config.config is not None
            assert db_config.embedding_strategies is not None
