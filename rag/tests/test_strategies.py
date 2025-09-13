"""Essential strategy tests."""

import pytest
from pathlib import Path

from rag.core.strategies.handler import SchemaHandler


class TestStrategies:
    """Core strategy functionality tests."""

    def test_schema_loading(self):
        """Test loading schema from YAML file."""
        handler = SchemaHandler("rag/tests/test_data/test_strategies.yaml")
        
        assert handler.rag_config is not None
        assert "databases" in handler.rag_config or "data_processing_strategies" in handler.rag_config

    def test_get_available_strategies(self):
        """Test getting available strategies."""
        handler = SchemaHandler("rag/tests/test_data/test_strategies.yaml")
        
        available = handler.get_available_strategies()
        assert isinstance(available, list)
        assert len(available) > 0

    def test_strategy_name_parsing(self):
        """Test parsing strategy names."""
        handler = SchemaHandler("rag/tests/test_data/test_strategies.yaml")
        available = handler.get_available_strategies()
        
        if available:
            proc_name, db_name = handler.parse_strategy_name(available[0])
            assert proc_name is not None or db_name is not None

    def test_get_combined_config(self):
        """Test getting combined configuration."""
        handler = SchemaHandler("rag/tests/test_data/test_strategies.yaml")
        available = handler.get_available_strategies()
        
        if available:
            config = handler.get_combined_config(available[0])
            assert config is not None
            assert "database" in config or "processing_strategy" in config

    def test_database_config(self):
        """Test database configuration."""
        handler = SchemaHandler("rag/tests/test_data/test_strategies.yaml")
        databases = handler.get_database_names()
        
        if databases:
            db_config = handler.create_database_config(databases[0])
            assert "vector_store" in db_config
            assert "embedding_strategies" in db_config