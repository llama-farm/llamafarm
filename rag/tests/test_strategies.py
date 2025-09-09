#!/usr/bin/env python3
"""
Tests for the strategy system with new RAG schema.
"""

import sys
from pathlib import Path
import unittest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.strategies.loader import StrategyLoader
from core.strategies.handler import SchemaHandler


def test_strategy_loading():
    """Test loading strategies from YAML file."""
    # Use an actual strategies file that exists
    loader = StrategyLoader(strategies_file='demos/demo_strategies.yaml')
    strategies = loader.load_strategies()
    
    # Should have loaded strategies
    assert strategies is not None
    assert 'databases' in strategies or 'data_processing_strategies' in strategies
    print(f"✓ Loaded strategies successfully")


def test_schema_handler_initialization():
    """Test SchemaHandler initialization with new schema."""
    # Test with default strategies file
    handler = SchemaHandler('demos/demo_strategies.yaml')
    
    # Get available strategies
    available = handler.get_available_strategies()
    assert len(available) > 0
    print(f"✓ Found {len(available)} strategies")
    
    # Test strategy name parsing
    proc_name, db_name = handler.parse_strategy_name(available[0])
    assert proc_name is not None or db_name is not None
    print(f"✓ Strategy name parsing works")


def test_get_combined_config():
    """Test getting combined configuration for a strategy."""
    handler = SchemaHandler('demos/demo_strategies.yaml')
    
    # Get available strategies
    available = handler.get_available_strategies()
    if available:
        strategy_name = available[0]
        config = handler.get_combined_config(strategy_name)
        
        assert config is not None
        assert 'database' in config or 'processing_strategy' in config
        print(f"✓ Combined config retrieved for {strategy_name}")


def test_create_component_config():
    """Test creating component configuration for CLI."""
    handler = SchemaHandler('demos/demo_strategies.yaml')
    
    # Get available strategies
    available = handler.get_available_strategies()
    if available:
        strategy_name = available[0]
        config = handler.create_component_config(strategy_name)
        
        assert config is not None
        assert 'version' in config
        assert 'rag' in config
        assert 'parsers' in config['rag']
        print(f"✓ Component config created for {strategy_name}")


def test_database_config():
    """Test getting database configuration."""
    handler = SchemaHandler('demos/demo_strategies.yaml')
    
    # Try to get a database config
    db_config = handler.get_database_config('main_database')
    if db_config:
        assert 'type' in db_config
        assert 'config' in db_config
        print(f"✓ Database config retrieved")


def test_processing_strategy_config():
    """Test getting processing strategy configuration."""
    handler = SchemaHandler('demos/demo_strategies.yaml')
    
    # Try to get a processing strategy config
    proc_config = handler.get_processing_strategy_config('research_papers_demo')
    if proc_config:
        assert 'name' in proc_config
        assert 'parsers' in proc_config or 'parser' in proc_config
        print(f"✓ Processing strategy config retrieved")


def test_embedder_config():
    """Test getting embedder configuration."""
    handler = SchemaHandler('demos/demo_strategies.yaml')
    
    # Get a database config first
    db_config = handler.get_database_config('main_database')
    if db_config:
        embedder_config = handler.get_embedder_config(db_config)
        
        assert embedder_config is not None
        assert 'type' in embedder_config
        assert 'config' in embedder_config
        print(f"✓ Embedder config retrieved")


def test_retrieval_strategy_config():
    """Test getting retrieval strategy configuration."""
    handler = SchemaHandler('demos/demo_strategies.yaml')
    
    # Get a database config first
    db_config = handler.get_database_config('main_database')
    if db_config:
        retrieval_config = handler.get_retrieval_strategy_config(db_config)
        
        assert retrieval_config is not None
        assert 'type' in retrieval_config
        assert 'config' in retrieval_config
        print(f"✓ Retrieval strategy config retrieved")


if __name__ == "__main__":
    print("Testing Strategy System with New RAG Schema")
    print("=" * 50)
    
    test_strategy_loading()
    test_schema_handler_initialization()
    test_get_combined_config()
    test_create_component_config()
    test_database_config()
    test_processing_strategy_config()
    test_embedder_config()
    test_retrieval_strategy_config()
    
    print("\n✅ All strategy tests passed!")