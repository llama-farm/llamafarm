#!/usr/bin/env python3
"""
Tests for memory configuration models.

Phase 9: Data Model & Configuration Schema
"""

import sys
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config_dict
from datamodel import (
    # Memory store configs - use the inline types for MemoryConfig.stores
    Consolidation as ConsolidationConfig,
)
from datamodel import (
    # Dataset type
    Dataset,
    LlamaFarmConfig,
    MemoryConfig,
)
from datamodel import (
    Graph as GraphConfig,
)
from datamodel import (
    Store as MemoryStoreConfig,
)
from datamodel import (
    Timeseries as TimeSeriesConfig,
)
from datamodel import (
    WorkingMemory as WorkingMemoryConfig,
)


class TestMemoryConfigModels:
    """Test memory configuration Pydantic models."""

    def test_working_memory_config_defaults(self):
        """Test WorkingMemoryConfig with default values."""
        config = WorkingMemoryConfig()
        assert config.ttl_seconds == 3600
        assert config.max_records == 10000

    def test_working_memory_config_custom(self):
        """Test WorkingMemoryConfig with custom values."""
        config = WorkingMemoryConfig(ttl_seconds=7200, max_records=5000)
        assert config.ttl_seconds == 7200
        assert config.max_records == 5000

    def test_timeseries_config_defaults(self):
        """Test TimeSeriesConfig with default values."""
        config = TimeSeriesConfig()
        assert config.retention_days == 30

    def test_timeseries_config_custom(self):
        """Test TimeSeriesConfig with custom values."""
        config = TimeSeriesConfig(retention_days=90)
        assert config.retention_days == 90

    def test_graph_config_defaults(self):
        """Test GraphConfig with default values."""
        config = GraphConfig()
        assert config.max_path_depth == 10

    def test_graph_config_custom(self):
        """Test GraphConfig with custom values."""
        config = GraphConfig(max_path_depth=20)
        assert config.max_path_depth == 20

    def test_consolidation_config_defaults(self):
        """Test ConsolidationConfig with default values."""
        config = ConsolidationConfig()
        assert config.min_records == 10
        assert config.batch_size == 100
        assert config.prune_after_consolidate is True

    def test_consolidation_config_custom(self):
        """Test ConsolidationConfig with custom values."""
        config = ConsolidationConfig(
            min_records=50, batch_size=200, prune_after_consolidate=False
        )
        assert config.min_records == 50
        assert config.batch_size == 200
        assert config.prune_after_consolidate is False


class TestMemoryStoreConfig:
    """Test MemoryStoreConfig validation."""

    def test_valid_name(self):
        """Test valid memory store names."""
        valid_names = ["brain_memory", "shortterm", "memory1", "a"]
        for name in valid_names:
            config = MemoryStoreConfig(name=name)
            assert config.name == name

    def test_invalid_name_uppercase(self):
        """Test that uppercase names are rejected."""
        with pytest.raises(ValidationError):
            MemoryStoreConfig(name="BrainMemory")

    def test_invalid_name_starts_with_number(self):
        """Test that names starting with number are rejected."""
        with pytest.raises(ValidationError):
            MemoryStoreConfig(name="1memory")

    def test_invalid_name_special_chars(self):
        """Test that names with special characters are rejected."""
        with pytest.raises(ValidationError):
            MemoryStoreConfig(name="brain-memory")

    def test_invalid_name_too_long(self):
        """Test that names exceeding 50 characters are rejected."""
        with pytest.raises(ValidationError):
            MemoryStoreConfig(name="a" * 51)

    def test_memory_store_with_all_configs(self):
        """Test MemoryStoreConfig with all sub-configurations."""
        config = MemoryStoreConfig(
            name="brain_memory",
            working_memory=WorkingMemoryConfig(ttl_seconds=1800),
            timeseries=TimeSeriesConfig(retention_days=60),
            graph=GraphConfig(max_path_depth=15),
            consolidation=ConsolidationConfig(min_records=20),
        )
        assert config.name == "brain_memory"
        assert config.working_memory.ttl_seconds == 1800
        assert config.timeseries.retention_days == 60
        assert config.graph.max_path_depth == 15
        assert config.consolidation.min_records == 20

    def test_memory_store_with_no_sub_configs(self):
        """Test MemoryStoreConfig with only name (uses defaults)."""
        config = MemoryStoreConfig(name="minimal_store")
        assert config.name == "minimal_store"
        assert config.working_memory is None
        assert config.timeseries is None
        assert config.graph is None
        assert config.consolidation is None


class TestMemoryConfig:
    """Test MemoryConfig model."""

    def test_empty_stores(self):
        """Test MemoryConfig with no stores."""
        config = MemoryConfig()
        assert config.stores == []
        assert config.default_store is None

    def test_single_store(self):
        """Test MemoryConfig with a single store."""
        config = MemoryConfig(
            stores=[MemoryStoreConfig(name="main_memory")], default_store="main_memory"
        )
        assert len(config.stores) == 1
        assert config.stores[0].name == "main_memory"
        assert config.default_store == "main_memory"

    def test_multiple_stores(self):
        """Test MemoryConfig with multiple stores."""
        config = MemoryConfig(
            stores=[
                MemoryStoreConfig(name="shortterm"),
                MemoryStoreConfig(name="longterm"),
            ],
            default_store="shortterm",
        )
        assert len(config.stores) == 2
        assert config.default_store == "shortterm"


class TestDatasetWithMemory:
    """Test Dataset model with optional memory field."""

    def test_dataset_without_memory(self):
        """Test Dataset without memory field."""
        dataset = Dataset(
            name="test_dataset", data_processing_strategy="default", database="main_db"
        )
        assert dataset.name == "test_dataset"
        assert dataset.memory is None

    def test_dataset_with_memory(self):
        """Test Dataset with memory field."""
        dataset = Dataset(
            name="test_dataset",
            data_processing_strategy="default",
            database="main_db",
            memory="brain_memory",
        )
        assert dataset.name == "test_dataset"
        assert dataset.memory == "brain_memory"


class TestLlamaFarmConfigWithMemory:
    """Test LlamaFarmConfig with memory configuration."""

    def test_config_without_memory(self):
        """Test LlamaFarmConfig without memory section."""
        config_dict = {
            "version": "v1",
            "name": "test_project",
            "namespace": "test_namespace",
            "runtime": {
                "models": [
                    {"name": "test_model", "provider": "openai", "model": "gpt-4"}
                ]
            },
        }
        config = LlamaFarmConfig(**config_dict)
        assert config.memory is None

    def test_config_with_memory(self):
        """Test LlamaFarmConfig with memory section."""
        config_dict = {
            "version": "v1",
            "name": "test_project",
            "namespace": "test_namespace",
            "runtime": {
                "models": [
                    {"name": "test_model", "provider": "openai", "model": "gpt-4"}
                ]
            },
            "memory": {
                "stores": [
                    {
                        "name": "brain_memory",
                        "working_memory": {"ttl_seconds": 3600},
                        "timeseries": {"retention_days": 30},
                    }
                ],
                "default_store": "brain_memory",
            },
        }
        config = LlamaFarmConfig(**config_dict)
        assert config.memory is not None
        assert len(config.memory.stores) == 1
        assert config.memory.stores[0].name == "brain_memory"
        assert config.memory.default_store == "brain_memory"


class TestMemoryConfigYAMLLoading:
    """Test loading memory configuration from YAML files."""

    def test_load_yaml_with_memory_config(self):
        """Test loading a YAML file with memory configuration."""
        yaml_content = """
version: v1
name: test_project
namespace: test_namespace

memory:
  stores:
    - name: scenario_memory
      working_memory:
        ttl_seconds: 3600
        max_records: 10000
      timeseries:
        retention_days: 30
      graph:
        max_path_depth: 10
      consolidation:
        min_records: 10
        batch_size: 100
        prune_after_consolidate: true
  default_store: scenario_memory

datasets:
  - name: military_protocols
    data_processing_strategy: default_strategy
    database: semantic_memory
    memory: scenario_memory

rag:
  databases:
    - name: semantic_memory
      type: ChromaStore
      embedding_strategies:
        - name: default
          type: SentenceTransformerEmbedder
          config:
            model: all-MiniLM-L6-v2
  data_processing_strategies:
    - name: default_strategy
      parsers:
        - type: TextParser_LlamaIndex

runtime:
  models:
    - name: test_model
      provider: openai
      model: gpt-4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config_dict(config_path=temp_path)

            # Verify memory configuration was loaded
            assert "memory" in config
            assert config["memory"]["default_store"] == "scenario_memory"
            assert len(config["memory"]["stores"]) == 1

            store = config["memory"]["stores"][0]
            assert store["name"] == "scenario_memory"
            assert store["working_memory"]["ttl_seconds"] == 3600
            assert store["timeseries"]["retention_days"] == 30
            assert store["graph"]["max_path_depth"] == 10
            assert store["consolidation"]["min_records"] == 10

            # Verify dataset with memory reference
            dataset = config["datasets"][0]
            assert dataset["memory"] == "scenario_memory"

        finally:
            import os

            os.unlink(temp_path)

    def test_load_yaml_without_memory_config(self):
        """Test loading a YAML file without memory configuration."""
        yaml_content = """
version: v1
name: test_project
namespace: test_namespace

rag:
  databases:
    - name: main_db
      type: ChromaStore
      embedding_strategies:
        - name: default
          type: SentenceTransformerEmbedder
          config:
            model: all-MiniLM-L6-v2
  data_processing_strategies:
    - name: default_strategy
      parsers:
        - type: TextParser_LlamaIndex

runtime:
  models:
    - name: test_model
      provider: openai
      model: gpt-4
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config_dict(config_path=temp_path)

            # Memory should not be present or be None/empty
            memory = config.get("memory")
            assert memory is None or memory == {}

        finally:
            import os

            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
