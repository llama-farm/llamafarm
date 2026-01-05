#!/usr/bin/env python3
"""
Tests for dataset type system and unified dataset configuration.

Phase 16: Dataset Type System & Schema Updates
"""

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config_dict
from datamodel import (
    # Dataset inline types (used by Dataset model directly)
    Consolidation1 as DatasetConsolidation,
)
from datamodel import (
    Dataset,
    # Dataset definition types (from definitions section)
    DatasetConsolidationConfig,
    DatasetGraphConfig,
    DatasetSpatialConfig,
    DatasetTimeSeriesConfig,
    DatasetType,
    DatasetWorkingMemoryConfig,
    # Top-level config
    LlamaFarmConfig,
    ProcessingConfig,
    StreamingConfig,
    VectorConfig,
)
from datamodel import (
    Graph1 as DatasetGraph,
)
from datamodel import (
    Processing as DatasetProcessing,
)
from datamodel import (
    Spatial as DatasetSpatial,
)
from datamodel import (
    Streaming as DatasetStreaming,
)
from datamodel import (
    Timeseries as DatasetTimeseries,
)
from datamodel import (
    Vector as DatasetVector,
)
from datamodel import (
    WorkingMemory as DatasetWorkingMemory,
)


class TestDatasetType:
    """Test DatasetType enum."""

    def test_knowledge_type(self):
        """Test knowledge dataset type."""
        assert DatasetType.knowledge.value == "knowledge"

    def test_realtime_type(self):
        """Test realtime dataset type."""
        assert DatasetType.realtime.value == "realtime"

    def test_graph_type(self):
        """Test graph-only dataset type."""
        assert DatasetType.graph.value == "graph"

    def test_timeseries_type(self):
        """Test timeseries dataset type."""
        assert DatasetType.timeseries.value == "timeseries"

    def test_spatial_type(self):
        """Test spatial dataset type (top-level, separate from timeseries)."""
        assert DatasetType.spatial.value == "spatial"

    def test_hybrid_type(self):
        """Test hybrid dataset type (all capabilities)."""
        assert DatasetType.hybrid.value == "hybrid"


class TestVectorConfig:
    """Test VectorConfig model."""

    def test_vector_config_defaults(self):
        """Test VectorConfig with default values."""
        config = VectorConfig()
        assert config.enabled is True
        assert config.store == "ChromaStore"
        assert config.collection is None
        assert config.embedding_strategy is None
        assert config.retrieval_strategy is None

    def test_vector_config_custom(self):
        """Test VectorConfig with custom values."""
        config = VectorConfig(
            enabled=True,
            store="QdrantStore",
            collection="my_docs",
            embedding_strategy="semantic",
            retrieval_strategy="hybrid_rerank",
        )
        assert config.store == "QdrantStore"
        assert config.collection == "my_docs"
        assert config.embedding_strategy == "semantic"
        assert config.retrieval_strategy == "hybrid_rerank"


class TestDatasetGraphConfig:
    """Test DatasetGraphConfig model."""

    def test_graph_config_defaults(self):
        """Test DatasetGraphConfig with default values."""
        config = DatasetGraphConfig()
        assert config.enabled is True
        assert config.entity_extraction is True
        assert config.relationship_extraction is False
        assert config.max_path_depth == 10

    def test_graph_config_with_llm_extraction(self):
        """Test DatasetGraphConfig with LLM relationship extraction."""
        config = DatasetGraphConfig(
            entity_extraction=True,
            relationship_extraction=True,
            max_path_depth=15,
        )
        assert config.relationship_extraction is True
        assert config.max_path_depth == 15


class TestDatasetTimeSeriesConfig:
    """Test DatasetTimeSeriesConfig model."""

    def test_timeseries_config_defaults(self):
        """Test DatasetTimeSeriesConfig with default values."""
        config = DatasetTimeSeriesConfig()
        assert config.enabled is True
        assert config.retention_days == 30

    def test_timeseries_config_custom(self):
        """Test DatasetTimeSeriesConfig with custom retention."""
        config = DatasetTimeSeriesConfig(retention_days=90)
        assert config.retention_days == 90


class TestDatasetSpatialConfig:
    """Test DatasetSpatialConfig model - top-level geo-spatial configuration."""

    def test_spatial_config_defaults(self):
        """Test DatasetSpatialConfig with default values."""
        config = DatasetSpatialConfig()
        assert config.enabled is True
        assert config.retention_days == 30
        assert config.index_type == "rtree"

    def test_spatial_config_custom(self):
        """Test DatasetSpatialConfig with custom values."""
        config = DatasetSpatialConfig(
            enabled=True,
            retention_days=60,
            index_type="geohash",
        )
        assert config.retention_days == 60
        assert config.index_type == "geohash"


class TestDatasetWorkingMemoryConfig:
    """Test DatasetWorkingMemoryConfig model."""

    def test_working_memory_config_defaults(self):
        """Test DatasetWorkingMemoryConfig with default values."""
        config = DatasetWorkingMemoryConfig()
        assert config.enabled is True
        assert config.ttl_seconds == 3600
        assert config.max_records == 10000

    def test_working_memory_config_custom(self):
        """Test DatasetWorkingMemoryConfig with custom values."""
        config = DatasetWorkingMemoryConfig(
            ttl_seconds=1800,
            max_records=50000,
        )
        assert config.ttl_seconds == 1800
        assert config.max_records == 50000


class TestStreamingConfig:
    """Test StreamingConfig model for realtime datasets."""

    def test_streaming_config_defaults(self):
        """Test StreamingConfig with default values."""
        config = StreamingConfig()
        assert config.enabled is False
        assert config.batch_size == 100
        assert config.flush_interval_ms == 1000

    def test_streaming_config_enabled(self):
        """Test StreamingConfig with streaming enabled."""
        config = StreamingConfig(
            enabled=True,
            batch_size=500,
            flush_interval_ms=500,
        )
        assert config.enabled is True
        assert config.batch_size == 500
        assert config.flush_interval_ms == 500


class TestDatasetConsolidationConfig:
    """Test DatasetConsolidationConfig model."""

    def test_consolidation_config_defaults(self):
        """Test DatasetConsolidationConfig with default values."""
        config = DatasetConsolidationConfig()
        assert config.enabled is True
        assert config.interval_seconds == 300
        assert config.min_records == 10
        assert config.extract_summaries is False
        assert config.prune_after is True

    def test_consolidation_config_with_summaries(self):
        """Test DatasetConsolidationConfig with summary extraction."""
        config = DatasetConsolidationConfig(
            interval_seconds=60,
            min_records=5,
            extract_summaries=True,
        )
        assert config.interval_seconds == 60
        assert config.extract_summaries is True


class TestProcessingConfig:
    """Test ProcessingConfig model."""

    def test_processing_config_required_strategy(self):
        """Test ProcessingConfig requires strategy."""
        config = ProcessingConfig(strategy="pdf_strategy")
        assert config.strategy == "pdf_strategy"
        assert config.chunking_size is None
        assert config.chunking_overlap is None

    def test_processing_config_with_chunking(self):
        """Test ProcessingConfig with chunking overrides."""
        config = ProcessingConfig(
            strategy="text_strategy",
            chunking_size=1000,
            chunking_overlap=200,
        )
        assert config.chunking_size == 1000
        assert config.chunking_overlap == 200


class TestDatasetModel:
    """Test the unified Dataset model."""

    def test_dataset_minimal(self):
        """Test Dataset with only required name."""
        dataset = Dataset(name="test_dataset")
        assert dataset.name == "test_dataset"
        # Dataset uses inline Type3 enum, check value matches
        assert dataset.type.value == "knowledge"

    def test_dataset_with_type(self):
        """Test Dataset with explicit type."""
        dataset = Dataset(name="realtime_data", type="realtime")
        assert dataset.type.value == "realtime"

    def test_dataset_knowledge_type_full_config(self):
        """Test knowledge dataset with vector and graph configs."""
        dataset = Dataset(
            name="military_intel",
            type="knowledge",
            description="Military protocols and entity knowledge",
            vector=DatasetVector(
                store="ChromaStore",
                collection="military_docs",
            ),
            graph=DatasetGraph(
                entity_extraction=True,
                relationship_extraction=True,
            ),
            processing=DatasetProcessing(strategy="pdf_strategy"),
        )
        assert dataset.name == "military_intel"
        assert dataset.vector.collection == "military_docs"
        assert dataset.graph.entity_extraction is True
        assert dataset.processing.strategy == "pdf_strategy"

    def test_dataset_realtime_type_full_config(self):
        """Test realtime dataset with all stores and streaming."""
        dataset = Dataset(
            name="soldier_telemetry",
            type="realtime",
            description="Real-time biometrics and location",
            vector=DatasetVector(collection="telemetry_summaries"),
            graph=DatasetGraph(entity_extraction=False),
            timeseries=DatasetTimeseries(retention_days=30),
            spatial=DatasetSpatial(index_type="rtree"),
            working_memory=DatasetWorkingMemory(
                ttl_seconds=3600,
                max_records=50000,
            ),
            streaming=DatasetStreaming(
                enabled=True,
                batch_size=100,
                flush_interval_ms=1000,
            ),
            consolidation=DatasetConsolidation(
                interval_seconds=60,
                extract_summaries=True,
            ),
        )
        assert dataset.type.value == "realtime"
        assert dataset.streaming.enabled is True
        assert dataset.spatial.index_type == "rtree"
        assert dataset.working_memory.max_records == 50000

    def test_dataset_spatial_only(self):
        """Test spatial-only dataset (geo-tracking)."""
        dataset = Dataset(
            name="fleet_tracking",
            type="spatial",
            spatial=DatasetSpatial(
                retention_days=90,
                index_type="geohash",
            ),
            working_memory=DatasetWorkingMemory(ttl_seconds=7200),
        )
        assert dataset.type.value == "spatial"
        assert dataset.spatial.index_type == "geohash"

    def test_dataset_backward_compatibility(self):
        """Test backward compatible deprecated fields."""
        dataset = Dataset(
            name="legacy_dataset",
            data_processing_strategy="old_strategy",
            database="old_db",
            memory="old_memory",
        )
        assert dataset.data_processing_strategy == "old_strategy"
        assert dataset.database == "old_db"
        assert dataset.memory == "old_memory"


class TestLlamaFarmConfigWithDatasets:
    """Test LlamaFarmConfig with new dataset configurations."""

    def test_config_with_knowledge_dataset(self):
        """Test LlamaFarmConfig with knowledge dataset."""
        config_dict = {
            "version": "v1",
            "name": "test_project",
            "namespace": "test_namespace",
            "runtime": {
                "models": [
                    {"name": "test_model", "provider": "openai", "model": "gpt-4"}
                ]
            },
            "datasets": [
                {
                    "name": "docs",
                    "type": "knowledge",
                    "vector": {"store": "ChromaStore"},
                    "graph": {"entity_extraction": True},
                }
            ],
        }
        config = LlamaFarmConfig(**config_dict)
        assert len(config.datasets) == 1
        assert config.datasets[0].name == "docs"

    def test_config_with_realtime_dataset(self):
        """Test LlamaFarmConfig with realtime streaming dataset."""
        config_dict = {
            "version": "v1",
            "name": "test_project",
            "namespace": "test_namespace",
            "runtime": {
                "models": [
                    {"name": "test_model", "provider": "openai", "model": "gpt-4"}
                ]
            },
            "datasets": [
                {
                    "name": "telemetry",
                    "type": "realtime",
                    "streaming": {"enabled": True, "batch_size": 200},
                    "timeseries": {"retention_days": 14},
                    "spatial": {"index_type": "rtree"},
                    "working_memory": {"ttl_seconds": 1800},
                }
            ],
        }
        config = LlamaFarmConfig(**config_dict)
        dataset = config.datasets[0]
        assert dataset.type.value == "realtime"


class TestDatasetYAMLLoading:
    """Test loading dataset configurations from YAML files."""

    def test_load_yaml_with_unified_dataset(self):
        """Test loading a YAML file with unified dataset configuration."""
        yaml_content = """
version: v1
name: test_project
namespace: test_namespace

datasets:
  - name: military_intel
    type: knowledge
    description: Military protocols and entity knowledge
    vector:
      enabled: true
      store: ChromaStore
      collection: military_docs
    graph:
      enabled: true
      entity_extraction: true
      relationship_extraction: false
      max_path_depth: 10
    processing:
      strategy: pdf_strategy
    consolidation:
      enabled: true
      interval_seconds: 300

  - name: soldier_telemetry
    type: realtime
    description: Real-time biometrics and location
    timeseries:
      retention_days: 30
    spatial:
      enabled: true
      index_type: rtree
    working_memory:
      ttl_seconds: 3600
      max_records: 50000
    streaming:
      enabled: true
      batch_size: 100

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
    - name: pdf_strategy
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

            # Verify datasets were loaded
            assert "datasets" in config
            assert len(config["datasets"]) == 2

            # Check knowledge dataset
            knowledge_ds = config["datasets"][0]
            assert knowledge_ds["name"] == "military_intel"
            assert knowledge_ds["type"] == "knowledge"
            assert knowledge_ds["vector"]["store"] == "ChromaStore"
            assert knowledge_ds["graph"]["entity_extraction"] is True

            # Check realtime dataset
            realtime_ds = config["datasets"][1]
            assert realtime_ds["name"] == "soldier_telemetry"
            assert realtime_ds["type"] == "realtime"
            assert realtime_ds["spatial"]["index_type"] == "rtree"
            assert realtime_ds["streaming"]["enabled"] is True

        finally:
            import os

            os.unlink(temp_path)

    def test_load_yaml_backward_compatible(self):
        """Test loading YAML with deprecated dataset fields still works."""
        yaml_content = """
version: v1
name: test_project
namespace: test_namespace

datasets:
  - name: legacy_dataset
    data_processing_strategy: default_strategy
    database: main_db
    memory: brain_memory

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

            # Verify deprecated fields still load
            dataset = config["datasets"][0]
            assert dataset["name"] == "legacy_dataset"
            assert dataset["data_processing_strategy"] == "default_strategy"
            assert dataset["database"] == "main_db"
            assert dataset["memory"] == "brain_memory"

        finally:
            import os

            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
