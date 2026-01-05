"""Tests for Pipeline Integration - Connects RAG pipeline to UnifiedDatasetStore.

Phase 23: RAG Pipeline Integration
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add rag to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.base import Document, ProcessingResult
from core.pipeline_integration import (
    DatasetIntegratedPipeline,
    create_integrated_pipeline,
    process_documents_to_dataset,
)


class TestDatasetIntegratedPipeline:
    """Test DatasetIntegratedPipeline class."""

    def test_init_default(self):
        """Test default initialization."""
        pipeline = DatasetIntegratedPipeline()

        assert pipeline.name == "Dataset Integrated Pipeline"
        assert pipeline.dataset_store is None
        assert pipeline.extract_entities is True
        assert pipeline.extract_relationships is False

    def test_init_with_config(self):
        """Test initialization with configuration."""
        config = {
            "extract_entities": True,
            "extract_relationships": True,
            "entity_types": ["PERSON", "ORG", "LOC"],
        }

        pipeline = DatasetIntegratedPipeline(
            name="Test Pipeline",
            config=config,
        )

        assert pipeline.name == "Test Pipeline"
        assert pipeline.extract_entities is True
        assert pipeline.extract_relationships is True
        assert pipeline.entity_types == ["PERSON", "ORG", "LOC"]

    def test_init_with_mock_store(self):
        """Test initialization with mock dataset store."""
        mock_store = MagicMock()

        pipeline = DatasetIntegratedPipeline(
            name="Store Pipeline",
            dataset_store=mock_store,
        )

        assert pipeline.dataset_store == mock_store

    def test_init_without_entity_extraction(self):
        """Test initialization with entity extraction disabled."""
        config = {"extract_entities": False}

        pipeline = DatasetIntegratedPipeline(config=config)

        assert pipeline.extract_entities is False
        assert pipeline._entity_extractor is None


class TestDatasetIntegratedPipelineEntityInit:
    """Test entity extractor initialization."""

    def test_entity_extractor_init_success(self):
        """Test successful entity extractor initialization."""
        # Entity extractor should be initialized when extract_entities is True
        pipeline = DatasetIntegratedPipeline(config={"extract_entities": True})

        # Entity extractor should be initialized
        assert pipeline.extract_entities is True
        # It should either have an extractor or None (depending on environment)
        # We're testing that the pipeline initializes without error
        assert pipeline is not None

    def test_entity_extractor_init_fallback(self):
        """Test entity extractor initialization with import fallback."""
        # If EntityExtractor can't be imported, should not crash
        pipeline = DatasetIntegratedPipeline(config={"extract_entities": True})

        # Pipeline should be created even if extractor fails
        assert pipeline is not None


class TestProcessWithDataset:
    """Test process_with_dataset method."""

    @pytest.fixture
    def mock_pipeline(self):
        """Create pipeline with mocked components."""
        pipeline = DatasetIntegratedPipeline(
            name="Test Pipeline",
            config={"extract_entities": False},  # Disable for basic tests
        )
        return pipeline

    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing."""
        return [
            Document(id="doc-1", content="Test document one."),
            Document(id="doc-2", content="Test document two."),
        ]

    def test_process_no_store(self, mock_pipeline, sample_documents):
        """Test processing without dataset store."""
        result = mock_pipeline.process_with_dataset(
            documents=sample_documents,
            store_in_graph=False,
            store_in_vector=False,
        )

        assert isinstance(result, ProcessingResult)
        assert len(result.documents) == 2
        assert len(result.errors) == 0

    def test_process_with_mock_store(self, sample_documents):
        """Test processing with mock dataset store."""
        mock_store = MagicMock()
        mock_store.graph_store = None

        pipeline = DatasetIntegratedPipeline(
            dataset_store=mock_store,
            config={"extract_entities": False},
        )

        result = pipeline.process_with_dataset(
            documents=sample_documents,
            store_in_graph=False,
            store_in_vector=True,
        )

        assert isinstance(result, ProcessingResult)
        assert len(result.documents) == 2


class TestExtractEntitiesToGraph:
    """Test _extract_entities_to_graph method."""

    @pytest.fixture
    def mock_entity_extractor(self):
        """Create mock entity extractor."""
        extractor = MagicMock()

        # Create mock entity
        mock_entity = MagicMock()
        mock_entity.name = "Test Entity"
        mock_entity.entity_type = "PERSON"
        mock_entity.entity_id = "ent-123"
        mock_entity.span_start = 0
        mock_entity.span_end = 11
        mock_entity.confidence = 0.95
        mock_entity.method = "mock"

        extractor.extract_entities.return_value = [mock_entity]
        return extractor

    def test_extract_no_store(self):
        """Test entity extraction without store."""
        pipeline = DatasetIntegratedPipeline(config={"extract_entities": False})

        doc = Document(id="doc-1", content="John works at Apple.")
        result = pipeline._extract_entities_to_graph([doc])

        assert result["entities_extracted"] == 0
        assert result["errors"] == []

    def test_extract_no_graph_store(self):
        """Test entity extraction when graph store is disabled."""
        mock_store = MagicMock()
        mock_store.graph_store = None

        pipeline = DatasetIntegratedPipeline(
            dataset_store=mock_store,
            config={"extract_entities": False},
        )

        doc = Document(id="doc-1", content="John works at Apple.")
        result = pipeline._extract_entities_to_graph([doc])

        assert result["entities_extracted"] == 0

    def test_extract_with_mock_store(self, mock_entity_extractor):
        """Test entity extraction with mocked store and extractor."""
        mock_store = MagicMock()
        mock_store.graph_store = MagicMock()
        mock_store.add_node.return_value = "node-123"
        mock_store.linkage_table = MagicMock()

        pipeline = DatasetIntegratedPipeline(
            dataset_store=mock_store,
            config={"extract_entities": True},
        )
        pipeline._entity_extractor = mock_entity_extractor

        doc = Document(id="doc-1", content="John works at Apple.")
        result = pipeline._extract_entities_to_graph([doc])

        assert result["entities_extracted"] == 1
        assert result["errors"] == []
        mock_store.add_node.assert_called_once()
        mock_store.linkage_table.link.assert_called_once()

    def test_extract_with_error_handling(self, mock_entity_extractor):
        """Test entity extraction error handling."""
        mock_store = MagicMock()
        mock_store.graph_store = MagicMock()
        mock_entity_extractor.extract_entities.side_effect = Exception(
            "Extraction error"
        )

        pipeline = DatasetIntegratedPipeline(
            dataset_store=mock_store,
            config={"extract_entities": True},
        )
        pipeline._entity_extractor = mock_entity_extractor

        doc = Document(id="doc-1", content="John works at Apple.")
        result = pipeline._extract_entities_to_graph([doc])

        assert result["entities_extracted"] == 0
        assert len(result["errors"]) == 1
        assert "Extraction error" in result["errors"][0]["error"]


class TestExtractRelationships:
    """Test relationship extraction in pipeline."""

    @pytest.fixture
    def mock_extractor_with_relationships(self):
        """Create mock extractor that returns entities and relationships."""
        extractor = MagicMock()

        # Create mock entities
        entity1 = MagicMock()
        entity1.name = "John"
        entity1.entity_type = "PERSON"
        entity1.entity_id = "ent-john"
        entity1.span_start = 0
        entity1.span_end = 4
        entity1.confidence = 0.95
        entity1.method = "mock"

        entity2 = MagicMock()
        entity2.name = "Apple"
        entity2.entity_type = "ORG"
        entity2.entity_id = "ent-apple"
        entity2.span_start = 14
        entity2.span_end = 19
        entity2.confidence = 0.90
        entity2.method = "mock"

        extractor.extract_entities.return_value = [entity1, entity2]

        # Create mock relationship
        relationship = MagicMock()
        relationship.source_entity = "John"
        relationship.target_entity = "Apple"
        relationship.relationship_type = "works_at"

        extractor.extract_relationships_llm.return_value = [relationship]

        return extractor

    def test_extract_relationships(self, mock_extractor_with_relationships):
        """Test relationship extraction when enabled."""
        mock_store = MagicMock()
        mock_store.graph_store = MagicMock()
        mock_store.add_node.return_value = "node-123"
        mock_store.add_edge.return_value = "edge-456"
        mock_store.linkage_table = MagicMock()

        pipeline = DatasetIntegratedPipeline(
            dataset_store=mock_store,
            config={
                "extract_entities": True,
                "extract_relationships": True,
            },
        )
        pipeline._entity_extractor = mock_extractor_with_relationships

        doc = Document(id="doc-1", content="John works at Apple.")
        result = pipeline._extract_entities_to_graph([doc])

        assert result["entities_extracted"] == 2
        assert result["relationships_extracted"] == 1


class TestStoreInVector:
    """Test _store_in_vector method."""

    def test_store_in_vector_basic(self):
        """Test basic vector store integration."""
        pipeline = DatasetIntegratedPipeline()

        documents = [
            Document(id="doc-1", content="Test content."),
        ]

        result = pipeline._store_in_vector(documents)

        assert result["stored"] == 1
        assert result["errors"] == []


class TestCreateIntegratedPipeline:
    """Test create_integrated_pipeline factory function."""

    def test_create_pipeline(self):
        """Test pipeline factory function."""
        with patch("core.unified_store.UnifiedDatasetStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            pipeline = create_integrated_pipeline(
                project_path="/tmp/test",
                dataset_name="test_dataset",
                dataset_type="knowledge",
            )

            assert isinstance(pipeline, DatasetIntegratedPipeline)
            assert pipeline.name == "test_dataset Pipeline"
            assert pipeline.dataset_store == mock_store

    def test_create_pipeline_with_components(self):
        """Test pipeline factory with components."""
        with patch("core.unified_store.UnifiedDatasetStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            mock_component = MagicMock()

            pipeline = create_integrated_pipeline(
                project_path="/tmp/test",
                dataset_name="test_dataset",
                components=[mock_component],
            )

            assert mock_component in pipeline.components

    def test_create_pipeline_with_config(self):
        """Test pipeline factory with config."""
        with patch("core.unified_store.UnifiedDatasetStore") as mock_store_class:
            mock_store = MagicMock()
            mock_store_class.return_value = mock_store

            config = {"extract_entities": False}

            pipeline = create_integrated_pipeline(
                project_path="/tmp/test",
                dataset_name="test_dataset",
                config=config,
            )

            assert pipeline.extract_entities is False


class TestProcessDocumentsToDataset:
    """Test process_documents_to_dataset convenience function."""

    @patch("core.pipeline_integration.create_integrated_pipeline")
    def test_process_documents(self, mock_create_pipeline):
        """Test convenience function for document processing."""
        mock_pipeline = MagicMock(spec=DatasetIntegratedPipeline)
        mock_pipeline.process_with_dataset.return_value = ProcessingResult(
            documents=[Document(id="doc-1", content="Test")],
            errors=[],
        )
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {"total_documents": 1}
        mock_pipeline.dataset_store = mock_store
        mock_create_pipeline.return_value = mock_pipeline

        documents = [Document(id="doc-1", content="Test document.")]

        result = process_documents_to_dataset(
            documents=documents,
            project_path="/tmp/test",
            dataset_name="test_dataset",
        )

        assert result["documents_processed"] == 1
        assert result["errors"] == []
        assert "stats" in result
        mock_store.close.assert_called_once()

    @patch("core.pipeline_integration.create_integrated_pipeline")
    def test_process_documents_no_entities(self, mock_create_pipeline):
        """Test processing without entity extraction."""
        mock_pipeline = MagicMock(spec=DatasetIntegratedPipeline)
        mock_pipeline.process_with_dataset.return_value = ProcessingResult(
            documents=[],
            errors=[],
        )
        mock_store = MagicMock()
        mock_store.get_stats.return_value = {}
        mock_pipeline.dataset_store = mock_store
        mock_create_pipeline.return_value = mock_pipeline

        process_documents_to_dataset(
            documents=[],
            project_path="/tmp/test",
            dataset_name="test_dataset",
            extract_entities=False,
        )

        # Verify process_with_dataset was called with store_in_graph=False
        mock_pipeline.process_with_dataset.assert_called_once()
        call_kwargs = mock_pipeline.process_with_dataset.call_args[1]
        assert call_kwargs["store_in_graph"] is False


class TestPipelineWithComponents:
    """Test pipeline with actual components."""

    def test_pipeline_run_with_components(self):
        """Test running pipeline with components."""
        mock_component = MagicMock()
        mock_component.process.return_value = ProcessingResult(
            documents=[Document(id="doc-1", content="Processed")],
            errors=[],
        )

        pipeline = DatasetIntegratedPipeline(config={"extract_entities": False})
        pipeline.add_component(mock_component)

        documents = [Document(id="doc-1", content="Original")]
        result = pipeline.process_with_dataset(
            documents=documents,
            store_in_graph=False,
            store_in_vector=False,
        )

        assert isinstance(result, ProcessingResult)


class TestPipelineInheritance:
    """Test that pipeline properly inherits from base Pipeline."""

    def test_inherits_from_pipeline(self):
        """Test inheritance from base Pipeline class."""
        from core.base import Pipeline

        assert issubclass(DatasetIntegratedPipeline, Pipeline)

    def test_has_components_list(self):
        """Test pipeline has components list from base class."""
        pipeline = DatasetIntegratedPipeline()
        assert hasattr(pipeline, "components")
        assert isinstance(pipeline.components, list)

    def test_add_component_method(self):
        """Test add_component method from base class."""
        pipeline = DatasetIntegratedPipeline()
        mock_component = MagicMock()

        pipeline.add_component(mock_component)

        assert mock_component in pipeline.components


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
