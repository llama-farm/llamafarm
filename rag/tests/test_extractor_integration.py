"""Tests for extractor_integration module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from core.extractor_integration import (
    ExtractorIntegrator,
    create_extractor_integrator,
    enhance_processing_result,
    apply_extractors_from_cli_args
)
from core.base import Document, ProcessingResult
from components.extractors.base import BaseExtractor, ExtractorPipeline, ExtractorRegistry


class MockExtractor(BaseExtractor):
    """Mock extractor for testing."""
    
    def __init__(self, name="MockExtractor", config=None):
        super().__init__(name=name, config=config or {})
    
    def extract(self, documents):
        """Mock extraction that adds test metadata."""
        for doc in documents:
            doc.metadata[f"{self.name}_extracted"] = True
            doc.metadata["test_data"] = "extracted"
        return documents
    
    def get_dependencies(self):
        return []


class TestExtractorIntegrator:
    """Test extractor integrator functionality."""
    
    @pytest.fixture
    def mock_registry(self):
        """Mock extractor registry."""
        registry = Mock()
        registry.create = Mock(side_effect=lambda name, config: MockExtractor(name, config))
        return registry
    
    @pytest.fixture
    def extractor_config(self):
        """Sample extractor configuration."""
        return [
            {"type": "MockExtractor1", "config": {"setting": "value1"}},
            {"type": "MockExtractor2", "config": {"setting": "value2"}}
        ]
    
    def test_initialization_empty(self):
        """Test initialization with no extractors."""
        integrator = ExtractorIntegrator()
        
        assert integrator.extractor_config == []
        assert integrator.pipeline is None
        assert not integrator.has_extractors()
    
    @patch('core.extractor_integration.create_pipeline_from_config')
    def test_initialization_with_config(self, mock_create_pipeline, extractor_config):
        """Test initialization with extractor configuration."""
        mock_pipeline = Mock()
        mock_pipeline.extractors = [Mock(), Mock()]
        mock_create_pipeline.return_value = mock_pipeline
        
        integrator = ExtractorIntegrator(extractor_config)
        
        assert integrator.extractor_config == extractor_config
        assert integrator.pipeline == mock_pipeline
        assert integrator.has_extractors()
        
        # Verify pipeline creation
        mock_create_pipeline.assert_called_once()
    
    @patch('core.extractor_integration.create_pipeline_from_config')
    def test_pipeline_creation_error(self, mock_create_pipeline, extractor_config):
        """Test error handling during pipeline creation."""
        mock_create_pipeline.side_effect = Exception("Pipeline creation failed")
        
        integrator = ExtractorIntegrator(extractor_config)
        
        # Should handle error gracefully
        assert integrator.pipeline is None
        assert not integrator.has_extractors()
    
    def test_process_documents_no_extractors(self):
        """Test document processing with no extractors."""
        integrator = ExtractorIntegrator()
        
        docs = [
            Document(content="Test 1", metadata={}),
            Document(content="Test 2", metadata={})
        ]
        
        result = integrator.process_documents(docs)
        
        # Should return original documents
        assert result == docs
        for doc in result:
            assert "extracted" not in doc.metadata
    
    @patch('core.extractor_integration.create_pipeline_from_config')
    def test_process_documents_with_extractors(self, mock_create_pipeline):
        """Test document processing with extractors."""
        # Setup mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.extractors = [MockExtractor("Extractor1"), MockExtractor("Extractor2")]
        mock_pipeline.run = Mock(side_effect=lambda docs: [
            Document(content=doc.content, metadata={**doc.metadata, "extracted": True})
            for doc in docs
        ])
        mock_create_pipeline.return_value = mock_pipeline
        
        integrator = ExtractorIntegrator([{"type": "MockExtractor", "config": {}}])
        
        docs = [
            Document(content="Test 1", metadata={"original": True}),
            Document(content="Test 2", metadata={"original": True})
        ]
        
        result = integrator.process_documents(docs)
        
        # Verify extraction was called
        mock_pipeline.run.assert_called_once_with(docs)
        
        # Verify metadata was enhanced
        for doc in result:
            assert doc.metadata["original"] is True
            assert doc.metadata["extracted"] is True
    
    @patch('core.extractor_integration.create_pipeline_from_config')
    def test_process_documents_with_file_path(self, mock_create_pipeline):
        """Test document processing with file path context."""
        mock_pipeline = Mock()
        mock_pipeline.extractors = [MockExtractor()]
        mock_pipeline.run = Mock(return_value=[])
        mock_create_pipeline.return_value = mock_pipeline
        
        integrator = ExtractorIntegrator([{"type": "MockExtractor", "config": {}}])
        
        docs = [Document(content="Test", metadata={})]
        file_path = Path("/test/document.txt")
        
        integrator.process_documents(docs, file_path)
        
        # Verify file path was added to metadata
        assert docs[0].metadata["source_file"] == str(file_path)
        assert docs[0].metadata["source_filename"] == "document.txt"
    
    @patch('core.extractor_integration.create_pipeline_from_config')
    def test_process_documents_error_handling(self, mock_create_pipeline):
        """Test error handling during document processing."""
        mock_pipeline = Mock()
        mock_pipeline.extractors = [Mock()]
        mock_pipeline.run = Mock(side_effect=Exception("Extraction failed"))
        mock_create_pipeline.return_value = mock_pipeline
        
        integrator = ExtractorIntegrator([{"type": "MockExtractor", "config": {}}])
        
        docs = [Document(content="Test", metadata={})]
        
        # Should return original documents on error
        result = integrator.process_documents(docs)
        assert result == docs
    
    @patch('core.extractor_integration.create_pipeline_from_config')
    def test_get_extractor_info(self, mock_create_pipeline):
        """Test getting extractor information."""
        mock_pipeline = Mock()
        mock_pipeline.extractors = [Mock(), Mock()]  # Mock extractors list for len() call
        mock_pipeline.get_pipeline_info = Mock(return_value={
            "extractors": ["Extractor1", "Extractor2"],
            "total_extractors": 2
        })
        mock_create_pipeline.return_value = mock_pipeline
        
        integrator = ExtractorIntegrator([{"type": "MockExtractor", "config": {}}])
        
        info = integrator.get_extractor_info()
        
        assert info["total_extractors"] == 2
        assert len(info["extractors"]) == 2
    
    def test_get_extractor_info_no_pipeline(self):
        """Test getting info with no pipeline."""
        integrator = ExtractorIntegrator()
        
        info = integrator.get_extractor_info()
        
        assert info["total_extractors"] == 0
        assert info["extractors"] == []


class TestFactoryFunctions:
    """Test factory and helper functions."""
    
    def test_create_extractor_integrator_with_config(self):
        """Test creating integrator from config."""
        config = {
            "extractors": [
                {"type": "TestExtractor", "config": {"param": "value"}}
            ]
        }
        
        integrator = create_extractor_integrator(config)
        
        assert integrator is not None
        assert integrator.extractor_config == config["extractors"]
    
    def test_create_extractor_integrator_no_extractors(self):
        """Test creating integrator with no extractors."""
        config = {}
        
        integrator = create_extractor_integrator(config)
        
        assert integrator is None
    
    def test_create_extractor_integrator_empty_extractors(self):
        """Test creating integrator with empty extractor list."""
        config = {"extractors": []}
        
        integrator = create_extractor_integrator(config)
        
        assert integrator is None
    
    def test_enhance_processing_result_no_integrator(self):
        """Test enhancing result with no integrator."""
        result = ProcessingResult(
            documents=[Document(content="Test", metadata={})],
            errors=[]
        )
        
        enhanced = enhance_processing_result(result, None)
        
        assert enhanced == result
        assert "extractors" not in enhanced.metrics
    
    @patch('core.extractor_integration.ExtractorIntegrator')
    def test_enhance_processing_result_with_integrator(self, mock_integrator_class):
        """Test enhancing result with integrator."""
        # Create mock integrator instance
        mock_integrator = Mock()
        mock_integrator.has_extractors.return_value = True
        mock_integrator.process_documents.return_value = [
            Document(content="Enhanced", metadata={"extracted": True})
        ]
        mock_integrator.get_extractor_info.return_value = {
            "extractors": ["Extractor1"],
            "total_extractors": 1
        }
        
        result = ProcessingResult(
            documents=[Document(content="Test", metadata={})],
            errors=[],
            metrics={}
        )
        
        enhanced = enhance_processing_result(result, mock_integrator)
        
        # Verify documents were enhanced
        assert enhanced.documents[0].content == "Enhanced"
        assert enhanced.documents[0].metadata["extracted"] is True
        
        # Verify metrics were added
        assert "extractors" in enhanced.metrics
        assert enhanced.metrics["extractors"]["applied"] is True
        assert enhanced.metrics["extractors"]["pipeline_info"]["total_extractors"] == 1
    
    def test_enhance_processing_result_error_handling(self):
        """Test error handling in enhance processing result."""
        mock_integrator = Mock()
        mock_integrator.has_extractors.return_value = True
        mock_integrator.process_documents.side_effect = Exception("Enhancement failed")
        
        result = ProcessingResult(
            documents=[Document(content="Test", metadata={})],
            errors=[]
        )
        
        # Should return original result on error
        enhanced = enhance_processing_result(result, mock_integrator)
        assert enhanced == result
    
    @patch('core.extractor_integration.registry')
    def test_apply_extractors_from_cli_args(self, mock_registry):
        """Test applying extractors from CLI arguments."""
        # Setup mock registry
        mock_registry.create.side_effect = lambda name, config: MockExtractor(name, config)
        
        docs = [
            Document(content="Test 1", metadata={}),
            Document(content="Test 2", metadata={})
        ]
        
        extractor_names = ["yake", "entities"]
        extractor_configs = {
            "yake": {"max_keywords": 10},
            "entities": {"entity_types": ["PERSON", "ORG"]}
        }
        
        result = apply_extractors_from_cli_args(docs, extractor_names, extractor_configs)
        
        # Verify extractors were created
        assert mock_registry.create.call_count == 2
        mock_registry.create.assert_any_call("yake", {"max_keywords": 10})
        mock_registry.create.assert_any_call("entities", {"entity_types": ["PERSON", "ORG"]})
        
        # Verify documents were processed
        assert len(result) == 2
    
    @patch('core.extractor_integration.registry')
    def test_apply_extractors_from_cli_args_no_names(self, mock_registry):
        """Test with no extractor names."""
        docs = [Document(content="Test", metadata={})]
        
        result = apply_extractors_from_cli_args(docs, [])
        
        # Should return original documents
        assert result == docs
        mock_registry.create.assert_not_called()
    
    @patch('core.extractor_integration.registry')
    @patch('core.extractor_integration.logger')
    def test_apply_extractors_from_cli_args_creation_failure(self, mock_logger, mock_registry):
        """Test handling of extractor creation failure."""
        # Some extractors fail to create
        mock_registry.create.side_effect = [None, MockExtractor("Working"), None]
        
        docs = [Document(content="Test", metadata={})]
        extractor_names = ["failing1", "working", "failing2"]
        
        result = apply_extractors_from_cli_args(docs, extractor_names)
        
        # Should still process with working extractors
        assert len(result) == 1
        
        # Should log warnings for failed extractors
        assert mock_logger.warning.call_count == 2
    
    @patch('core.extractor_integration.registry')
    @patch('core.extractor_integration.ExtractorPipeline')
    def test_apply_extractors_from_cli_args_pipeline_creation(self, mock_pipeline_class, mock_registry):
        """Test pipeline creation in CLI extractor application."""
        mock_extractor1 = MockExtractor("Extractor1")
        mock_extractor2 = MockExtractor("Extractor2")
        mock_registry.create.side_effect = [mock_extractor1, mock_extractor2]
        
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = []
        mock_pipeline_class.return_value = mock_pipeline
        
        docs = [Document(content="Test", metadata={})]
        extractor_names = ["extractor1", "extractor2"]
        
        apply_extractors_from_cli_args(docs, extractor_names)
        
        # Verify pipeline was created with extractors
        mock_pipeline_class.assert_called_once_with([mock_extractor1, mock_extractor2])
        mock_pipeline.run.assert_called_once_with(docs)