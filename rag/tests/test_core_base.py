"""Essential core functionality tests."""

import pytest
from rag.core.base import Document, ProcessingResult, Pipeline


class TestCoreBase:
    """Test core RAG base classes."""

    def test_document_creation(self):
        """Test Document creation and methods."""
        doc = Document(
            content="Test content",
            metadata={"key": "value"},
            id="doc-1",
            source="test.txt"
        )
        
        assert doc.content == "Test content"
        assert doc.metadata == {"key": "value"}
        assert doc.id == "doc-1"
        assert doc.source == "test.txt"
        
        # Test to_dict
        d = doc.to_dict()
        assert d["content"] == "Test content"
        assert d["metadata"] == {"key": "value"}

    def test_processing_result(self):
        """Test ProcessingResult container."""
        docs = [
            Document(content="Doc 1", id="1"),
            Document(content="Doc 2", id="2")
        ]
        result = ProcessingResult(documents=docs)
        
        assert len(result.documents) == 2
        assert result.documents[0].content == "Doc 1"
        assert result.errors == []

    def test_processing_result_with_errors(self):
        """Test ProcessingResult with errors."""
        result = ProcessingResult(
            documents=[],
            errors=["Error 1", "Error 2"]
        )
        
        assert len(result.documents) == 0
        assert len(result.errors) == 2
        assert "Error 1" in result.errors

    def test_pipeline_initialization(self):
        """Test Pipeline basic initialization."""
        pipeline = Pipeline()
        
        assert pipeline.components == []
        assert hasattr(pipeline, 'add_component')
        assert hasattr(pipeline, 'run')

