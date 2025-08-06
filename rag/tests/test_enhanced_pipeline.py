"""Tests for enhanced_pipeline module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import sys

from core.enhanced_pipeline import EnhancedPipeline
from core.base import Document, ProcessingResult, Parser, Embedder, VectorStore


class TestEnhancedPipeline:
    """Test enhanced pipeline with progress tracking."""
    
    @pytest.fixture
    def mock_parser(self):
        """Create mock parser."""
        parser = Mock(spec=Parser)
        parser.name = "MockParser"
        parser.parse = Mock(return_value=ProcessingResult(
            documents=[
                Document(content="Test doc 1", metadata={}),
                Document(content="Test doc 2", metadata={})
            ],
            errors=[]
        ))
        return parser
    
    @pytest.fixture
    def mock_embedder(self):
        """Create mock embedder."""
        embedder = Mock(spec=Embedder)
        embedder.name = "MockEmbedder"
        embedder.batch_size = 2
        embedder.embed = Mock(return_value=[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])
        embedder.process = Mock(return_value=ProcessingResult(
            documents=[
                Document(content="Test doc 1", metadata={}, embeddings=[0.1, 0.2, 0.3]),
                Document(content="Test doc 2", metadata={}, embeddings=[0.4, 0.5, 0.6])
            ],
            errors=[]
        ))
        return embedder
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store."""
        store = Mock(spec=VectorStore)
        store.name = "MockVectorStore"
        store.add_documents = Mock(return_value=True)
        store.process = Mock(return_value=ProcessingResult(
            documents=[
                Document(content="Test doc 1", metadata={}),
                Document(content="Test doc 2", metadata={})
            ],
            errors=[],
            metrics={"stored_count": 2}
        ))
        return store
    
    @pytest.fixture
    def pipeline(self):
        """Create enhanced pipeline instance."""
        return EnhancedPipeline("Test Pipeline")
    
    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.name == "Test Pipeline"
        assert pipeline.tracker is not None
        assert len(pipeline.components) == 0
    
    def test_run_with_progress_from_source(self, pipeline, mock_parser, mock_embedder):
        """Test running pipeline with progress tracking from source."""
        # Add components
        pipeline.add_component(mock_parser)
        pipeline.add_component(mock_embedder)
        
        # Run pipeline
        result = pipeline.run_with_progress(source="test.txt")
        
        # Verify parsing was called
        mock_parser.parse.assert_called_once_with("test.txt")
        
        # Enhanced pipeline calls embed() directly for embedders, not process()
        assert mock_embedder.embed.called
        
        # Verify result
        assert len(result.documents) == 2
        assert len(result.errors) == 0
    
    def test_run_with_progress_from_documents(self, pipeline, mock_embedder):
        """Test running pipeline with documents input."""
        # Add component
        pipeline.add_component(mock_embedder)
        
        # Input documents
        docs = [
            Document(content="Test doc 1", metadata={}),
            Document(content="Test doc 2", metadata={})
        ]
        
        # Run pipeline
        result = pipeline.run_with_progress(documents=docs)
        
        # Enhanced pipeline calls embed() directly for embedders, not process()
        assert mock_embedder.embed.called
        
        # Verify result
        assert len(result.documents) == 2
        assert len(result.errors) == 0
    
    def test_run_with_no_input(self, pipeline):
        """Test error handling for no input."""
        with pytest.raises(ValueError, match="Either source or documents must be provided"):
            pipeline.run_with_progress()
    
    def test_run_with_no_parser_for_source(self, pipeline, mock_embedder):
        """Test error when no parser for source input."""
        pipeline.add_component(mock_embedder)
        
        with pytest.raises(ValueError, match="Pipeline must start with a Parser"):
            pipeline.run_with_progress(source="test.txt")
    
    def test_component_error_handling(self, pipeline, mock_parser):
        """Test error handling during component processing."""
        # Create failing component (not embedder or vector store)
        failing_component = Mock()
        failing_component.name = "FailingComponent"
        failing_component.process = Mock(side_effect=Exception("Test error"))
        # Make sure it's not detected as embedder or vector store by deleting these attributes
        if hasattr(failing_component, 'embed'):
            delattr(failing_component, 'embed')
        if hasattr(failing_component, 'add_documents'):
            delattr(failing_component, 'add_documents')
        
        # Add components
        pipeline.add_component(mock_parser)
        pipeline.add_component(failing_component)
        
        # Run pipeline
        result = pipeline.run_with_progress(source="test.txt")
        
        # Should continue and collect errors
        assert len(result.errors) == 1
        assert result.errors[0]["component"] == "FailingComponent"
        assert "Test error" in result.errors[0]["error"]
    
    def test_process_embeddings_with_progress(self, pipeline, mock_embedder):
        """Test embedding processing with progress tracking."""
        # Test documents
        docs = [Document(content=f"Doc {i}", metadata={}) for i in range(2)]
        
        # Process embeddings
        pipeline._process_embeddings_with_progress(mock_embedder, docs)
        
        # Verify embedder was called
        mock_embedder.embed.assert_called()
        
        # Verify embeddings were added to documents
        for doc in docs:
            assert doc.embeddings is not None
    
    def test_process_storage_with_progress(self, pipeline, mock_vector_store):
        """Test vector storage with progress tracking."""
        # Test documents
        docs = [Document(content=f"Doc {i}", metadata={}) for i in range(2)]
        
        # Process storage
        pipeline._process_storage_with_progress(mock_vector_store, docs)
        
        # Verify store was called
        mock_vector_store.add_documents.assert_called_once_with(docs)
    
    def test_embeddings_error_recovery(self, pipeline):
        """Test error recovery during embedding generation."""
        # Create failing embedder
        failing_embedder = Mock()
        failing_embedder.batch_size = 2
        failing_embedder.embed = Mock(side_effect=Exception("Embedding failed"))
        
        # Test documents
        docs = [Document(content="Test", metadata={})]
        
        # Should raise exception
        with pytest.raises(Exception, match="Embedding failed"):
            pipeline._process_embeddings_with_progress(failing_embedder, docs)
        
        # Verify embed was called
        assert failing_embedder.embed.called
    
    def test_empty_document_handling(self, pipeline, mock_parser):
        """Test handling of empty document list."""
        # Mock parser returns empty list
        mock_parser.parse = Mock(return_value=ProcessingResult(
            documents=[],
            errors=[]
        ))
        
        pipeline.add_component(mock_parser)
        
        # Run pipeline
        result = pipeline.run_with_progress(source="empty.txt")
        
        # Should handle gracefully
        assert len(result.documents) == 0
        assert len(result.errors) == 0
    
    def test_tracker_messages(self, pipeline):
        """Test that tracker methods work without errors."""
        # Access tracker methods to ensure they work
        pipeline.tracker.print_llama_art()
        pipeline.tracker.print_header("Test Header")
        pipeline.tracker.print_info("Test info")
        pipeline.tracker.print_success("Test success") 
        pipeline.tracker.print_warning("Test warning")
        pipeline.tracker.print_error("Test error")
        
        # Test random messages
        pun = pipeline.tracker.get_random_pun()
        assert isinstance(pun, str)
        assert len(pun) > 0
        
        motivation = pipeline.tracker.get_random_motivation()
        assert isinstance(motivation, str)
        assert len(motivation) > 0
        
        completion = pipeline.tracker.get_completion_message()
        assert isinstance(completion, str)
        assert len(completion) > 0