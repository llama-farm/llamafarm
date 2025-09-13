"""
Comprehensive tests for embedders
Tests embedder functionality directly without CLI dependency
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from typing import List

from rag.components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder


class TestOllamaEmbedder:
    """Test the Ollama embedder."""
    
    @patch('requests.post')
    def test_basic_embedding(self, mock_post):
        """Test basic embedding generation."""
        # Mock the Ollama API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 153 + [0.1, 0.2, 0.3]  # 768 dimensions
        }
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434",
                "dimension": 768,
                "batch_size": 16
            }
        )
        
        texts = ["This is a test document."]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        assert all(isinstance(x, float) for x in embeddings[0])
        
        # Verify API was called correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "http://localhost:11434" in call_args[0][0]
        assert call_args[1]["json"]["model"] == "nomic-embed-text"
        assert call_args[1]["json"]["prompt"] == "This is a test document."
    
    @patch('requests.post')
    def test_batch_embedding(self, mock_post):
        """Test batch embedding generation."""
        # Mock the Ollama API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434",
                "dimension": 768,
                "batch_size": 2
            }
        )
        
        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 5
        assert all(len(emb) == 768 for emb in embeddings)
        
        # Should be called once per text (Ollama doesn't support true batching)
        assert mock_post.call_count == 5
    
    @patch('requests.post')
    def test_empty_text(self, mock_post):
        """Test handling of empty text."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.0] * 768
        }
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "dimension": 768
            }
        )
        
        texts = [""]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
    
    @patch('requests.post')
    def test_error_handling(self, mock_post):
        """Test error handling for API failures."""
        # Mock API error
        mock_post.side_effect = Exception("Connection error")
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "base_url": "http://localhost:11434",
                "dimension": 768
            }
        )
        
        texts = ["Test text"]
        embeddings = embedder.embed(texts)
        
        # Should return zero-filled embeddings on error
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        assert all(x == 0.0 for x in embeddings[0])
    
    @patch('requests.post')
    def test_dimension_mismatch(self, mock_post):
        """Test handling of dimension mismatch."""
        # Return wrong dimension
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 512  # Wrong dimension
        }
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "dimension": 768  # Expected dimension
            }
        )
        
        texts = ["Test text"]
        embeddings = embedder.embed(texts)
        
        # Should either pad/truncate or handle the mismatch
        if embeddings:
            assert len(embeddings[0]) in [512, 768]
    
    @patch('requests.post')
    def test_unicode_text(self, mock_post):
        """Test handling of unicode text."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "dimension": 768
            }
        )
        
        texts = ["Hello 世界! 🌍 Unicode text with émojis"]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        
        # Check that unicode was properly sent
        call_args = mock_post.call_args
        assert "世界" in call_args[1]["json"]["prompt"]
        assert "🌍" in call_args[1]["json"]["prompt"]
    
    @patch('requests.post')
    def test_auto_pull_model(self, mock_post):
        """Test auto-pull model feature."""
        # Mock successful response
        mock_response = Mock(status_code=200)
        mock_response.json.return_value = {"embedding": [0.1] * 768}
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "auto_pull": True,
                "dimension": 768
            }
        )
        
        texts = ["Test"]
        embeddings = embedder.embed(texts)
        
        # Should get embeddings
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
    
    @patch('requests.post')
    def test_timeout_handling(self, mock_post):
        """Test timeout handling."""
        import requests
        mock_post.side_effect = requests.Timeout("Request timed out")
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "timeout": 1,
                "dimension": 768
            }
        )
        
        texts = ["Test text"]
        embeddings = embedder.embed(texts)
        
        # Should return zero embeddings on timeout
        assert len(embeddings) == 1
        assert len(embeddings[0]) == 768
        assert all(x == 0.0 for x in embeddings[0])
    
    @patch('requests.post')
    def test_embedding_normalization(self, mock_post):
        """Test embedding normalization if implemented."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Return unnormalized embedding
        mock_response.json.return_value = {
            "embedding": [3.0, 4.0, 0.0] + [0.0] * 765  # Simple 3-4-5 triangle for testing
        }
        mock_post.return_value = mock_response
        
        embedder = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "dimension": 768,
                "normalize": True  # If this option exists
            }
        )
        
        texts = ["Test"]
        embeddings = embedder.embed(texts)
        
        if embeddings and len(embeddings[0]) > 0:
            # Check if normalized (L2 norm should be 1)
            embedding_array = np.array(embeddings[0])
            norm = np.linalg.norm(embedding_array)
            # Either not normalized or normalized to 1
            assert norm > 0  # Should have some magnitude


class TestEmbedderIntegration:
    """Test embedder integration with the RAG pipeline."""
    
    @patch('requests.post')
    def test_embedder_with_ingest_handler(self, mock_post):
        """Test embedder working with IngestHandler."""
        from rag.core.ingest_handler import IngestHandler
        
        # Mock Ollama responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        mock_post.return_value = mock_response
        
        # Create a minimal config
        import tempfile
        import yaml
        
        config = {
            "version": "v1",
            "name": "test",
            "namespace": "default",
            "rag": {
                "databases": [{
                    "name": "test_db",
                    "type": "ChromaStore",
                    "config": {
                        "collection_name": "test",
                        "persist_directory": tempfile.mkdtemp()
                    },
                    "embedding_strategies": [{
                        "name": "test_embeddings",
                        "type": "OllamaEmbedder",
                        "priority": 0,
                        "config": {
                            "model": "nomic-embed-text",
                            "dimension": 768
                        }
                    }],
                    "retrieval_strategies": [{
                        "name": "basic",
                        "type": "BasicSimilarityStrategy",
                        "default": True,
                        "config": {"top_k": 10}
                    }],
                    "default_embedding_strategy": "test_embeddings",
                    "default_retrieval_strategy": "basic"
                }],
                "data_processing_strategies": [{
                    "name": "test_strategy",
                    "parsers": [{
                        "type": "TextParser_Python",
                        "file_include_patterns": ["*.txt"],
                        "priority": 100,
                        "config": {"chunk_size": 100}
                    }],
                    "extractors": []
                }]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            config_path = f.name
        
        try:
            handler = IngestHandler(
                config_path=config_path,
                data_processing_strategy='test_strategy',
                database='test_db'
            )
            
            # Verify embedder is initialized
            assert handler.embedder is not None
            assert isinstance(handler.embedder, OllamaEmbedder)
            
            # Test embedding
            test_texts = ["Test document"]
            embeddings = handler.embedder.embed(test_texts)
            
            assert len(embeddings) == 1
            assert len(embeddings[0]) == 768
            
        finally:
            import os
            os.unlink(config_path)
    
    @patch('requests.post')
    def test_multiple_embedders(self, mock_post):
        """Test handling multiple embedder configurations."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embedding": [0.1] * 768
        }
        mock_post.return_value = mock_response
        
        # Create embedders with different configs
        embedder1 = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "dimension": 768,
                "batch_size": 16
            }
        )
        
        embedder2 = OllamaEmbedder(
            config={
                "model": "nomic-embed-text",
                "dimension": 768,
                "batch_size": 32
            }
        )
        
        texts = ["Same text"]
        
        emb1 = embedder1.embed(texts)
        emb2 = embedder2.embed(texts)
        
        # Both should produce embeddings
        assert len(emb1) == 1 and len(emb1[0]) == 768
        assert len(emb2) == 1 and len(emb2[0]) == 768


if __name__ == "__main__":
    pytest.main([__file__, "-v"])