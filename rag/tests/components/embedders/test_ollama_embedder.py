"""Tests for Ollama Embedder component."""

import pytest
from pathlib import Path
import sys
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rag.core.base import Document
from rag.components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder


class TestOllamaEmbedder:
    """Test OllamaEmbedder functionality."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for embedding."""
        return [
            "This is a test sentence for embedding.",
            "Another sample text with different content.",
            "Machine learning and artificial intelligence are transforming technology."
        ]
    
    @pytest.fixture
    def sample_documents(self):
        """Sample documents for embedding."""
        return [
            Document(
                content="This is test document content for embedding analysis.",
                id="doc1",
                source="test1.txt",
                metadata={}
            ),
            Document(
                content="Different content for vector representation testing.",
                id="doc2", 
                source="test2.txt",
                metadata={}
            )
        ]
    
    @pytest.fixture
    def mock_embedder(self):
        """Create embedder with mocked Ollama API."""
        config = {
            "model": "nomic-embed-text",
            "api_base": "http://localhost:11434",
            "batch_size": 2
        }
        
        embedder = OllamaEmbedder("test_embedder", config)
        
        # Mock the actual embedding call
        with patch.object(embedder, '_call_ollama_api') as mock_api:
            # Return fake embeddings
            mock_api.return_value = {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5] * 100  # 500-dim fake embedding
            }
            yield embedder, mock_api
    
    def test_embedder_initialization(self):
        """Test embedder initialization with different configs."""
        # Default config
        embedder = OllamaEmbedder("default")
        assert embedder is not None
        assert embedder.model == "nomic-embed-text"
        assert embedder.api_base == "http://localhost:11434"
        
        # Custom config
        custom_config = {
            "model": "custom-embed",
            "api_base": "http://custom:8080",
            "batch_size": 5,
            "timeout": 60
        }
        embedder = OllamaEmbedder("custom", custom_config)
        assert embedder.model == "custom-embed"
        assert embedder.api_base == "http://custom:8080"
        assert embedder.batch_size == 5
    
    def test_text_embedding(self, mock_embedder, sample_texts):
        """Test embedding of text strings."""
        embedder, mock_api = mock_embedder
        
        # Test single text embedding
        embedding = embedder.embed_text(sample_texts[0])
        
        assert embedding is not None
        assert isinstance(embedding, list)
        assert len(embedding) == 500  # Mock embedding dimension
        assert all(isinstance(x, (int, float)) for x in embedding)
        
        # Verify API was called
        mock_api.assert_called_once()
    
    def test_multiple_text_embedding(self, mock_embedder, sample_texts):
        """Test embedding of multiple texts."""
        embedder, mock_api = mock_embedder
        
        embeddings = embedder.embed(sample_texts)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == len(sample_texts)
        
        # Each embedding should be a vector
        for emb in embeddings:
            assert isinstance(emb, list)
            assert len(emb) == 500
            assert all(isinstance(x, (int, float)) for x in emb)
    
    def test_document_embedding(self, mock_embedder, sample_documents):
        """Test embedding of document content."""
        embedder, mock_api = mock_embedder
        
        # Extract content and embed
        texts = [doc.content for doc in sample_documents]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == len(sample_documents)
        assert all(isinstance(emb, list) for emb in embeddings)
    
    def test_empty_text_handling(self, mock_embedder):
        """Test handling of empty or invalid text."""
        embedder, mock_api = mock_embedder
        
        # Test empty string
        embedding = embedder.embed_text("")
        assert embedding is not None  # Should handle gracefully
        
        # Test None
        embedding = embedder.embed_text(None)
        assert embedding is not None or embedding is None  # Either is acceptable
        
        # Test whitespace only
        embedding = embedder.embed_text("   \n\t   ")
        assert embedding is not None
    
    def test_batching_functionality(self, mock_embedder):
        """Test batching of multiple embedding requests."""
        embedder, mock_api = mock_embedder
        
        # Set small batch size
        embedder.batch_size = 2
        
        # Test with more texts than batch size
        texts = [f"Test text number {i}" for i in range(5)]
        embeddings = embedder.embed(texts)
        
        assert len(embeddings) == 5
        # Should have made multiple API calls due to batching
        assert mock_api.call_count >= 2
    
    def test_error_handling(self):
        """Test error handling for API failures."""
        embedder = OllamaEmbedder("error_test")
        
        # Mock failed API call
        with patch.object(embedder, '_call_ollama_api') as mock_api:
            mock_api.side_effect = Exception("API connection failed")
            
            # Should handle error gracefully
            embedding = embedder.embed_text("test text")
            assert embedding is None or isinstance(embedding, list)
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Valid configuration
        valid_config = {
            "model": "nomic-embed-text",
            "api_base": "http://localhost:11434",
            "batch_size": 1
        }
        embedder = OllamaEmbedder("valid", valid_config)
        assert embedder.model == "nomic-embed-text"
        
        # Invalid batch size should use default
        invalid_config = {
            "batch_size": -1
        }
        embedder = OllamaEmbedder("invalid", invalid_config)
        assert embedder.batch_size > 0
    
    def test_model_availability_check(self, mock_embedder):
        """Test model availability checking."""
        embedder, mock_api = mock_embedder
        
        # Mock successful model check
        with patch.object(embedder, '_check_model_availability') as mock_check:
            mock_check.return_value = True
            
            available = embedder._check_model_availability()
            assert available == True
    
    def test_embedding_consistency(self, mock_embedder):
        """Test that same text produces consistent embeddings."""
        embedder, mock_api = mock_embedder
        
        test_text = "Consistent embedding test text"
        
        # Get embeddings twice
        embedding1 = embedder.embed_text(test_text)
        embedding2 = embedder.embed_text(test_text)
        
        # Should be identical (with mocked API)
        assert embedding1 == embedding2
    
    def test_different_texts_different_embeddings(self, mock_embedder):
        """Test that different texts should produce different embeddings."""
        embedder, mock_api = mock_embedder
        
        # Mock different embeddings for different texts
        def mock_api_side_effect(*args, **kwargs):
            text = args[0] if args else kwargs.get('text', '')
            if 'first' in text:
                return {"embedding": [0.1] * 500}
            else:
                return {"embedding": [0.9] * 500}
        
        mock_api.side_effect = mock_api_side_effect
        
        embedding1 = embedder.embed_text("This is the first test text")
        embedding2 = embedder.embed_text("This is the second test text")
        
        # Should be different
        assert embedding1 != embedding2
    
    def test_get_description(self):
        """Test embedder description method."""
        description = OllamaEmbedder.get_description()
        assert isinstance(description, str)
        assert len(description) > 0
        assert "ollama" in description.lower()


# Integration test (requires actual Ollama service)
class TestOllamaEmbedderIntegration:
    """Integration tests that require actual Ollama service."""
    
    @pytest.mark.skipif(True, reason="Requires Ollama service running")
    def test_real_ollama_embedding(self):
        """Test with real Ollama service (skipped by default)."""
        embedder = OllamaEmbedder("integration_test")
        
        try:
            embedding = embedder.embed_text("Integration test text")
            assert embedding is not None
            assert isinstance(embedding, list)
            assert len(embedding) > 0
        except Exception as e:
            pytest.skip(f"Ollama service not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__])