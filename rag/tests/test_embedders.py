"""Tests for embedding generators."""

import pytest
from unittest.mock import patch, MagicMock

from components.embedders.ollama_embedder import OllamaEmbedder


class TestOllamaEmbedder:
    """Test the Ollama embedder."""

    def test_embedder_initialization(self):
        """Test embedder initialization with config."""
        config = {
            "model": "custom-model",
            "base_url": "http://custom:11434",
            "batch_size": 16,
            "timeout": 30,
        }

        embedder = OllamaEmbedder(config=config)

        assert embedder.model == "custom-model"
        assert embedder.base_url == "http://custom:11434"
        assert embedder.batch_size == 16
        assert embedder.timeout == 30

    def test_embedder_default_config(self):
        """Test embedder with default configuration."""
        embedder = OllamaEmbedder()

        assert embedder.model == "nomic-embed-text"
        assert embedder.base_url == "http://localhost:11434"
        assert embedder.batch_size == 32
        assert embedder.timeout == 60

    @pytest.mark.ollama
    def test_validation_with_ollama(self):
        """Test config validation with actual Ollama (requires Ollama running)."""
        embedder = OllamaEmbedder()

        try:
            result = embedder.validate_config()
            assert result is True
        except ValueError as e:
            pytest.skip(f"Ollama not available: {e}")

    def test_validation_mock_success(self, mock_ollama_available):
        """Test config validation with mocked Ollama success."""
        embedder = OllamaEmbedder()
        result = embedder.validate_config()
        assert result is True

    def test_validation_mock_failure(self):
        """Test config validation with mocked Ollama failure."""
        with patch("requests.get") as mock_get:
            mock_get.return_value.status_code = 500

            embedder = OllamaEmbedder()

            # Should return False instead of raising an exception
            result = embedder.validate_config()
            assert result is False

    def test_embed_mock(self, mock_ollama_available):
        """Test embedding generation with mocked Ollama."""
        embedder = OllamaEmbedder()
        texts = ["Hello world", "This is a test"]

        embeddings = embedder.embed(texts)

        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) == 768 for emb in embeddings)  # 768 dimensions

    @pytest.mark.ollama
    def test_embed_real(self):
        """Test embedding generation with real Ollama (requires Ollama running)."""
        embedder = OllamaEmbedder()

        try:
            embedder.validate_config()
        except ValueError:
            pytest.skip("Ollama not available")

        texts = ["Hello world", "This is a test"]
        embeddings = embedder.embed(texts)

        assert len(embeddings) == 2
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)

    def test_embed_batch_processing(self, mock_ollama_available):
        """Test batch processing of embeddings."""
        config = {"batch_size": 2}
        embedder = OllamaEmbedder(config=config)

        texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
        embeddings = embedder.embed(texts)

        assert len(embeddings) == 5
        assert all(isinstance(emb, list) for emb in embeddings)
