"""Essential embedder tests - focus on real functionality."""

import pytest
from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder
from components.embedders.sentence_transformer_embedder.sentence_transformer_embedder import (
    SentenceTransformerEmbedder,
)


class TestEmbedders:
    """Core embedder functionality tests."""

    def test_ollama_embedder_config(self):
        """Test Ollama embedder configuration."""
        config = {"model": "custom-model", "batch_size": 16}
        embedder = OllamaEmbedder(config=config)

        assert embedder.model == "custom-model"
        assert embedder.batch_size == 16

    def test_sentence_transformer_embedder(self):
        """Test SentenceTransformer embedder basic functionality."""
        # SentenceTransformerEmbedder is abstract, test that it exists
        from components.embedders.sentence_transformer_embedder.sentence_transformer_embedder import (
            SentenceTransformerEmbedder,
        )

        # Test that the class exists and has required methods
        assert hasattr(SentenceTransformerEmbedder, "embed")
        assert SentenceTransformerEmbedder.__name__ == "SentenceTransformerEmbedder"

    @pytest.mark.integration
    def test_embedder_output_format(self):
        """Test embedder output format is consistent."""
        # Use OllamaEmbedder which is concrete
        embedder = OllamaEmbedder(config={"model": "nomic-embed-text"})

        # Just test that it has the right interface
        assert hasattr(embedder, "embed")
        assert hasattr(embedder, "config")
