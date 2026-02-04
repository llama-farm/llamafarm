"""Tests for Universal Runtime embedder."""

from unittest.mock import Mock, patch

import pytest

from components.embedders.universal_embedder.universal_embedder import UniversalEmbedder
from utils.embedding_safety import EmbedderUnavailableError


class TestUniversalEmbedder:
    """Test suite for UniversalEmbedder."""

    def test_init_default_config(self):
        """Test initialization with default configuration."""
        embedder = UniversalEmbedder()

        assert embedder.name == "UniversalEmbedder"
        assert embedder.model == "sentence-transformers/all-MiniLM-L6-v2"
        assert embedder.batch_size == 32
        assert embedder.timeout == 120
        assert embedder.normalize is True
        assert "/v1" in embedder.base_url

    def test_init_custom_config(self):
        """Test initialization with custom configuration."""
        config = {
            "model": "BAAI/bge-base-en-v1.5",
            "base_url": "http://localhost:8080/v1",
            "api_key": "test-key",
            "batch_size": 16,
            "timeout": 60,
            "normalize": False,
        }
        embedder = UniversalEmbedder(config=config)

        assert embedder.model == "BAAI/bge-base-en-v1.5"
        assert embedder.base_url == "http://localhost:8080/v1"
        assert embedder.api_key == "test-key"
        assert embedder.batch_size == 16
        assert embedder.timeout == 60
        assert embedder.normalize is False

    def test_base_url_normalization(self):
        """Test that base_url is normalized to include /v1."""
        # Without /v1
        config1 = {"base_url": "http://localhost:11540"}
        embedder1 = UniversalEmbedder(config=config1)
        assert embedder1.base_url.endswith("/v1")

        # Already has /v1
        config2 = {"base_url": "http://localhost:11540/v1"}
        embedder2 = UniversalEmbedder(config=config2)
        assert embedder2.base_url == "http://localhost:11540/v1"

    def test_get_embedding_dimension(self):
        """Test embedding dimension detection."""
        test_cases = [
            ("sentence-transformers/all-MiniLM-L6-v2", 384),
            ("sentence-transformers/all-mpnet-base-v2", 768),
            ("BAAI/bge-base-en-v1.5", 768),
            ("BAAI/bge-large-en-v1.5", 1024),
            ("nomic-ai/nomic-embed-text-v1.5", 768),
            ("unknown-model", 768),  # Default
        ]

        for model, expected_dim in test_cases:
            embedder = UniversalEmbedder(config={"model": model})
            assert embedder.get_embedding_dimension() == expected_dim

    @patch("requests.get")
    def test_validate_config_success(self, mock_get):
        """Test successful configuration validation."""
        # Mock health check
        mock_health_response = Mock()
        mock_health_response.status_code = 200

        # Mock models endpoint
        mock_models_response = Mock()
        mock_models_response.status_code = 200

        mock_get.side_effect = [mock_health_response, mock_models_response]

        embedder = UniversalEmbedder()
        assert embedder.validate_config() is True

    @patch("time.sleep")
    @patch("requests.get")
    def test_validate_config_failure(self, mock_get, mock_sleep):
        """Test configuration validation failure after all retries."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        embedder = UniversalEmbedder()
        assert embedder.validate_config() is False
        # Should have retried (4 attempts total = 3 sleeps)
        assert mock_sleep.call_count == 3

    @patch("time.sleep")
    @patch("requests.get")
    def test_validate_config_exception(self, mock_get, mock_sleep):
        """Test configuration validation with exception after all retries."""
        mock_get.side_effect = Exception("Connection error")

        embedder = UniversalEmbedder()
        assert embedder.validate_config() is False
        # Should have retried (4 attempts total = 3 sleeps)
        assert mock_sleep.call_count == 3

    @patch("time.sleep")
    @patch("requests.get")
    def test_validate_config_retry_then_success(self, mock_get, mock_sleep):
        """Test validation succeeds after initial failures."""
        mock_health_ok = Mock()
        mock_health_ok.status_code = 200
        mock_models_ok = Mock()
        mock_models_ok.status_code = 200

        # First attempt: exception, second attempt: success
        mock_get.side_effect = [
            Exception("Connection refused"),
            mock_health_ok,
            mock_models_ok,
        ]

        embedder = UniversalEmbedder()
        assert embedder.validate_config() is True
        # One retry sleep
        assert mock_sleep.call_count == 1

    @patch("time.sleep")
    @patch("requests.get")
    def test_validate_config_health_retry_then_success(self, mock_get, mock_sleep):
        """Test validation retries when health check fails then succeeds."""
        mock_health_fail = Mock()
        mock_health_fail.status_code = 503
        mock_health_ok = Mock()
        mock_health_ok.status_code = 200
        mock_models_ok = Mock()
        mock_models_ok.status_code = 200

        mock_get.side_effect = [
            mock_health_fail,  # attempt 1: health fails
            mock_health_ok,    # attempt 2: health ok
            mock_models_ok,    # attempt 2: models ok
        ]

        embedder = UniversalEmbedder()
        assert embedder.validate_config() is True
        assert mock_sleep.call_count == 1

    @patch("requests.post")
    def test_embed_single_text(self, mock_post):
        """Test embedding a single text."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        embedder = UniversalEmbedder()
        result = embedder.embed_text("test text")

        assert result == [0.1, 0.2, 0.3]
        assert mock_post.called

    @patch("requests.post")
    def test_embed_multiple_texts(self, mock_post):
        """Test embedding multiple texts."""
        mock_response = Mock()
        mock_response.status_code = 200
        # Now each text is embedded individually, so mock returns single embedding per call
        mock_response.json.side_effect = [
            {"data": [{"embedding": [0.1, 0.2, 0.3]}]},
            {"data": [{"embedding": [0.4, 0.5, 0.6]}]},
        ]
        mock_post.return_value = mock_response

        embedder = UniversalEmbedder(config={"batch_size": 2})
        texts = ["text 1", "text 2"]
        results = embedder.embed(texts)

        assert len(results) == 2
        assert results[0] == [0.1, 0.2, 0.3]
        assert results[1] == [0.4, 0.5, 0.6]

    @patch("requests.post")
    def test_embed_empty_list(self, mock_post):
        """Test embedding empty list."""
        embedder = UniversalEmbedder()
        results = embedder.embed([])

        assert results == []
        assert not mock_post.called

    @patch("requests.post")
    def test_embed_error_handling(self, mock_post):
        """Test error handling during embedding with fail_fast enabled (default)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        embedder = UniversalEmbedder()

        # With fail_fast=True (default), should raise EmbedderUnavailableError
        with pytest.raises(EmbedderUnavailableError):
            embedder.embed_text("test text")

    @patch("requests.post")
    def test_embed_error_handling_legacy_mode(self, mock_post):
        """Test error handling during embedding with fail_fast=False (legacy mode)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response

        embedder = UniversalEmbedder(config={"fail_fast": False})
        expected_dim = embedder.get_embedding_dimension()
        result = embedder.embed_text("test text")

        # With fail_fast=False, should return zero vector on error
        assert len(result) == expected_dim  # Ensure proper dimensions, not empty
        assert result == [0.0] * expected_dim

    @patch("requests.post")
    def test_embed_batch_processing(self, mock_post):
        """Test batch processing of texts."""
        mock_response = Mock()
        mock_response.status_code = 200

        # Now each text is embedded individually via embed_text
        mock_response.json.side_effect = [
            {"data": [{"embedding": [0.1]}]},
            {"data": [{"embedding": [0.2]}]},
            {"data": [{"embedding": [0.3]}]},
        ]
        mock_post.return_value = mock_response

        embedder = UniversalEmbedder(config={"batch_size": 2})
        texts = ["text 1", "text 2", "text 3"]
        results = embedder.embed(texts)

        assert len(results) == 3
        assert mock_post.call_count == 3  # One call per text

    def test_get_description(self):
        """Test class description."""
        description = UniversalEmbedder.get_description()
        assert "Universal Runtime" in description
        assert "HuggingFace" in description

    def test_check_model_availability(self):
        """Test model availability check."""
        with patch.object(UniversalEmbedder, "validate_config", return_value=True):
            embedder = UniversalEmbedder()
            assert embedder._check_model_availability() is True

    @patch("requests.post")
    def test_api_key_header(self, mock_post):
        """Test that API key is included in headers."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"embedding": [0.1, 0.2, 0.3]}]}
        mock_post.return_value = mock_response

        embedder = UniversalEmbedder(config={"api_key": "test-key"})
        embedder.embed_text("test")

        # Check that Authorization header was set
        call_kwargs = mock_post.call_args[1]
        headers = call_kwargs.get("headers", {})
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test-key"

    @patch("requests.post")
    def test_empty_text_handling(self, mock_post):
        """Test handling of empty or whitespace-only text with fail_fast enabled (default)."""
        embedder = UniversalEmbedder()

        # Empty string - should raise with fail_fast=True
        with pytest.raises(EmbedderUnavailableError):
            embedder.embed_text("")

        # Whitespace only - should also raise
        with pytest.raises(EmbedderUnavailableError):
            embedder.embed_text("   ")

        # Should not call API
        assert not mock_post.called

    @patch("requests.post")
    def test_empty_text_handling_legacy_mode(self, mock_post):
        """Test handling of empty or whitespace-only text with fail_fast=False (legacy mode)."""
        embedder = UniversalEmbedder(config={"fail_fast": False})
        expected_dim = embedder.get_embedding_dimension()

        # Empty string - should return zero vector
        result = embedder.embed_text("")
        assert len(result) == expected_dim  # Ensure proper dimensions, not empty
        assert result == [0.0] * expected_dim

        # Whitespace only - should return zero vector
        result = embedder.embed_text("   ")
        assert len(result) == expected_dim  # Ensure proper dimensions, not empty
        assert result == [0.0] * expected_dim

        # Should not call API
        assert not mock_post.called
