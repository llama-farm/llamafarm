"""Tests for runtime-specific model format detection utilities."""

from unittest.mock import Mock, patch

import pytest


class TestDetectModelFormat:
    """Test model format detection (runtime-specific)."""

    @patch("utils.model_format.HfApi")
    def test_detect_model_format_gguf(self, mock_hf_api_class):
        """Test detecting GGUF format."""
        from utils.model_format import clear_format_cache, detect_model_format

        # Clear cache for fresh test
        clear_format_cache()

        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "README.md",
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
        ]
        mock_hf_api_class.return_value = mock_api

        # Test
        result = detect_model_format("test/model")

        # Verify
        assert result == "gguf"

    @patch("utils.model_format.HfApi")
    def test_detect_model_format_transformers(self, mock_hf_api_class):
        """Test detecting transformers format."""
        from utils.model_format import clear_format_cache, detect_model_format

        # Clear cache for fresh test
        clear_format_cache()

        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
        ]
        mock_hf_api_class.return_value = mock_api

        # Test
        result = detect_model_format("test/model")

        # Verify
        assert result == "transformers"

    @patch("utils.model_format.HfApi")
    def test_detect_model_format_strips_quantization_suffix(self, mock_hf_api_class):
        """
        Test that detect_model_format() strips quantization suffix before calling HF API.

        This ensures 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M' is passed to HF API as
        'unsloth/Qwen3-1.7B-GGUF' (without the ':Q4_K_M' suffix).
        """
        from utils.model_format import clear_format_cache, detect_model_format

        # Clear cache to ensure fresh API call
        clear_format_cache()

        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ["model.Q4_K_M.gguf", "model.Q8_0.gguf"]
        mock_hf_api_class.return_value = mock_api

        # Test with quantization suffix
        result = detect_model_format("unsloth/Qwen3-1.7B-GGUF:Q4_K_M")

        # Verify HF API was called with CLEAN model ID (no suffix)
        mock_api.list_repo_files.assert_called_once_with(
            repo_id="unsloth/Qwen3-1.7B-GGUF",  # Should NOT have :Q4_K_M
            token=None,
        )

        # Verify correct format was detected
        assert result == "gguf"

    @patch("utils.model_format.HfApi")
    def test_caching_with_quantization_suffix(self, mock_hf_api_class):
        """
        Test that format detection cache works correctly with quantization suffixes.

        Both 'model:Q4_K_M' and 'model:Q8_0' should use the same cached result
        since they're the same base model.
        """
        from utils.model_format import clear_format_cache, detect_model_format

        # Clear cache for fresh test
        clear_format_cache()

        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ["model.Q4_K_M.gguf"]
        mock_hf_api_class.return_value = mock_api

        # First call with Q4_K_M suffix
        result1 = detect_model_format("test/model:Q4_K_M")
        assert result1 == "gguf"
        assert mock_api.list_repo_files.call_count == 1

        # Second call with Q8_0 suffix - should use cache (same base model)
        result2 = detect_model_format("test/model:Q8_0")
        assert result2 == "gguf"
        assert mock_api.list_repo_files.call_count == 1  # Still 1, cache was used

        # Third call without suffix - should also use cache
        result3 = detect_model_format("test/model")
        assert result3 == "gguf"
        assert mock_api.list_repo_files.call_count == 1  # Still 1, cache was used


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
