"""
Test that cache keys properly differentiate models with different configurations.

This test verifies that:
1. GGUF models with different quantizations get separate cache entries
2. GGUF models with different context sizes get separate cache entries
3. The same model configuration reuses the same cache entry
"""

import pytest
from server import _make_language_cache_key, _make_encoder_cache_key


class TestCacheKeys:
    """Test cache key generation for model loading."""

    def test_load_language_cache_key_includes_quantization(self):
        """Test that different quantizations result in different cache keys."""
        # Test case 1: Same model, different quantizations should have different keys
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192

        # Use actual production cache key generation
        cache_key1 = _make_language_cache_key(model_id, n_ctx, "Q4_K_M")
        cache_key2 = _make_language_cache_key(model_id, n_ctx, "Q8_0")

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert cache_key1 == f"language:{model_id}:ctx{n_ctx}:quantQ4_K_M"
        assert cache_key2 == f"language:{model_id}:ctx{n_ctx}:quantQ8_0"

    def test_load_language_cache_key_includes_context_size(self):
        """Test that different context sizes result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        quant = "Q4_K_M"

        # Test case 2: Same model, different context sizes should have different keys
        cache_key1 = _make_language_cache_key(model_id, 8192, quant)
        cache_key2 = _make_language_cache_key(model_id, 16384, quant)

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert cache_key1 == f"language:{model_id}:ctx8192:quant{quant}"
        assert cache_key2 == f"language:{model_id}:ctx16384:quant{quant}"

    def test_load_language_cache_key_none_quantization(self):
        """Test that None quantization uses 'default' in cache key."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192

        # Test case 3: None quantization should use "default"
        cache_key = _make_language_cache_key(model_id, n_ctx, None)

        assert cache_key == f"language:{model_id}:ctx{n_ctx}:quantdefault"

    def test_load_encoder_cache_key_includes_quantization(self):
        """Test that encoder models with different quantizations get different cache keys."""
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        task = "embedding"
        model_format = "gguf"

        # Test case 4: Same encoder model, different quantizations should have different keys
        cache_key1 = _make_encoder_cache_key(model_id, task, model_format, "Q4_K_M")
        cache_key2 = _make_encoder_cache_key(model_id, task, model_format, "Q8_0")

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert cache_key1 == f"encoder:{task}:{model_format}:{model_id}:quantQ4_K_M"
        assert cache_key2 == f"encoder:{task}:{model_format}:{model_id}:quantQ8_0"

    def test_load_encoder_cache_key_none_quantization(self):
        """Test that encoder models with None quantization use 'default' in cache key."""
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        task = "embedding"
        model_format = "gguf"

        # Test case 5: None quantization should use "default"
        cache_key = _make_encoder_cache_key(model_id, task, model_format, None)

        assert cache_key == f"encoder:{task}:{model_format}:{model_id}:quantdefault"

    def test_same_configuration_uses_same_cache_key(self):
        """Test that identical configurations produce identical cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case 6: Same configuration should produce same cache key
        cache_key1 = _make_language_cache_key(model_id, n_ctx, quant)
        cache_key2 = _make_language_cache_key(model_id, n_ctx, quant)

        # Cache keys should be identical
        assert cache_key1 == cache_key2
