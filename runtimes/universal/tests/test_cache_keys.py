"""
Test that cache keys properly differentiate models with different configurations.

This test verifies that:
1. GGUF models with different quantizations get separate cache entries
2. GGUF models with different context sizes get separate cache entries
3. GGUF models with different n_batch get separate cache entries
4. GGUF models with different n_gpu_layers get separate cache entries
5. GGUF models with different flash_attn settings get separate cache entries
6. GGUF models with different use_mmap settings get separate cache entries
7. GGUF models with different use_mlock settings get separate cache entries
8. The same model configuration reuses the same cache entry
"""

from server import _make_encoder_cache_key, _make_language_cache_key


class TestCacheKeys:
    """Test cache key generation for model loading."""

    def test_load_language_cache_key_includes_quantization(self):
        """Test that different quantizations result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192

        # Use actual production cache key generation
        cache_key1 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, preferred_quantization="Q4_K_M"
        )
        cache_key2 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, preferred_quantization="Q8_0"
        )

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert "quantQ4_K_M" in cache_key1
        assert "quantQ8_0" in cache_key2

    def test_load_language_cache_key_includes_context_size(self):
        """Test that different context sizes result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        quant = "Q4_K_M"

        # Test case 2: Same model, different context sizes should have different keys
        cache_key1 = _make_language_cache_key(
            model_id, n_ctx=8192, preferred_quantization=quant
        )
        cache_key2 = _make_language_cache_key(
            model_id, n_ctx=16384, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert "ctx8192" in cache_key1
        assert "ctx16384" in cache_key2

    def test_load_language_cache_key_includes_batch_size(self):
        """Test that different n_batch values result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different batch sizes should have different keys
        cache_key1 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_batch=512, preferred_quantization=quant
        )
        cache_key2 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_batch=2048, preferred_quantization=quant
        )
        cache_key_auto = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_batch=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert cache_key1 != cache_key_auto
        assert "batch512" in cache_key1
        assert "batch2048" in cache_key2
        assert "batchauto" in cache_key_auto

    def test_load_language_cache_key_includes_gpu_layers(self):
        """Test that different n_gpu_layers result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different GPU layers should have different keys
        cache_key1 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_gpu_layers=29, preferred_quantization=quant
        )
        cache_key2 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_gpu_layers=-1, preferred_quantization=quant
        )
        cache_key_auto = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_gpu_layers=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert cache_key1 != cache_key_auto
        assert "gpu29" in cache_key1
        assert "gpu-1" in cache_key2
        assert "gpuauto" in cache_key_auto

    def test_load_language_cache_key_includes_threads(self):
        """Test that different n_threads values result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different thread counts should have different keys
        cache_key1 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_threads=6, preferred_quantization=quant
        )
        cache_key2 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_threads=8, preferred_quantization=quant
        )
        cache_key_auto = _make_language_cache_key(
            model_id, n_ctx=n_ctx, n_threads=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key1 != cache_key2
        assert cache_key1 != cache_key_auto
        assert "threads6" in cache_key1
        assert "threads8" in cache_key2
        assert "threadsauto" in cache_key_auto

    def test_load_language_cache_key_includes_flash_attn(self):
        """Test that different flash_attn settings result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different flash_attn should have different keys
        cache_key_true = _make_language_cache_key(
            model_id, n_ctx=n_ctx, flash_attn=True, preferred_quantization=quant
        )
        cache_key_false = _make_language_cache_key(
            model_id, n_ctx=n_ctx, flash_attn=False, preferred_quantization=quant
        )
        cache_key_default = _make_language_cache_key(
            model_id, n_ctx=n_ctx, flash_attn=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key_true != cache_key_false
        assert cache_key_true != cache_key_default
        assert "flashTrue" in cache_key_true
        assert "flashFalse" in cache_key_false
        assert "flashdefault" in cache_key_default

    def test_load_language_cache_key_includes_mmap(self):
        """Test that different use_mmap settings result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different use_mmap should have different keys
        cache_key_true = _make_language_cache_key(
            model_id, n_ctx=n_ctx, use_mmap=True, preferred_quantization=quant
        )
        cache_key_false = _make_language_cache_key(
            model_id, n_ctx=n_ctx, use_mmap=False, preferred_quantization=quant
        )
        cache_key_default = _make_language_cache_key(
            model_id, n_ctx=n_ctx, use_mmap=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key_true != cache_key_false
        assert cache_key_true != cache_key_default
        assert "mmapTrue" in cache_key_true
        assert "mmapFalse" in cache_key_false
        assert "mmapdefault" in cache_key_default

    def test_load_language_cache_key_includes_mlock(self):
        """Test that different use_mlock settings result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different use_mlock should have different keys
        cache_key_true = _make_language_cache_key(
            model_id, n_ctx=n_ctx, use_mlock=True, preferred_quantization=quant
        )
        cache_key_false = _make_language_cache_key(
            model_id, n_ctx=n_ctx, use_mlock=False, preferred_quantization=quant
        )
        cache_key_default = _make_language_cache_key(
            model_id, n_ctx=n_ctx, use_mlock=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key_true != cache_key_false
        assert cache_key_true != cache_key_default
        assert "mlockTrue" in cache_key_true
        assert "mlockFalse" in cache_key_false
        assert "mlockdefault" in cache_key_default

    def test_load_language_cache_key_includes_cache_type_k(self):
        """Test that different cache_type_k settings result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different cache_type_k should have different keys
        cache_key_q4 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, cache_type_k="q4_0", preferred_quantization=quant
        )
        cache_key_q8 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, cache_type_k="q8_0", preferred_quantization=quant
        )
        cache_key_default = _make_language_cache_key(
            model_id, n_ctx=n_ctx, cache_type_k=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key_q4 != cache_key_q8
        assert cache_key_q4 != cache_key_default
        assert "cachekq4_0" in cache_key_q4
        assert "cachekq8_0" in cache_key_q8
        assert "cachekdefault" in cache_key_default

    def test_load_language_cache_key_includes_cache_type_v(self):
        """Test that different cache_type_v settings result in different cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same model, different cache_type_v should have different keys
        cache_key_q4 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, cache_type_v="q4_0", preferred_quantization=quant
        )
        cache_key_f16 = _make_language_cache_key(
            model_id, n_ctx=n_ctx, cache_type_v="f16", preferred_quantization=quant
        )
        cache_key_default = _make_language_cache_key(
            model_id, n_ctx=n_ctx, cache_type_v=None, preferred_quantization=quant
        )

        # Cache keys should be different
        assert cache_key_q4 != cache_key_f16
        assert cache_key_q4 != cache_key_default
        assert "cachevq4_0" in cache_key_q4
        assert "cachevf16" in cache_key_f16
        assert "cachevdefault" in cache_key_default

    def test_load_language_cache_key_none_quantization(self):
        """Test that None quantization uses 'default' in cache key."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192

        # Test case: None quantization should use "default"
        cache_key = _make_language_cache_key(model_id, n_ctx=n_ctx)

        assert "quantdefault" in cache_key

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
        # Note: cache keys now include max_length suffix (lenauto when not specified)
        assert cache_key1 == f"encoder:{task}:{model_format}:{model_id}:quantQ4_K_M:lenauto"
        assert cache_key2 == f"encoder:{task}:{model_format}:{model_id}:quantQ8_0:lenauto"

    def test_load_encoder_cache_key_none_quantization(self):
        """Test that encoder models with None quantization use 'default' in cache key."""
        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        task = "embedding"
        model_format = "gguf"

        # Test case 5: None quantization should use "default"
        cache_key = _make_encoder_cache_key(model_id, task, model_format, None)

        # Note: cache keys now include max_length suffix (lenauto when not specified)
        assert cache_key == f"encoder:{task}:{model_format}:{model_id}:quantdefault:lenauto"

    def test_same_configuration_uses_same_cache_key(self):
        """Test that identical configurations produce identical cache keys."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"
        n_ctx = 8192
        quant = "Q4_K_M"

        # Test case: Same configuration should produce same cache key
        cache_key1 = _make_language_cache_key(
            model_id,
            n_ctx=n_ctx,
            n_batch=512,
            n_gpu_layers=29,
            n_threads=6,
            flash_attn=True,
            use_mmap=True,
            use_mlock=False,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            preferred_quantization=quant,
        )
        cache_key2 = _make_language_cache_key(
            model_id,
            n_ctx=n_ctx,
            n_batch=512,
            n_gpu_layers=29,
            n_threads=6,
            flash_attn=True,
            use_mmap=True,
            use_mlock=False,
            cache_type_k="q4_0",
            cache_type_v="q4_0",
            preferred_quantization=quant,
        )

        # Cache keys should be identical
        assert cache_key1 == cache_key2

    def test_jetson_optimized_config_cache_key(self):
        """Test cache key for Jetson-optimized configuration."""
        model_id = "unsloth/Qwen3-1.7B-GGUF"

        # Jetson Orin Nano optimized settings
        cache_key = _make_language_cache_key(
            model_id,
            n_ctx=2048,
            n_batch=512,  # Critical: reduces compute buffer from 1.2GB to ~300MB
            n_gpu_layers=29,  # All layers on GPU for Qwen
            n_threads=6,  # Orin Nano has 6 CPU cores
            flash_attn=True,  # Use Ampere optimizations
            use_mmap=True,  # Efficient memory swapping
            use_mlock=False,  # Allow OS memory management on 8GB device
            cache_type_k="q4_0",  # Quantize KV cache keys for memory savings
            cache_type_v="q4_0",  # Quantize KV cache values for memory savings
            preferred_quantization="Q4_K_M",
        )

        # Verify all optimized settings are in the cache key
        assert "ctx2048" in cache_key
        assert "batch512" in cache_key
        assert "gpu29" in cache_key
        assert "threads6" in cache_key
        assert "flashTrue" in cache_key
        assert "mmapTrue" in cache_key
        assert "mlockFalse" in cache_key
        assert "cachekq4_0" in cache_key
        assert "cachevq4_0" in cache_key
        assert "quantQ4_K_M" in cache_key
