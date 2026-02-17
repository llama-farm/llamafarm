"""Tests for GGUF metadata cache."""

import pytest

from utils.gguf_metadata_cache import (
    GGUFMetadata,
    clear_metadata_cache,
    get_cache_stats,
    get_gguf_metadata_cached,
)


class TestGGUFMetadataCache:
    """Tests for GGUF metadata caching."""

    def setup_method(self):
        """Clear cache before each test."""
        clear_metadata_cache()

    def test_cache_stats_empty(self):
        """Test that cache starts empty."""
        stats = get_cache_stats()
        assert stats["entry_count"] == 0
        assert stats["cached_paths"] == []

    def test_clear_metadata_cache(self):
        """Test clearing the cache."""
        # This should not raise even if cache is empty
        clear_metadata_cache()
        stats = get_cache_stats()
        assert stats["entry_count"] == 0

    def test_file_not_found_raises(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_gguf_metadata_cached("/nonexistent/path/model.gguf")

    def test_gguf_metadata_dataclass(self):
        """Test GGUFMetadata dataclass fields."""
        metadata = GGUFMetadata(
            file_path="/test/path.gguf",
            file_size_bytes=1000,
            file_size_mb=1.0,
            n_ctx_train=4096,
            chat_template="test template",
            bos_token="<s>",
            eos_token="</s>",
            n_layer=32,
            n_head_kv=8,
            head_k_size=128,
            head_v_size=128,
        )

        assert metadata.file_path == "/test/path.gguf"
        assert metadata.file_size_bytes == 1000
        assert metadata.file_size_mb == 1.0
        assert metadata.n_ctx_train == 4096
        assert metadata.chat_template == "test template"
        assert metadata.bos_token == "<s>"
        assert metadata.eos_token == "</s>"
        assert metadata.n_layer == 32
        assert metadata.n_head_kv == 8
        assert metadata.head_k_size == 128
        assert metadata.head_v_size == 128

    def test_gguf_metadata_defaults(self):
        """Test GGUFMetadata default values."""
        metadata = GGUFMetadata(
            file_path="/test/path.gguf",
            file_size_bytes=1000,
            file_size_mb=1.0,
        )

        assert metadata.n_ctx_train is None
        assert metadata.chat_template is None
        assert metadata.bos_token == ""
        assert metadata.eos_token == ""
        assert metadata.n_layer is None
        assert metadata.n_head_kv is None
        assert metadata.head_k_size is None
        assert metadata.head_v_size is None
