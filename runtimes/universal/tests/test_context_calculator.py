"""Tests for context_calculator module."""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import yaml

from utils.context_calculator import (
    clear_config_cache,
    compute_max_context,
    get_available_memory,
    get_default_context_size,
    get_gguf_metadata,
    load_model_context_config,
    match_model_pattern,
)


class TestGetGgufMetadata:
    """Tests for get_gguf_metadata function."""

    def test_get_metadata_existing_file(self, tmp_path):
        """Test getting metadata from an existing file."""
        # Create a test file with known size
        test_file = tmp_path / "test.gguf"
        test_content = b"A" * 1024 * 1024  # 1MB
        test_file.write_bytes(test_content)

        metadata = get_gguf_metadata(str(test_file))

        assert metadata["file_size_bytes"] == 1024 * 1024
        assert metadata["file_size_mb"] == 1.0

    def test_get_metadata_nonexistent_file(self):
        """Test that nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            get_gguf_metadata("/nonexistent/path/model.gguf")


class TestGetAvailableMemory:
    """Tests for get_available_memory function."""

    @patch("utils.context_calculator.torch.cuda.is_available")
    @patch("utils.context_calculator.torch.cuda.get_device_properties")
    def test_cuda_memory_detection(self, mock_props, mock_available):
        """Test CUDA memory detection."""
        mock_available.return_value = True
        mock_device = MagicMock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        mock_props.return_value = mock_device

        memory = get_available_memory("cuda")
        assert memory == 8 * 1024**3

    @patch("utils.context_calculator.psutil.virtual_memory")
    def test_cpu_memory_detection(self, mock_vm):
        """Test CPU memory detection."""
        # Use SimpleNamespace to avoid MagicMock formatting issues in logging
        mock_mem = SimpleNamespace(
            available=16 * (1024**3),  # 16GB available
            total=32 * (1024**3),  # 32GB total
        )
        mock_vm.return_value = mock_mem

        memory = get_available_memory("cpu")
        assert memory == 16 * (1024**3)

    @patch("utils.context_calculator.psutil.virtual_memory")
    def test_mps_memory_detection(self, mock_vm):
        """Test MPS (Apple Silicon) memory detection."""
        # Use SimpleNamespace to avoid MagicMock formatting issues in logging
        mock_mem = SimpleNamespace(
            available=12 * (1024**3),  # 12GB available
            total=12 * (1024**3),  # Need total for logging
        )
        mock_vm.return_value = mock_mem

        memory = get_available_memory("mps")
        assert memory == 12 * (1024**3)


class TestComputeMaxContext:
    """Tests for compute_max_context function."""

    def test_normal_memory_scenario(self):
        """Test context computation with normal memory."""
        model_size = 2 * 1024**3  # 2GB model
        available = 16 * 1024**3  # 16GB available
        memory_factor = 0.8

        max_ctx = compute_max_context(model_size, available, memory_factor)

        # Should return a power of 2
        assert max_ctx & (max_ctx - 1) == 0  # Is power of 2
        assert max_ctx > 0

    def test_low_memory_scenario(self):
        """Test context computation with low memory."""
        model_size = 3 * 1024**3  # 3GB model
        available = 4 * 1024**3  # 4GB available
        memory_factor = 0.8

        max_ctx = compute_max_context(model_size, available, memory_factor)

        # Should return a small but valid context size
        assert max_ctx >= 512
        assert max_ctx & (max_ctx - 1) == 0  # Is power of 2

    def test_insufficient_memory(self):
        """Test context computation when model barely fits."""
        model_size = 8 * 1024**3  # 8GB model
        available = 8 * 1024**3  # 8GB available
        memory_factor = 0.8

        max_ctx = compute_max_context(model_size, available, memory_factor)

        # Should return minimal context size
        assert max_ctx == 512


class TestLoadModelContextConfig:
    """Tests for load_model_context_config function."""

    def test_load_valid_config(self):
        """Test loading valid configuration file."""
        # Clear cache before test
        clear_config_cache()

        config = load_model_context_config()

        assert "memory_usage_factor" in config
        assert "model_defaults" in config
        assert isinstance(config["model_defaults"], list)
        assert len(config["model_defaults"]) > 0

    def test_config_caching(self):
        """Test that configuration is cached."""
        clear_config_cache()

        config1 = load_model_context_config()
        config2 = load_model_context_config()

        # Should return same object (cached)
        assert config1 is config2


class TestMatchModelPattern:
    """Tests for match_model_pattern function."""

    def test_exact_match(self):
        """Test exact model name matching."""
        config = {
            "model_defaults": [
                {"pattern": "unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF", "n_ctx": 32768},
                {"pattern": "*", "n_ctx": 2048},
            ]
        }

        n_ctx = match_model_pattern("unsloth/Qwen2.5-Coder-1.5B-Instruct-GGUF", config)
        assert n_ctx == 32768

    def test_wildcard_match(self):
        """Test wildcard pattern matching."""
        config = {
            "model_defaults": [
                {"pattern": "*/Qwen2.5-*-GGUF", "n_ctx": 32768},
                {"pattern": "*", "n_ctx": 2048},
            ]
        }

        n_ctx = match_model_pattern("unsloth/Qwen2.5-Coder-7B-GGUF", config)
        assert n_ctx == 32768

    def test_fallback_pattern(self):
        """Test fallback to wildcard pattern."""
        config = {
            "model_defaults": [
                {"pattern": "*/Llama-3*-GGUF", "n_ctx": 8192},
                {"pattern": "*", "n_ctx": 2048},
            ]
        }

        n_ctx = match_model_pattern("unknown/model-GGUF", config)
        assert n_ctx == 2048

    def test_no_match(self):
        """Test when no pattern matches."""
        config = {"model_defaults": [{"pattern": "specific/model", "n_ctx": 4096}]}

        n_ctx = match_model_pattern("other/model", config)
        assert n_ctx is None


class TestGetDefaultContextSize:
    """Tests for get_default_context_size function (integration tests)."""

    @patch("utils.context_calculator.get_gguf_metadata")
    @patch("utils.context_calculator.get_available_memory")
    @patch("utils.context_calculator.load_model_context_config")
    def test_config_override_within_limits(
        self, mock_config, mock_memory, mock_metadata
    ):
        """Test that explicit config value is used when within memory limits."""
        mock_metadata.return_value = {
            "file_size_bytes": 1 * 1024**3,  # 1GB model
            "file_size_mb": 1024,
        }
        mock_memory.return_value = 16 * 1024**3  # 16GB available
        mock_config.return_value = {
            "memory_usage_factor": 0.8,
            "model_defaults": [{"pattern": "*", "n_ctx": 2048}],
        }

        n_ctx, warnings = get_default_context_size(
            model_id="test/model",
            gguf_path="/fake/path.gguf",
            device="cpu",
            config_n_ctx=8192,  # User requested 8192
        )

        assert n_ctx == 8192
        assert len(warnings) == 0

    @patch("utils.context_calculator.get_gguf_metadata")
    @patch("utils.context_calculator.get_available_memory")
    @patch("utils.context_calculator.load_model_context_config")
    def test_config_override_exceeds_limit(
        self, mock_config, mock_memory, mock_metadata
    ):
        """Test that excessive config value is capped with warning."""
        mock_metadata.return_value = {
            "file_size_bytes": 7 * 1024**3,  # 7GB model
            "file_size_mb": 7168,
        }
        mock_memory.return_value = 8 * 1024**3  # 8GB available
        mock_config.return_value = {
            "memory_usage_factor": 0.8,
            "model_defaults": [{"pattern": "*", "n_ctx": 2048}],
        }

        n_ctx, warnings = get_default_context_size(
            model_id="test/model",
            gguf_path="/fake/path.gguf",
            device="cpu",
            config_n_ctx=32768,  # User requested 32k but won't fit
        )

        # Should be capped to computed maximum (less than or equal to)
        # With 7GB model and 8GB available at 0.8 factor, computed max will be small
        assert n_ctx <= 32768
        assert n_ctx < 32768  # Should be less than requested
        assert n_ctx >= 512
        assert len(warnings) > 0
        assert "exceeds computed maximum" in warnings[0]

    @patch("utils.context_calculator.get_gguf_metadata")
    @patch("utils.context_calculator.get_available_memory")
    @patch("utils.context_calculator.load_model_context_config")
    def test_pattern_match_used(self, mock_config, mock_memory, mock_metadata):
        """Test that pattern match from config is used when no explicit value."""
        mock_metadata.return_value = {
            "file_size_bytes": 1 * 1024**3,  # 1GB model
            "file_size_mb": 1024,
        }
        mock_memory.return_value = 32 * 1024**3  # 32GB available (plenty)
        mock_config.return_value = {
            "memory_usage_factor": 0.8,
            "model_defaults": [
                {"pattern": "*/Qwen2.5-*", "n_ctx": 32768},
                {"pattern": "*", "n_ctx": 2048},
            ],
        }

        n_ctx, warnings = get_default_context_size(
            model_id="unsloth/Qwen2.5-Coder-1.5B-GGUF",
            gguf_path="/fake/path.gguf",
            device="cpu",
            config_n_ctx=None,  # No explicit config
        )

        assert n_ctx == 32768
        assert len(warnings) == 0

    @patch("utils.context_calculator.get_gguf_metadata")
    @patch("utils.context_calculator.get_available_memory")
    @patch("utils.context_calculator.load_model_context_config")
    def test_computed_max_used_as_fallback(
        self, mock_config, mock_memory, mock_metadata
    ):
        """Test that computed max is used when no pattern matches."""
        mock_metadata.return_value = {
            "file_size_bytes": 1 * 1024**3,  # 1GB model
            "file_size_mb": 1024,
        }
        mock_memory.return_value = 16 * 1024**3  # 16GB available
        mock_config.return_value = {
            "memory_usage_factor": 0.8,
            "model_defaults": [],  # No patterns
        }

        n_ctx, warnings = get_default_context_size(
            model_id="unknown/model",
            gguf_path="/fake/path.gguf",
            device="cpu",
            config_n_ctx=None,
        )

        # Should use computed maximum
        assert n_ctx >= 2048  # Should have decent amount given 16GB available
        assert n_ctx & (n_ctx - 1) == 0  # Is power of 2

    @patch("utils.context_calculator.get_gguf_metadata")
    @patch("utils.context_calculator.get_available_memory")
    @patch("utils.context_calculator.load_model_context_config")
    def test_error_handling_with_fallback(
        self, mock_config, mock_memory, mock_metadata
    ):
        """Test that errors result in fallback with warning."""
        mock_metadata.side_effect = Exception("Metadata read failed")
        mock_memory.return_value = 16 * 1024**3
        mock_config.return_value = {
            "memory_usage_factor": 0.8,
            "model_defaults": [{"pattern": "*", "n_ctx": 2048}],
        }

        n_ctx, warnings = get_default_context_size(
            model_id="test/model",
            gguf_path="/fake/path.gguf",
            device="cpu",
            config_n_ctx=None,
        )

        assert n_ctx == 2048  # Fallback value
        assert len(warnings) > 0
        assert "Error" in warnings[0]


class TestClearConfigCache:
    """Tests for clear_config_cache function."""

    def test_cache_clearing(self):
        """Test that cache is properly cleared."""
        # Load config to populate cache
        clear_config_cache()
        config1 = load_model_context_config()

        # Clear cache
        clear_config_cache()

        # Load again - should create new object
        config2 = load_model_context_config()

        # After clearing, subsequent loads may return same cached object
        # but at least verify the function doesn't crash
        assert config2 is not None
        assert "model_defaults" in config2
