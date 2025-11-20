"""Tests for model utility functions."""

import pytest
from llamafarm_common.model_utils import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    parse_model_with_quantization,
    parse_quantization_from_filename,
    select_gguf_file,
)


class TestParseModelWithQuantization:
    """Test parsing model names with quantization suffix."""

    def test_parse_with_quantization(self):
        """Test parsing model name with quantization."""
        model_id, quant = parse_model_with_quantization("unsloth/Qwen3-4B-GGUF:Q4_K_M")
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quant == "Q4_K_M"

    def test_parse_lowercase_quantization(self):
        """Test parsing with lowercase quantization (normalized to uppercase)."""
        model_id, quant = parse_model_with_quantization("unsloth/Qwen3-4B-GGUF:q8_0")
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quant == "Q8_0"

    def test_parse_without_quantization(self):
        """Test parsing model name without quantization suffix."""
        model_id, quant = parse_model_with_quantization("unsloth/Qwen3-4B-GGUF")
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quant is None


class TestParseQuantizationFromFilename:
    """Test parsing quantization from GGUF filenames."""

    def test_parse_q4_k_m(self):
        """Test parsing Q4_K_M quantization."""
        result = parse_quantization_from_filename("qwen3-1.7b.Q4_K_M.gguf")
        assert result == "Q4_K_M"

    def test_parse_q8_0(self):
        """Test parsing Q8_0 quantization."""
        result = parse_quantization_from_filename("model.Q8_0.gguf")
        assert result == "Q8_0"

    def test_parse_no_quantization(self):
        """Test filename with no recognizable quantization."""
        result = parse_quantization_from_filename("model.gguf")
        assert result is None


class TestSelectGGUFFile:
    """Test GGUF file selection logic."""

    def test_select_default_q4_k_m(self):
        """Test that Q4_K_M is selected by default."""
        files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.F16.gguf"]
        result = select_gguf_file(files)
        assert result == "model.Q4_K_M.gguf"

    def test_select_preferred_quantization(self):
        """Test selecting a preferred quantization."""
        files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "model.F16.gguf"]
        result = select_gguf_file(files, preferred_quantization="Q8_0")
        assert result == "model.Q8_0.gguf"

    def test_select_empty_list_returns_none(self):
        """Test that empty file list returns None."""
        result = select_gguf_file([])
        assert result is None

    def test_preference_order(self):
        """Test that GGUF_QUANTIZATION_PREFERENCE_ORDER is defined correctly."""
        assert GGUF_QUANTIZATION_PREFERENCE_ORDER[0] == "Q4_K_M"
        assert "Q8_0" in GGUF_QUANTIZATION_PREFERENCE_ORDER
        assert "F16" in GGUF_QUANTIZATION_PREFERENCE_ORDER
