"""Tests for model format detection and GGUF file selection."""

import pytest
from unittest.mock import Mock, patch
from llamafarm_common import (
    parse_quantization_from_filename,
    parse_model_with_quantization,
    select_gguf_file,
)
from utils.model_format import list_gguf_files


class TestParseQuantizationFromFilename:
    """Test parsing quantization types from GGUF filenames."""

    def test_parse_q4_k_m(self):
        """Test parsing Q4_K_M quantization."""
        filename = "qwen3-1.7b.Q4_K_M.gguf"
        result = parse_quantization_from_filename(filename)
        assert result == "Q4_K_M"

    def test_parse_q8_0(self):
        """Test parsing Q8_0 quantization."""
        filename = "model.Q8_0.gguf"
        result = parse_quantization_from_filename(filename)
        assert result == "Q8_0"

    def test_parse_f16(self):
        """Test parsing F16 quantization."""
        filename = "llama-3.2-3b.F16.gguf"
        result = parse_quantization_from_filename(filename)
        assert result == "F16"

    def test_parse_q5_k_s(self):
        """Test parsing Q5_K_S quantization."""
        filename = "model.Q5_K_S.gguf"
        result = parse_quantization_from_filename(filename)
        assert result == "Q5_K_S"

    def test_parse_case_insensitive(self):
        """Test that parsing is case-insensitive."""
        filename = "model.q4_k_m.gguf"
        result = parse_quantization_from_filename(filename)
        assert result == "Q4_K_M"

    def test_parse_no_quantization(self):
        """Test filename with no recognizable quantization."""
        filename = "model.gguf"
        result = parse_quantization_from_filename(filename)
        assert result is None

    def test_parse_complex_filename(self):
        """Test parsing from complex filename with multiple dots."""
        filename = "unsloth_qwen3-1.7b-instruct.Q4_K_M.gguf"
        result = parse_quantization_from_filename(filename)
        assert result == "Q4_K_M"


class TestParseModelWithQuantization:
    """Test parsing model names with quantization suffix."""

    def test_parse_with_q4_k_m(self):
        """Test parsing model name with Q4_K_M quantization."""
        model_name = "unsloth/Qwen3-4B-GGUF:Q4_K_M"
        model_id, quantization = parse_model_with_quantization(model_name)
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quantization == "Q4_K_M"

    def test_parse_with_lowercase_quantization(self):
        """Test parsing with lowercase quantization (should be normalized)."""
        model_name = "unsloth/Qwen3-4B-GGUF:q8_0"
        model_id, quantization = parse_model_with_quantization(model_name)
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quantization == "Q8_0"

    def test_parse_without_quantization(self):
        """Test parsing model name without quantization suffix."""
        model_name = "unsloth/Qwen3-4B-GGUF"
        model_id, quantization = parse_model_with_quantization(model_name)
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quantization is None

    def test_parse_with_multiple_colons(self):
        """Test that only the last colon is used for quantization."""
        model_name = "org:user/model:Q4_K_M"
        model_id, quantization = parse_model_with_quantization(model_name)
        assert model_id == "org:user/model"
        assert quantization == "Q4_K_M"

    def test_parse_with_empty_quantization(self):
        """Test parsing with empty string after colon."""
        model_name = "unsloth/Qwen3-4B-GGUF:"
        model_id, quantization = parse_model_with_quantization(model_name)
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quantization is None


class TestSelectGGUFFile:
    """Test GGUF file selection logic."""

    def test_select_single_file(self):
        """Test that single file is returned regardless of quantization."""
        files = ["model.Q8_0.gguf"]
        result = select_gguf_file(files)
        assert result == "model.Q8_0.gguf"

    def test_select_default_q4_k_m(self):
        """Test that Q4_K_M is selected by default."""
        files = [
            "model.Q2_K.gguf",
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
            "model.F16.gguf",
        ]
        result = select_gguf_file(files)
        assert result == "model.Q4_K_M.gguf"

    def test_select_preferred_quantization(self):
        """Test selecting specific preferred quantization."""
        files = [
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
            "model.F16.gguf",
        ]
        result = select_gguf_file(files, preferred_quantization="Q8_0")
        assert result == "model.Q8_0.gguf"

    def test_select_preferred_case_insensitive(self):
        """Test that preferred quantization matching is case-insensitive."""
        files = [
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
        ]
        result = select_gguf_file(files, preferred_quantization="q8_0")
        assert result == "model.Q8_0.gguf"

    def test_select_fallback_when_preferred_not_found(self):
        """Test fallback to default when preferred not found."""
        files = [
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
        ]
        result = select_gguf_file(files, preferred_quantization="F16")
        # Should fall back to Q4_K_M (default preference)
        assert result == "model.Q4_K_M.gguf"

    def test_select_priority_order(self):
        """Test that selection follows priority order."""
        # Test Q5_K_M selected when Q4_K_M not available
        files = ["model.Q8_0.gguf", "model.Q5_K_M.gguf", "model.F16.gguf"]
        result = select_gguf_file(files)
        assert result == "model.Q5_K_M.gguf"

        # Test Q8_0 selected when neither Q4 nor Q5 available
        files = ["model.Q8_0.gguf", "model.F16.gguf", "model.Q2_K.gguf"]
        result = select_gguf_file(files)
        assert result == "model.Q8_0.gguf"

    def test_select_first_when_no_quantization_found(self):
        """Test that first file is selected when no quantization recognized."""
        files = ["model_a.gguf", "model_b.gguf"]
        result = select_gguf_file(files)
        assert result == "model_a.gguf"

    def test_select_empty_list_returns_none(self):
        """Test that empty file list returns None."""
        result = select_gguf_file([])
        assert result is None


class TestListGGUFFiles:
    """Test listing GGUF files from HuggingFace repositories."""

    @patch("utils.model_format.HfApi")
    def test_list_gguf_files_filters_correctly(self, mock_hf_api_class):
        """Test that only .gguf files are returned."""
        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "README.md",
            "config.json",
            "model.Q4_K_M.gguf",
            "model.Q8_0.gguf",
            "tokenizer.json",
            "model.F16.gguf",
        ]
        mock_hf_api_class.return_value = mock_api

        # Test
        result = list_gguf_files("test/model")

        # Verify
        assert len(result) == 3
        assert "model.Q4_K_M.gguf" in result
        assert "model.Q8_0.gguf" in result
        assert "model.F16.gguf" in result
        assert "README.md" not in result
        assert "config.json" not in result

    @patch("utils.model_format.HfApi")
    def test_list_gguf_files_with_token(self, mock_hf_api_class):
        """Test that token is passed to HuggingFace API."""
        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ["model.gguf"]
        mock_hf_api_class.return_value = mock_api

        # Test
        list_gguf_files("test/model", token="test_token")

        # Verify token was passed
        mock_api.list_repo_files.assert_called_once_with(
            repo_id="test/model", token="test_token"
        )

    @patch("utils.model_format.HfApi")
    def test_list_gguf_files_no_gguf_files(self, mock_hf_api_class):
        """Test handling when no GGUF files exist."""
        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "README.md",
            "config.json",
            "model.safetensors",
        ]
        mock_hf_api_class.return_value = mock_api

        # Test
        result = list_gguf_files("test/model")

        # Verify
        assert len(result) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
