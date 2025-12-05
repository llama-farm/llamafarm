"""Tests for model utility functions in llamafarm_common."""

import os
import tempfile
from unittest.mock import Mock, patch

import pytest

from llamafarm_common.model_utils import (
    GGUF_QUANTIZATION_PREFERENCE_ORDER,
    get_gguf_file_path,
    list_gguf_files,
    parse_model_with_quantization,
    parse_quantization_from_filename,
    select_gguf_file,
    select_gguf_file_with_logging,
)


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

    def test_parse_with_quantization(self):
        """Test parsing model name with quantization."""
        model_id, quant = parse_model_with_quantization("unsloth/Qwen3-4B-GGUF:Q4_K_M")
        assert model_id == "unsloth/Qwen3-4B-GGUF"
        assert quant == "Q4_K_M"

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

    def test_preference_order_defined(self):
        """Test that GGUF_QUANTIZATION_PREFERENCE_ORDER is defined correctly."""
        assert GGUF_QUANTIZATION_PREFERENCE_ORDER[0] == "Q4_K_M"
        assert "Q8_0" in GGUF_QUANTIZATION_PREFERENCE_ORDER
        assert "F16" in GGUF_QUANTIZATION_PREFERENCE_ORDER


class TestListGGUFFiles:
    """Test listing GGUF files from HuggingFace repositories."""

    @patch("llamafarm_common.model_utils.HfApi")
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

    @patch("llamafarm_common.model_utils.HfApi")
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

    @patch("llamafarm_common.model_utils.HfApi")
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

    @patch("llamafarm_common.model_utils.HfApi")
    def test_list_gguf_files_strips_quantization_suffix(self, mock_hf_api_class):
        """
        Test that list_gguf_files() strips quantization suffix before calling HF API.

        Regression test: HuggingFace API calls with quantization suffixes
        should pass clean model IDs without the colon-separated suffix.
        """
        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "qwen3-1.7b.Q4_K_M.gguf",
            "qwen3-1.7b.Q8_0.gguf",
            "qwen3-1.7b.F16.gguf",
        ]
        mock_hf_api_class.return_value = mock_api

        # Test with quantization suffix
        result = list_gguf_files("unsloth/Qwen3-1.7B-GGUF:Q8_0")

        # Verify HF API was called with CLEAN model ID (no suffix)
        mock_api.list_repo_files.assert_called_once_with(
            repo_id="unsloth/Qwen3-1.7B-GGUF",  # Should NOT have :Q8_0
            token=None,
        )

        # Verify correct files were returned
        assert len(result) == 3
        assert "qwen3-1.7b.Q4_K_M.gguf" in result


class TestSelectGGUFFileWithLogging:
    """Test GGUF file selection with logging."""

    def test_select_with_logging_single_file(self):
        """Test that single file is returned with logging."""
        files = ["model.Q8_0.gguf"]
        result = select_gguf_file_with_logging(files)
        assert result == "model.Q8_0.gguf"

    def test_select_with_logging_default(self):
        """Test that Q4_K_M is selected by default with logging."""
        files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf"]
        result = select_gguf_file_with_logging(files)
        assert result == "model.Q4_K_M.gguf"

    def test_select_with_logging_preferred(self):
        """Test selecting preferred quantization with logging."""
        files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf"]
        result = select_gguf_file_with_logging(files, preferred_quantization="Q8_0")
        assert result == "model.Q8_0.gguf"

    def test_select_with_logging_empty_raises(self):
        """Test that empty list raises ValueError."""
        with pytest.raises(ValueError, match="No GGUF files provided"):
            select_gguf_file_with_logging([])


class TestGetGGUFFilePath:
    """Test getting GGUF file path with download."""

    @patch("llamafarm_common.model_utils.snapshot_download")
    @patch("llamafarm_common.model_utils.HfApi")
    def test_get_gguf_file_path_strips_quantization_suffix(
        self, mock_hf_api_class, mock_snapshot_download
    ):
        """
        Test that get_gguf_file_path() strips quantization suffix before calling HF APIs.

        This test ensures both list_repo_files() and snapshot_download() receive
        clean model IDs without quantization suffixes.
        """
        # Setup mocks
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "qwen3-1.7b.Q4_K_M.gguf",
            "qwen3-1.7b.Q8_0.gguf",
        ]
        mock_hf_api_class.return_value = mock_api

        # Create temp directory and file for snapshot_download
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = os.path.join(tmpdir, "qwen3-1.7b.Q4_K_M.gguf")
            with open(gguf_file, "w") as f:
                f.write("fake gguf")

            mock_snapshot_download.return_value = tmpdir

            # Test with quantization suffix in model ID
            result = get_gguf_file_path("unsloth/Qwen3-1.7B-GGUF:Q4_K_M")

            # Verify list_repo_files was called with CLEAN model ID
            mock_api.list_repo_files.assert_called_once_with(
                repo_id="unsloth/Qwen3-1.7B-GGUF",  # Should NOT have :Q4_K_M
                token=None,
            )

            # Verify snapshot_download was called with CLEAN model ID
            mock_snapshot_download.assert_called_once()
            call_kwargs = mock_snapshot_download.call_args[1]
            assert (
                call_kwargs["repo_id"] == "unsloth/Qwen3-1.7B-GGUF"
            )  # Should NOT have :Q4_K_M

            # Verify the quantization from the suffix was used for file selection
            assert call_kwargs["allow_patterns"] == ["qwen3-1.7b.Q4_K_M.gguf"]

            # Verify correct path was returned
            assert result == gguf_file

    @patch("llamafarm_common.model_utils.snapshot_download")
    @patch("llamafarm_common.model_utils.HfApi")
    def test_get_gguf_file_path_explicit_quantization(
        self, mock_hf_api_class, mock_snapshot_download
    ):
        """Test that explicit preferred_quantization is used when provided."""
        # Setup mocks
        mock_api = Mock()
        mock_api.list_repo_files.return_value = [
            "qwen3-1.7b.Q4_K_M.gguf",
            "qwen3-1.7b.Q8_0.gguf",
        ]
        mock_hf_api_class.return_value = mock_api

        # Create temp directory and file for snapshot_download
        with tempfile.TemporaryDirectory() as tmpdir:
            gguf_file = os.path.join(tmpdir, "qwen3-1.7b.Q8_0.gguf")
            with open(gguf_file, "w") as f:
                f.write("fake gguf")

            mock_snapshot_download.return_value = tmpdir

            # Test with explicit preferred_quantization
            result = get_gguf_file_path(
                "unsloth/Qwen3-1.7B-GGUF", preferred_quantization="Q8_0"
            )

            # Verify Q8_0 was selected
            call_kwargs = mock_snapshot_download.call_args[1]
            assert call_kwargs["allow_patterns"] == ["qwen3-1.7b.Q8_0.gguf"]

            # Verify correct path was returned
            assert result == gguf_file

    @patch("llamafarm_common.model_utils.HfApi")
    def test_get_gguf_file_path_no_files_raises(self, mock_hf_api_class):
        """Test that FileNotFoundError is raised when no GGUF files exist."""
        # Setup mock
        mock_api = Mock()
        mock_api.list_repo_files.return_value = ["README.md", "config.json"]
        mock_hf_api_class.return_value = mock_api

        # Test
        with pytest.raises(FileNotFoundError, match="No GGUF files found"):
            get_gguf_file_path("test/model")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
