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


class TestQuantizationSuffixParsing:
    """
    Test that utility functions correctly parse quantization suffixes from model IDs
    before calling HuggingFace APIs.
    
    This test class specifically catches the bug where model IDs like
    'unsloth/Qwen3-1.7B-GGUF:Q4_K_M' were passed directly to HuggingFace APIs,
    which don't allow colons in repo IDs, causing validation errors.
    
    Regression test for: HuggingFace API calls with quantization suffixes
    """

    @patch("utils.model_format.HfApi")
    def test_detect_model_format_strips_quantization_suffix(self, mock_hf_api_class):
        """
        Test that detect_model_format() strips quantization suffix before calling HF API.
        
        This ensures 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M' is passed to HF API as
        'unsloth/Qwen3-1.7B-GGUF' (without the ':Q4_K_M' suffix).
        """
        from utils.model_format import detect_model_format, clear_format_cache
        
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
            token=None
        )
        
        # Verify correct format was detected
        assert result == "gguf"

    @patch("utils.model_format.HfApi")
    def test_list_gguf_files_strips_quantization_suffix(self, mock_hf_api_class):
        """
        Test that list_gguf_files() strips quantization suffix before calling HF API.
        """
        from utils.model_format import list_gguf_files
        
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
            token=None
        )
        
        # Verify correct files were returned
        assert len(result) == 3
        assert "qwen3-1.7b.Q4_K_M.gguf" in result

    @patch("utils.model_format.snapshot_download")
    @patch("utils.model_format.HfApi")
    def test_get_gguf_file_path_strips_quantization_suffix(
        self, mock_hf_api_class, mock_snapshot_download
    ):
        """
        Test that get_gguf_file_path() strips quantization suffix before calling HF APIs.
        
        This test ensures both list_repo_files() and snapshot_download() receive
        clean model IDs without quantization suffixes.
        """
        from utils.model_format import get_gguf_file_path
        import tempfile
        import os
        
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
            with open(gguf_file, 'w') as f:
                f.write("fake gguf")
            
            mock_snapshot_download.return_value = tmpdir

            # Test with quantization suffix in model ID
            result = get_gguf_file_path("unsloth/Qwen3-1.7B-GGUF:Q4_K_M")

            # Verify list_repo_files was called with CLEAN model ID
            mock_api.list_repo_files.assert_called_once_with(
                repo_id="unsloth/Qwen3-1.7B-GGUF",  # Should NOT have :Q4_K_M
                token=None
            )
            
            # Verify snapshot_download was called with CLEAN model ID
            mock_snapshot_download.assert_called_once()
            call_kwargs = mock_snapshot_download.call_args[1]
            assert call_kwargs["repo_id"] == "unsloth/Qwen3-1.7B-GGUF"  # Should NOT have :Q4_K_M
            
            # Verify the quantization from the suffix was used for file selection
            assert call_kwargs["allow_patterns"] == ["qwen3-1.7b.Q4_K_M.gguf"]
            
            # Verify correct path was returned
            assert result == gguf_file

    @patch("utils.model_format.HfApi")
    def test_caching_with_quantization_suffix(self, mock_hf_api_class):
        """
        Test that format detection cache works correctly with quantization suffixes.
        
        Both 'model:Q4_K_M' and 'model:Q8_0' should use the same cached result
        since they're the same base model.
        """
        from utils.model_format import detect_model_format, clear_format_cache
        
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
