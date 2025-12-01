"""Tests for disk space service."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from server.services.disk_space_service import (
    CRITICAL_THRESHOLD_BYTES,
    DiskSpaceInfo,
    DiskSpaceService,
    ValidationResult,
    WARNING_THRESHOLD_PERCENT,
)


def test_check_disk_space(tmp_path):
    """Test checking disk space at a given path."""
    info = DiskSpaceService.check_disk_space(tmp_path)

    assert isinstance(info, DiskSpaceInfo)
    assert info.total_bytes > 0
    assert info.used_bytes >= 0
    assert info.free_bytes >= 0
    assert info.path == str(tmp_path.resolve())
    assert 0 <= info.percent_free <= 100


def test_check_disk_space_invalid_path():
    """Test checking disk space with invalid path."""
    invalid_path = Path("/nonexistent/path/that/does/not/exist")

    with pytest.raises(OSError):
        DiskSpaceService.check_disk_space(invalid_path)


def test_get_cache_directory():
    """Test getting HuggingFace cache directory."""
    cache_dir = DiskSpaceService.get_cache_directory()

    assert isinstance(cache_dir, Path)
    # Should be a valid path structure
    assert len(str(cache_dir)) > 0


def test_get_system_disk():
    """Test getting system disk root."""
    system_disk = DiskSpaceService.get_system_disk()

    assert isinstance(system_disk, Path)
    if os.name == "nt":  # Windows
        assert str(system_disk) == "C:\\"
    else:  # Unix-like
        assert str(system_disk) == "/"


def test_check_both_disks(tmp_path):
    """Test checking both cache and system disk."""
    with patch.object(
        DiskSpaceService, "get_cache_directory", return_value=tmp_path
    ):
        cache_info, system_info = DiskSpaceService.check_both_disks()

        assert isinstance(cache_info, DiskSpaceInfo)
        assert isinstance(system_info, DiskSpaceInfo)
        assert cache_info.path == str(tmp_path.resolve())


@patch("huggingface_hub.HfApi")
def test_get_model_size_success(mock_hf_api):
    """Test getting model size from HuggingFace API."""
    # Mock HfApi
    mock_api = MagicMock()
    mock_hf_api.return_value = mock_api

    # Mock file listing (non-GGUF model)
    mock_api.list_repo_files.return_value = ["model.safetensors", "config.json"]

    # Mock model info with safetensors
    mock_model_info = MagicMock()
    mock_safetensor = MagicMock()
    mock_safetensor.size = 500000000  # 500MB
    mock_model_info.safetensors = [mock_safetensor, mock_safetensor]
    mock_api.model_info.return_value = mock_model_info

    size = DiskSpaceService.get_model_size("test/model")

    # Should return sum of safetensors sizes
    assert size == 1000000000  # 1GB total (2 * 500MB)


@patch("huggingface_hub.HfApi")
def test_get_model_size_gguf(mock_hf_api):
    """Test getting model size for GGUF model."""
    # Mock HfApi
    mock_api = MagicMock()
    mock_hf_api.return_value = mock_api

    # Mock GGUF file listing
    mock_api.list_repo_files.return_value = [
        "model.Q4_K_M.gguf",
        "model.Q8_0.gguf",
        "config.json",
    ]

    # Mock model info with files attribute (for GGUF, size may not be easily available)
    # In practice, GGUF size estimation may return None, which is acceptable
    mock_model_info = MagicMock()
    mock_model_info.safetensors = None
    mock_file = MagicMock()
    mock_file.size = 2000000000  # 2GB
    mock_model_info.files = [mock_file]
    mock_api.model_info.return_value = mock_model_info

    size = DiskSpaceService.get_model_size("test/model-gguf")

    # May return size from files or None (both are acceptable)
    assert size is None or size == 2000000000


@patch("huggingface_hub.HfApi")
def test_get_model_size_not_found(mock_hf_api):
    """Test getting model size when model not found."""
    # Mock HfApi to raise exception
    mock_api = MagicMock()
    mock_hf_api.return_value = mock_api
    mock_api.list_repo_files.side_effect = Exception("Model not found")

    size = DiskSpaceService.get_model_size("nonexistent/model")

    assert size is None


def test_validate_space_for_download_sufficient_space(tmp_path):
    """Test validation when sufficient space is available."""
    with patch.object(
        DiskSpaceService, "check_both_disks"
    ) as mock_check, patch.object(
        DiskSpaceService, "get_model_size", return_value=1000000000
    ) as mock_size:
        # Mock disk info with plenty of space
        mock_cache_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=50000000000,
            free_bytes=50000000000,  # 50GB free
            path=str(tmp_path),
            percent_free=50.0,
        )
        mock_system_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=50000000000,
            free_bytes=50000000000,  # 50GB free
            path="/",
            percent_free=50.0,
        )
        mock_check.return_value = (mock_cache_info, mock_system_info)

        result = DiskSpaceService.validate_space_for_download("test/model")

        assert isinstance(result, ValidationResult)
        assert result.can_download is True
        assert result.warning is False
        assert result.required_bytes == 1000000000
        mock_size.assert_called_once_with("test/model")


def test_validate_space_for_download_low_space_warning(tmp_path):
    """Test validation when space is low (warning threshold)."""
    with patch.object(
        DiskSpaceService, "check_both_disks"
    ) as mock_check, patch.object(
        DiskSpaceService, "get_model_size", return_value=1000000000
    ):
        # Mock disk info with low space (< 10%)
        mock_cache_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=95000000000,
            free_bytes=5000000000,  # 5GB free (5%)
            path=str(tmp_path),
            percent_free=5.0,
        )
        mock_system_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=95000000000,
            free_bytes=5000000000,  # 5GB free (5%)
            path="/",
            percent_free=5.0,
        )
        mock_check.return_value = (mock_cache_info, mock_system_info)

        result = DiskSpaceService.validate_space_for_download("test/model")

        assert isinstance(result, ValidationResult)
        assert result.can_download is True
        assert result.warning is True
        assert "Nearing disk space max" in result.message


def test_validate_space_for_download_critical_space(tmp_path):
    """Test validation when space is critically low."""
    with patch.object(
        DiskSpaceService, "check_both_disks"
    ) as mock_check, patch.object(
        DiskSpaceService, "get_model_size", return_value=1000000000
    ):
        # Mock disk info with critical space (< 100MB)
        mock_cache_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=99900000000,
            free_bytes=50000000,  # 50MB free (< 100MB threshold)
            path=str(tmp_path),
            percent_free=0.05,
        )
        mock_system_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=99900000000,
            free_bytes=50000000,  # 50MB free
            path="/",
            percent_free=0.05,
        )
        mock_check.return_value = (mock_cache_info, mock_system_info)

        result = DiskSpaceService.validate_space_for_download("test/model")

        assert isinstance(result, ValidationResult)
        assert result.can_download is False
        assert "Insufficient disk space" in result.message


def test_validate_space_for_download_model_too_large(tmp_path):
    """Test validation when model is larger than available space."""
    with patch.object(
        DiskSpaceService, "check_both_disks"
    ) as mock_check, patch.object(
        DiskSpaceService, "get_model_size", return_value=10000000000
    ):
        # Mock disk info with less space than model size
        mock_cache_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=95000000000,
            free_bytes=5000000000,  # 5GB free
            path=str(tmp_path),
            percent_free=5.0,
        )
        mock_system_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=95000000000,
            free_bytes=5000000000,  # 5GB free
            path="/",
            percent_free=5.0,
        )
        mock_check.return_value = (mock_cache_info, mock_system_info)

        result = DiskSpaceService.validate_space_for_download("test/model")

        assert isinstance(result, ValidationResult)
        assert result.can_download is False
        assert "Insufficient disk space" in result.message


def test_validate_space_for_download_check_fails(tmp_path):
    """Test validation when disk space check fails (graceful degradation)."""
    with patch.object(
        DiskSpaceService, "check_both_disks", side_effect=OSError("Permission denied")
    ):
        result = DiskSpaceService.validate_space_for_download("test/model")

        assert isinstance(result, ValidationResult)
        # Should allow download with warning
        assert result.can_download is True
        assert "Disk space check unavailable" in result.message


def test_validate_space_for_download_size_unavailable(tmp_path):
    """Test validation when model size cannot be determined."""
    with patch.object(
        DiskSpaceService, "check_both_disks"
    ) as mock_check, patch.object(
        DiskSpaceService, "get_model_size", return_value=None
    ):
        # Mock disk info
        mock_cache_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=50000000000,
            free_bytes=50000000000,  # 50GB free
            path=str(tmp_path),
            percent_free=50.0,
        )
        mock_system_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=50000000000,
            free_bytes=50000000000,  # 50GB free
            path="/",
            percent_free=50.0,
        )
        mock_check.return_value = (mock_cache_info, mock_system_info)

        result = DiskSpaceService.validate_space_for_download("test/model")

        assert isinstance(result, ValidationResult)
        assert result.can_download is True
        assert result.required_bytes == 0
        assert "Model size unavailable" in result.message

