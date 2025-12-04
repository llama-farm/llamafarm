"""Tests for disk space service."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from server.services.disk_space_service import (
    DiskSpaceInfo,
    DiskSpaceService,
    ValidationResult,
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
    """Test checking disk space with invalid path that causes psutil to fail."""
    invalid_path = Path("/nonexistent/path/that/does/not/exist")

    # Mock psutil.disk_usage to raise OSError since the actual method
    # walks up parent directories and would succeed for non-existent paths
    with (
        patch(
            "server.services.disk_space_service.psutil.disk_usage",
            side_effect=OSError("Permission denied"),
        ),
        pytest.raises(OSError),
    ):
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
    with patch.object(DiskSpaceService, "get_cache_directory", return_value=tmp_path):
        cache_info, system_info = DiskSpaceService.check_both_disks()

        assert isinstance(cache_info, DiskSpaceInfo)
        assert isinstance(system_info, DiskSpaceInfo)
        assert cache_info.path == str(tmp_path.resolve())


@patch("huggingface_hub.HfApi")
def test_get_model_size_success(mock_hf_api):
    """Test getting model size from HuggingFace API."""
    mock_api = MagicMock()
    mock_hf_api.return_value = mock_api

    # Mock model info with siblings (primary method)
    mock_model_info = MagicMock()
    mock_sibling1 = MagicMock()
    mock_sibling1.size = 500000000  # 500MB
    mock_sibling2 = MagicMock()
    mock_sibling2.size = 500000000  # 500MB
    mock_model_info.siblings = [mock_sibling1, mock_sibling2]
    mock_api.model_info.return_value = mock_model_info

    size = DiskSpaceService.get_model_size("test/model")

    assert size == 1000000000  # 1GB total (2 * 500MB)


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
    with (
        patch.object(DiskSpaceService, "check_both_disks") as mock_check,
        patch.object(
            DiskSpaceService, "get_model_size", return_value=1000000000
        ) as mock_size,
    ):
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
    with (
        patch.object(DiskSpaceService, "check_both_disks") as mock_check,
        patch.object(DiskSpaceService, "get_model_size", return_value=1000000000),
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
        assert "below the 10% threshold" in result.message


def test_validate_space_for_download_critical_space(tmp_path):
    """Test validation when space is critically low."""
    with (
        patch.object(DiskSpaceService, "check_both_disks") as mock_check,
        patch.object(DiskSpaceService, "get_model_size", return_value=1000000000),
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
    with (
        patch.object(DiskSpaceService, "check_both_disks") as mock_check,
        patch.object(DiskSpaceService, "get_model_size", return_value=10000000000),
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
    with (
        patch.object(DiskSpaceService, "check_both_disks") as mock_check,
        patch.object(DiskSpaceService, "get_model_size", return_value=None),
    ):
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
        assert result.required_bytes == 0
        assert "Sufficient space available" in result.message


def test_validate_space_for_download_size_unavailable_low_space(tmp_path):
    """Test validation when model size cannot be determined and space is low."""
    with (
        patch.object(DiskSpaceService, "check_both_disks") as mock_check,
        patch.object(DiskSpaceService, "get_model_size", return_value=None),
    ):
        # Mock disk info with low space (< 20%)
        mock_cache_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=85000000000,
            free_bytes=15000000000,  # 15GB free (15%)
            path=str(tmp_path),
            percent_free=15.0,
        )
        mock_system_info = DiskSpaceInfo(
            total_bytes=100000000000,
            used_bytes=85000000000,
            free_bytes=15000000000,  # 15GB free (15%)
            path="/",
            percent_free=15.0,
        )
        mock_check.return_value = (mock_cache_info, mock_system_info)

        result = DiskSpaceService.validate_space_for_download("test/model")

        assert isinstance(result, ValidationResult)
        assert result.can_download is True
        assert result.warning is True
        assert result.required_bytes == 0
        assert (
            "low disk space" in result.message.lower()
            or "model size could not be determined" in result.message.lower()
        )
