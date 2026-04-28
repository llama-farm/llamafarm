"""Offline-mode tests for llamafarm_llama._binary.

These tests verify that:
  - `get_lib_path()` raises FileNotFoundError (not a network call) when
    `LLAMAFARM_OFFLINE=1` is set and neither the bundled nor cached binary
    exists.
  - The error message names both tried paths and references
    `lf runtime binary pull`.
  - `get_binary_info()` reports the offline flag in its output dict.
  - Online-mode behavior is preserved when the env var is absent.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from llamafarm_llama import _binary


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
    yield


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache dir at a tmp directory and make bundled absent."""
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("LLAMAFARM_CACHE_DIR", str(cache_root))
    return cache_root


class TestIsOfflineInline:
    def test_unset(self, monkeypatch):
        monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
        assert _binary._is_offline() is False

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", "True"])
    def test_truthy(self, monkeypatch, value):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", value)
        assert _binary._is_offline() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", ""])
    def test_falsy(self, monkeypatch, value):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", value)
        assert _binary._is_offline() is False


class TestGetLibPathOffline:
    def test_offline_no_bundled_no_cache_raises_without_download(
        self, isolated_cache, monkeypatch
    ):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        # Force bundled path to look absent.
        with patch.object(Path, "exists", lambda self: False), patch.object(
            _binary, "download_binary"
        ) as mock_dl:
            with pytest.raises(FileNotFoundError) as excinfo:
                _binary.get_lib_path()
            mock_dl.assert_not_called()

        msg = str(excinfo.value)
        assert "offline mode" in msg
        assert "lf runtime binary pull" in msg

    def test_offline_error_lists_tried_paths(self, isolated_cache, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        with patch.object(Path, "exists", lambda self: False), patch.object(
            _binary, "download_binary"
        ):
            with pytest.raises(FileNotFoundError) as excinfo:
                _binary.get_lib_path()

        msg = str(excinfo.value)
        # Should mention "Tried:" at least twice (bundled + cached).
        assert msg.count("Tried:") >= 2

    def test_offline_error_includes_platform(self, isolated_cache, monkeypatch):
        import platform as platform_mod

        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        with patch.object(Path, "exists", lambda self: False), patch.object(
            _binary, "download_binary"
        ):
            with pytest.raises(FileNotFoundError) as excinfo:
                _binary.get_lib_path()

        msg = str(excinfo.value)
        sys_name = platform_mod.system().lower()
        # Error should name the current platform somewhere.
        assert sys_name in msg or "arm64" in msg or "amd64" in msg

    def test_online_mode_still_downloads_when_missing(
        self, isolated_cache, monkeypatch
    ):
        # Ensure offline is NOT set.
        monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)

        with patch.object(Path, "exists", lambda self: False), patch.object(
            _binary, "download_binary"
        ) as mock_dl:
            mock_dl.return_value = Path("/fake/libllama.so")
            result = _binary.get_lib_path()
            mock_dl.assert_called_once()
            assert result == Path("/fake/libllama.so")

    def test_offline_cached_hit_returns_without_error(
        self, tmp_path, monkeypatch
    ):
        """When the cached binary exists, offline mode should succeed."""
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        cache_root = tmp_path / "cache"
        monkeypatch.setenv("LLAMAFARM_CACHE_DIR", str(cache_root))

        # Create a fake cached library file at the expected path.
        version_dir = cache_root / _binary.LLAMA_CPP_VERSION
        version_dir.mkdir(parents=True)
        lib_name = _binary._get_lib_name()
        cached_path = version_dir / lib_name
        cached_path.write_bytes(b"stub")

        missing_bundled = tmp_path / "_bundled" / "linux-x86_64" / lib_name
        monkeypatch.setattr(_binary, "_bundled_binary_path", lambda: missing_bundled)

        with patch.object(_binary, "download_binary") as mock_dl:
            result = _binary.get_lib_path()
            mock_dl.assert_not_called()

        assert result == cached_path


class TestGetBinaryInfoOffline:
    def test_offline_flag_in_info(self, isolated_cache, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        info = _binary.get_binary_info()
        assert info["offline"] is True

    def test_online_flag_in_info(self, isolated_cache, monkeypatch):
        monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
        info = _binary.get_binary_info()
        assert info["offline"] is False
