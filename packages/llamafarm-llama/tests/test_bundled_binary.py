"""Tests for bundled binary resolution."""

from pathlib import Path
from unittest.mock import Mock

from llamafarm_llama import _binary


def test_get_lib_path_returns_bundled_binary_without_download(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    bundled_path = tmp_path / "bundle" / "linux-x86_64" / "libllama.so"
    bundled_path.parent.mkdir(parents=True, exist_ok=True)
    bundled_path.write_bytes(b"bundled")

    monkeypatch.setenv("LLAMAFARM_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(_binary, "_bundled_binary_path", lambda: bundled_path)
    download_mock = Mock()
    monkeypatch.setattr(_binary, "download_binary", download_mock)

    result = _binary.get_lib_path()

    assert result == bundled_path
    download_mock.assert_not_called()


def test_get_lib_path_falls_back_to_download_when_bundled_absent(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    missing_bundled = tmp_path / "bundle" / "linux-x86_64" / "libllama.so"
    expected = Path("/tmp/downloaded/libllama.so")
    download_mock = Mock(return_value=expected)

    monkeypatch.setenv("LLAMAFARM_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(_binary, "_bundled_binary_path", lambda: missing_bundled)
    monkeypatch.setattr(_binary, "download_binary", download_mock)

    result = _binary.get_lib_path()

    assert result == expected
    download_mock.assert_called_once_with(cache_root / _binary.LLAMA_CPP_VERSION)
