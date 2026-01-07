"""Tests for binary download and management."""

import os
import platform
from pathlib import Path
from unittest.mock import patch

import pytest


class TestPlatformDetection:
    """Test platform and backend detection."""

    def test_get_platform_key_returns_tuple(self):
        """Platform key should be a tuple of (system, machine, backend)."""
        from llamafarm_llama._binary import get_platform_key

        key = get_platform_key()
        assert isinstance(key, tuple)
        assert len(key) == 3

    def test_get_platform_key_valid_system(self):
        """System should be linux, darwin, or win32."""
        from llamafarm_llama._binary import get_platform_key

        key = get_platform_key()
        assert key[0] in ("linux", "darwin", "win32")

    def test_get_platform_key_valid_machine(self):
        """Machine should be x86_64, amd64, or arm64."""
        from llamafarm_llama._binary import get_platform_key

        key = get_platform_key()
        assert key[1] in ("x86_64", "amd64", "arm64")

    def test_backend_override_via_env(self):
        """Backend can be overridden via LLAMAFARM_BACKEND."""
        from llamafarm_llama._binary import get_platform_key

        with patch.dict(os.environ, {"LLAMAFARM_BACKEND": "cpu"}):
            key = get_platform_key()
            assert key[2] == "cpu"

    def test_backend_override_via_param(self):
        """Backend can be overridden via parameter."""
        from llamafarm_llama._binary import get_platform_key

        key = get_platform_key(backend_override="vulkan")
        assert key[2] == "vulkan"

    @pytest.mark.skipif(
        platform.system() != "Darwin" or platform.machine() != "arm64",
        reason="Metal only on macOS ARM",
    )
    def test_macos_arm_detects_metal(self):
        """macOS ARM should detect Metal backend."""
        from llamafarm_llama._binary import get_platform_key

        # Clear any override
        with patch.dict(os.environ, {}, clear=True):
            key = get_platform_key()
            assert key[2] == "metal"


class TestCacheDir:
    """Test cache directory handling."""

    def test_cache_dir_env_override(self):
        """LLAMAFARM_CACHE_DIR should override default."""
        from llamafarm_llama._binary import _get_cache_dir

        with patch.dict(os.environ, {"LLAMAFARM_CACHE_DIR": "/custom/cache"}):
            cache_dir = _get_cache_dir()
            assert cache_dir == Path("/custom/cache")

    def test_cache_dir_platform_specific(self):
        """Cache dir should be platform-specific by default."""
        from llamafarm_llama._binary import _get_cache_dir

        with patch.dict(os.environ, {}, clear=True):
            cache_dir = _get_cache_dir()
            assert cache_dir.is_absolute()
            assert "llamafarm-llama" in str(cache_dir)


class TestLibName:
    """Test library name detection."""

    def test_linux_lib_name(self):
        """Linux should use .so extension."""
        from llamafarm_llama._binary import _get_lib_name

        with patch("platform.system", return_value="Linux"):
            assert _get_lib_name() == "libllama.so"

    def test_macos_lib_name(self):
        """macOS should use .dylib extension."""
        from llamafarm_llama._binary import _get_lib_name

        with patch("platform.system", return_value="Darwin"):
            assert _get_lib_name() == "libllama.dylib"

    def test_windows_lib_name(self):
        """Windows should use .dll extension."""
        from llamafarm_llama._binary import _get_lib_name

        with patch("platform.system", return_value="Windows"):
            assert _get_lib_name() == "llama.dll"


class TestBinaryInfo:
    """Test binary info retrieval."""

    def test_get_binary_info_structure(self):
        """Binary info should have expected keys."""
        from llamafarm_llama._binary import get_binary_info

        info = get_binary_info()
        assert "version" in info
        assert "platform_key" in info
        assert "lib_path" in info
        assert "lib_name" in info
        assert "source" in info
        assert "cache_dir" in info

    def test_get_binary_info_version(self):
        """Version should match LLAMA_CPP_VERSION."""
        from llamafarm_llama._binary import get_binary_info, LLAMA_CPP_VERSION

        info = get_binary_info()
        assert info["version"] == LLAMA_CPP_VERSION


class TestBinaryManifest:
    """Test binary manifest completeness."""

    def test_manifest_has_linux_cpu(self):
        """Manifest should have Linux CPU build."""
        from llamafarm_llama._binary import BINARY_MANIFEST

        assert ("linux", "x86_64", "cpu") in BINARY_MANIFEST

    def test_manifest_has_macos_metal(self):
        """Manifest should have macOS Metal build."""
        from llamafarm_llama._binary import BINARY_MANIFEST

        assert ("darwin", "arm64", "metal") in BINARY_MANIFEST

    def test_manifest_has_windows_cpu(self):
        """Manifest should have Windows CPU build."""
        from llamafarm_llama._binary import BINARY_MANIFEST

        assert ("win32", "amd64", "cpu") in BINARY_MANIFEST

    def test_manifest_entries_have_artifact(self):
        """All manifest entries should have artifact key."""
        from llamafarm_llama._binary import BINARY_MANIFEST

        for key, value in BINARY_MANIFEST.items():
            assert "artifact" in value, f"Missing artifact for {key}"
            assert "lib" in value, f"Missing lib for {key}"
