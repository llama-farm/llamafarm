#!/usr/bin/env python3
"""
Unit tests for build_addon_wheels.py

Run with: python3 -m pytest tools/test_build_addon_wheels.py -v
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from build_addon_wheels import (
    normalize_package_name,
    extract_package_name_from_wheel,
    parse_uv_lock_packages,
    get_base_exclusion_set,
    parse_pyproject_extras,
)


def test_normalize_package_name():
    """Test package name normalization."""
    test_cases = [
        ("torch", "torch"),
        ("torch-audio", "torch_audio"),
        ("PyTorch", "pytorch"),
        ("scikit-learn", "scikit_learn"),
        ("pillow", "pillow"),
        ("opencv-python-headless", "opencv_python_headless"),
        ("en_core_web_sm", "en_core_web_sm"),
        ("uvicorn[standard]", "uvicorn"),
        ("package[extra1,extra2]", "package"),
    ]

    for input_name, expected in test_cases:
        result = normalize_package_name(input_name)
        assert result == expected, f"normalize_package_name({input_name!r}) = {result!r}, expected {expected!r}"


def test_extract_package_name_from_wheel():
    """Test wheel filename parsing."""
    test_cases = [
        ("torch-2.0.0-cp310-cp310-linux_x86_64.whl", "torch"),
        ("scikit_learn-1.3.0-cp310-cp310-linux_x86_64.whl", "scikit_learn"),
        ("en_core_web_sm-3.8.0-py3-none-any.whl", "en_core_web_sm"),
        ("opencv_python_headless-4.8.0-cp310-cp310-linux_x86_64.whl", "opencv_python_headless"),
        ("kokoro-0.9.4-py3-none-any.whl", "kokoro"),
    ]

    for wheel_name, expected in test_cases:
        result = extract_package_name_from_wheel(wheel_name)
        assert result == expected, f"extract_package_name_from_wheel({wheel_name!r}) = {result!r}, expected {expected!r}"


def test_filtering_logic():
    """Test the filtering logic with sample data."""
    # Sample base packages (would be excluded)
    base_excluded = {
        "torch", "transformers", "numpy", "pillow", "fastapi", "uvicorn", "pydantic"
    }

    # Sample addon keep packages (TTS example)
    addon_keep = {
        "kokoro", "misaki", "spacy", "pydub", "av", "pocket_tts"
    }

    # Sample wheel files
    test_wheels = [
        ("torch-2.0.0-cp310-cp310-linux_x86_64.whl", False, "base package"),
        ("transformers-4.35.0-py3-none-any.whl", False, "base package"),
        ("kokoro-0.9.4-py3-none-any.whl", True, "addon-specific"),
        ("misaki-0.9.0-py3-none-any.whl", True, "addon-specific"),
        ("spacy-3.8.11-py3-none-any.whl", True, "addon-specific"),
        ("some_other_package-1.0.0-py3-none-any.whl", True, "not in base"),
    ]

    for wheel_name, should_keep, reason in test_wheels:
        package_name = extract_package_name_from_wheel(wheel_name)

        # Apply filtering logic
        if package_name in addon_keep:
            keep = True
        elif package_name in base_excluded:
            keep = False
        else:
            keep = True

        assert keep == should_keep, f"Filtering logic failed for {wheel_name}: got {keep}, expected {should_keep} ({reason})"


def test_parse_uv_lock_packages():
    """Test parsing uv.lock with dependency graph filtering (real dict format)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Mock pyproject.toml with base and extras dependencies
        pyproject_content = """[project]
dependencies = [
    "fastapi>=0.104.0",
    "transformers>=4.35.0",
    "numpy>=1.24.0"
]

[project.optional-dependencies]
gpu = ["torch>=2.0.0"]
tts = ["kokoro>=0.9.4", "spacy>=3.8.0"]
speech = ["faster-whisper>=1.0.0"]

[dependency-groups]
dev = ["pytest>=7.0.0", "ruff>=0.14.0"]
"""

        # Mock uv.lock using real dict format for dependencies
        uv_lock_content = """version = 1

[[package]]
name = "fastapi"
version = "0.104.0"
dependencies = [
    { name = "pydantic" },
    { name = "starlette" },
]

[[package]]
name = "pydantic"
version = "2.0.0"
dependencies = [
    { name = "typing-extensions" },
]

[[package]]
name = "starlette"
version = "0.27.0"
dependencies = []

[[package]]
name = "typing-extensions"
version = "4.8.0"
dependencies = []

[[package]]
name = "transformers"
version = "4.35.0"
dependencies = [
    { name = "numpy" },
    { name = "huggingface-hub" },
]

[[package]]
name = "numpy"
version = "1.24.0"
dependencies = []

[[package]]
name = "huggingface-hub"
version = "0.19.0"
dependencies = []

[[package]]
name = "torch"
version = "2.0.0"
dependencies = [
    { name = "nvidia-cuda-runtime-cu12", marker = "platform_machine == 'x86_64' and sys_platform == 'linux'" },
]

[[package]]
name = "nvidia-cuda-runtime-cu12"
version = "12.1.0"
dependencies = []

[[package]]
name = "kokoro"
version = "0.9.4"
dependencies = [
    { name = "misaki" },
]

[[package]]
name = "misaki"
version = "0.9.0"
dependencies = []

[[package]]
name = "spacy"
version = "3.8.0"
dependencies = [
    { name = "cymem" },
    { name = "preshed" },
]

[[package]]
name = "cymem"
version = "2.0.8"
dependencies = []

[[package]]
name = "preshed"
version = "3.0.9"
dependencies = []

[[package]]
name = "faster-whisper"
version = "1.0.0"
dependencies = [
    { name = "openai-whisper" },
]

[[package]]
name = "openai-whisper"
version = "20231117"
dependencies = []

[[package]]
name = "pytest"
version = "7.0.0"
dependencies = []

[[package]]
name = "ruff"
version = "0.14.0"
dependencies = []
"""

        pyproject_path = tmpdir / "pyproject.toml"
        uv_lock_path = tmpdir / "uv.lock"

        pyproject_path.write_text(pyproject_content)
        uv_lock_path.write_text(uv_lock_content)

        # Test that only base-reachable packages are included
        result = parse_uv_lock_packages(uv_lock_path, pyproject_path)

        # Should include base dependencies and their transitive deps
        expected_base_reachable = {
            "fastapi", "pydantic", "starlette", "typing_extensions",  # fastapi tree
            "transformers", "huggingface_hub",  # transformers tree
            "numpy"  # numpy (base dep)
        }

        # Should NOT include extras-only packages
        extras_only_packages = {
            "torch", "nvidia_cuda_runtime_cu12",  # gpu extra
            "kokoro", "misaki",  # tts extra
            "spacy", "cymem", "preshed",  # tts extra transitive
            "faster_whisper", "openai_whisper",  # speech extra
            "pytest", "ruff"  # dev group
        }

        # Verify base packages are included
        for pkg in expected_base_reachable:
            assert pkg in result, f"Base-reachable package {pkg} should be included"

        # Verify extras-only packages are NOT included
        for pkg in extras_only_packages:
            assert pkg not in result, f"Extras-only package {pkg} should NOT be included"


def test_parse_pyproject_extras():
    """Test that only GPU extra is parsed (not tts, speech, dev)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        pyproject_content = """[project]
dependencies = ["fastapi>=0.104.0"]

[project.optional-dependencies]
gpu = ["torch>=2.0.0", "torchvision>=0.15.0"]
tts = ["kokoro>=0.9.4", "spacy>=3.8.0"]
tts-mlx = ["mlx-audio>=0.3.1"]
speech = ["faster-whisper>=1.0.0"]

[dependency-groups]
dev = ["pytest>=7.0.0", "ruff>=0.14.0"]
"""

        pyproject_path = tmpdir / "pyproject.toml"
        pyproject_path.write_text(pyproject_content)

        result = parse_pyproject_extras(pyproject_path)

        # Should only include GPU extra packages
        expected = {"torch", "torchvision"}
        assert result == expected, f"Expected {expected}, got {result}"


def test_get_base_exclusion_set_empty_fails():
    """Test that empty exclusion set raises error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create empty/minimal files that would result in empty exclusion set
        pyproject_content = """[project]
dependencies = []
[project.optional-dependencies]
gpu = []
"""
        uv_lock_content = """version = 1
"""

        pyproject_path = tmpdir / "pyproject.toml"
        uv_lock_path = tmpdir / "uv.lock"
        pyproject_path.write_text(pyproject_content)
        uv_lock_path.write_text(uv_lock_content)

        def mock_get_base_exclusion_set(no_exclude=False):
            if no_exclude:
                return set()

            uv_packages = parse_uv_lock_packages(uv_lock_path, pyproject_path)
            extra_packages = parse_pyproject_extras(pyproject_path)
            all_excluded = uv_packages | extra_packages

            if not all_excluded:
                raise RuntimeError(
                    "Base exclusion set is empty - this would include all packages and reproduce "
                    "the original 4GB addon problem. Check that uv.lock and pyproject.toml are valid, "
                    "or use --no-exclude if this is intentional."
                )

            return all_excluded

        with patch("test_build_addon_wheels.get_base_exclusion_set", side_effect=mock_get_base_exclusion_set):
            # Should raise error when exclusion set is empty
            with pytest.raises(RuntimeError, match="Base exclusion set is empty"):
                get_base_exclusion_set(no_exclude=False)

            # Should not raise error with no_exclude=True
            result = get_base_exclusion_set(no_exclude=True)
            assert result == set()


def test_get_base_exclusion_set_missing_files():
    """Test that missing files raise appropriate errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        pyproject_path = tmpdir / "pyproject.toml"
        uv_lock_path = tmpdir / "uv.lock"

        pyproject_path.write_text("[project]\ndependencies = []")
        # Don't create uv.lock

        def mock_get_base_exclusion_set(no_exclude=False):
            if no_exclude:
                return set()

            if not uv_lock_path.exists():
                raise FileNotFoundError(
                    f"uv.lock file not found at {uv_lock_path}. "
                    f"Run 'uv sync' to generate it or use --no-exclude to disable filtering."
                )

            return set()

        with patch("test_build_addon_wheels.get_base_exclusion_set", side_effect=mock_get_base_exclusion_set):
            with pytest.raises(FileNotFoundError, match="uv.lock file not found"):
                get_base_exclusion_set(no_exclude=False)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
