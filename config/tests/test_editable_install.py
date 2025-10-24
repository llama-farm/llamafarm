"""
Test that the config package can be imported when installed as an editable dependency.

This test ensures that the setuptools configuration in pyproject.toml correctly
handles editable installs from other packages in the monorepo (server, rag, etc.).

Context: Previously used hatchling which had issues with editable installs creating
empty .pth files. Switched to setuptools which properly handles the package-dir
configuration needed for this monorepo structure.
"""

import sys
from pathlib import Path


def test_config_package_is_importable():
    """Test that config package can be imported."""
    try:
        import config

        assert config is not None
    except ImportError as e:
        raise AssertionError(
            f"Failed to import config package: {e}\n"
            "This usually means the editable install is not configured correctly."
        ) from e


def test_config_datamodel_is_importable():
    """Test that config.datamodel module can be imported."""
    try:
        from config.datamodel import LlamaFarmConfig, Model

        assert LlamaFarmConfig is not None
        assert Model is not None
    except ImportError as e:
        raise AssertionError(
            f"Failed to import config.datamodel: {e}\n"
            "This usually means the editable install is not configured correctly."
        ) from e


def test_editable_install_path_is_correct():
    """Test that the config module is loaded from the expected location."""
    import config

    config_file = Path(config.__file__)
    expected_parent = Path(__file__).parent.parent  # config/tests/../ = config/

    assert config_file.parent == expected_parent, (
        f"Config module loaded from unexpected location.\n"
        f"Expected: {expected_parent}\n"
        f"Got: {config_file.parent}\n"
        f"This indicates the editable install may not be working correctly."
    )


def test_config_in_sys_path():
    """Test that the config package parent directory is in sys.path."""
    import config

    config_parent = str(Path(config.__file__).parent.parent)

    assert any(config_parent in path for path in sys.path), (
        f"Config parent directory not found in sys.path.\n"
        f"Looking for: {config_parent}\n"
        f"sys.path: {sys.path}\n"
        f"This indicates the .pth file may not be configured correctly."
    )


if __name__ == "__main__":
    # Run tests manually for quick verification
    test_config_package_is_importable()
    print("✅ config package is importable")

    test_config_datamodel_is_importable()
    print("✅ config.datamodel is importable")

    test_editable_install_path_is_correct()
    print("✅ Editable install path is correct")

    test_config_in_sys_path()
    print("✅ Config parent directory is in sys.path")

    print("\n🎉 All editable install tests passed!")
