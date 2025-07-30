"""
Pytest configuration and shared fixtures for config tests.
"""

import sys
from pathlib import Path
import pytest

# Add the config module to Python path for testing
config_dir = Path(__file__).parent.parent
sys.path.insert(0, str(config_dir))


@pytest.fixture(scope="session")
def config_module_path():
    """Fixture providing path to the config module."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_configs_dir():
    """Fixture providing path to sample config files."""
    return Path(__file__).parent


@pytest.fixture
def temp_config_file():
    """Fixture for creating temporary config files in tests."""
    import tempfile
    import os

    temp_files = []

    def _create_temp_config(content: str, suffix: str = ".yaml"):
        """Create a temporary config file with given content."""
        fd, path = tempfile.mkstemp(suffix=suffix, text=True)
        with os.fdopen(fd, 'w') as f:
            f.write(content)
        temp_files.append(path)
        return path

    yield _create_temp_config

    # Cleanup
    for path in temp_files:
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.fixture(autouse=True)
def reset_imports():
    """Reset import cache to ensure clean imports in each test."""
    import sys

    # Remove any previously imported config modules
    modules_to_remove = [name for name in sys.modules.keys()
                        if name.startswith(('loader', 'config_types', 'config.'))]

    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]

    yield

    # Clean up again after test
    modules_to_remove = [name for name in sys.modules.keys()
                        if name.startswith(('loader', 'config_types', 'config.'))]

    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]