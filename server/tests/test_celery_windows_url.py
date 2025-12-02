"""
Test Celery file:// URL construction on Windows and Unix platforms for server.

This ensures that Windows paths with backslashes are properly converted
to valid file:// URLs that Celery can parse.
"""

import os
import sys
from unittest.mock import patch

import pytest


def test_windows_file_url_construction():
    """Test that Windows paths are converted to proper file:// URLs."""
    # Simulate Windows path
    windows_path = r"C:\Users\runneradmin\.llamafarm"

    # Convert as done in server/core/celery/celery.py
    result_backend_path = f"{windows_path}/broker/results".replace("\\", "/")

    # Check if it's a Windows absolute path
    if len(result_backend_path) > 1 and result_backend_path[1] == ":":
        result_backend_url = f"file:///{result_backend_path}"
    else:
        result_backend_url = f"file://{result_backend_path}"

    # Should produce a valid file:// URL
    assert (
        result_backend_url == "file:///C:/Users/runneradmin/.llamafarm/broker/results"
    )
    # Should not contain backslashes
    assert "\\" not in result_backend_url
    # Should be parseable by urllib
    from urllib.parse import urlparse

    parsed = urlparse(result_backend_url)
    assert parsed.scheme == "file"


def test_unix_file_url_construction():
    """Test that Unix paths are converted to proper file:// URLs."""
    # Simulate Unix path
    unix_path = "/home/user/.llamafarm"

    # Convert as done in server/core/celery/celery.py
    result_backend_path = f"{unix_path}/broker/results".replace("\\", "/")

    # Check if it's a Windows absolute path
    if len(result_backend_path) > 1 and result_backend_path[1] == ":":
        result_backend_url = f"file:///{result_backend_path}"
    else:
        result_backend_url = f"file://{result_backend_path}"

    # Should produce a valid file:// URL
    assert result_backend_url == "file:///home/user/.llamafarm/broker/results"
    # Should be parseable by urllib
    from urllib.parse import urlparse

    parsed = urlparse(result_backend_url)
    assert parsed.scheme == "file"


def test_relative_path_file_url_construction():
    """Test that relative paths are converted to proper file:// URLs."""
    # Simulate relative path
    relative_path = ".llamafarm"

    # Convert as done in server/core/celery/celery.py
    result_backend_path = f"{relative_path}/broker/results".replace("\\", "/")

    # Check if it's a Windows absolute path
    if len(result_backend_path) > 1 and result_backend_path[1] == ":":
        result_backend_url = f"file:///{result_backend_path}"
    else:
        result_backend_url = f"file://{result_backend_path}"

    # Should produce a valid file:// URL
    assert result_backend_url == "file://.llamafarm/broker/results"
    # Should be parseable by urllib
    from urllib.parse import urlparse

    parsed = urlparse(result_backend_url)
    assert parsed.scheme == "file"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific test")
def test_celery_module_loads_on_windows():
    """Test that the celery module can be imported on Windows without errors."""
    # Set Windows environment
    with patch.dict(os.environ, {"LF_DATA_DIR": r"C:\Users\test\.llamafarm"}):
        try:
            # Import should not raise ValueError about port casting
            import importlib

            from core.celery import celery

            importlib.reload(celery)
        except ValueError as e:
            if "Port could not be cast" in str(e):
                pytest.fail(f"Celery module failed to load due to malformed URL: {e}")
            raise


def test_url_parsing_with_kombu():
    """Test that the constructed URLs can be parsed by kombu's URL parser."""
    from kombu.utils.url import _parse_url

    # Test Windows-style URL
    windows_url = "file:///C:/Users/runneradmin/.llamafarm/broker/results"
    try:
        parts = _parse_url(windows_url)
        # Should not raise ValueError
        assert parts is not None
    except ValueError as e:
        pytest.fail(f"kombu failed to parse Windows file:// URL: {e}")

    # Test Unix-style URL
    unix_url = "file:///home/user/.llamafarm/broker/results"
    try:
        parts = _parse_url(unix_url)
        # Should not raise ValueError
        assert parts is not None
    except ValueError as e:
        pytest.fail(f"kombu failed to parse Unix file:// URL: {e}")
