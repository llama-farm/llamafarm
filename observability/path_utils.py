"""
Path utilities for observability module.

Follows the same patterns as server/services/project_service.py for consistency.
"""

import os
from pathlib import Path


def get_data_dir() -> str:
    """
    Get the LF data directory, trying settings first, then environment variable.

    Returns:
        str: Path to the LF data directory (e.g., ~/.llamafarm or /var/lib/llamafarm)
    """
    try:
        # Try to import from server/rag settings
        from core.settings import settings
        # Server uses lf_data_dir, RAG uses LF_DATA_DIR
        return getattr(settings, 'lf_data_dir', None) or getattr(settings, 'LF_DATA_DIR', None)
    except ImportError:
        # Fall back to environment variable (for testing or standalone use)
        return os.getenv("LF_DATA_DIR", str(Path.home() / ".llamafarm"))


def get_project_path(namespace: str, project: str) -> str:
    """
    Get the project directory path with security validation.

    Follows the same pattern as ProjectService.get_project_dir():
    - Uses os.path.join() for path construction
    - Uses os.normpath() for normalization
    - Validates against path traversal with startswith() check

    Args:
        namespace: Project namespace
        project: Project name

    Returns:
        str: Validated absolute path to the project directory

    Raises:
        ValueError: If path traversal is detected
    """
    base_path = os.path.join(get_data_dir(), "projects")
    raw_path = os.path.join(base_path, namespace, project)
    norm_path = os.path.normpath(raw_path)

    # Security: Ensure the normalized path is within the base_path
    # This is the same check used by ProjectService (project_service.py:79-82)
    if not norm_path.startswith(os.path.abspath(base_path) + os.sep):
        raise ValueError(
            f"Invalid namespace or project: path traversal detected"
        )

    return norm_path
