"""
Path utilities for observability module.

Follows the same patterns as server/services/project_service.py for consistency.
"""

import os
import re
from pathlib import Path


def validate_path_component(component: str, name: str) -> None:
    """
    Validate a path component (namespace or project) for security.

    Ensures components only contain safe characters and prevents path traversal.

    Args:
        component: The path component to validate
        name: Name of the component (for error messages)

    Raises:
        ValueError: If the component contains invalid characters or patterns
    """
    if not component:
        raise ValueError(f"Invalid {name}: must be non-empty")

    # Allow alphanumeric, dash, underscore, and dot (for project names like "test-project-1")
    # But explicitly check for path traversal patterns first
    if ".." in component or "/" in component or "\\" in component or "\0" in component:
        raise ValueError(f"Invalid {name}: path traversal patterns not allowed")

    # Then validate against safe character set
    if not re.match(r"^[a-zA-Z0-9_.-]+$", component):
        raise ValueError(
            f"Invalid {name}: must contain only alphanumeric characters, dashes, underscores, and dots"
        )


def get_data_dir() -> str:
    """
    Get the LF data directory, trying settings first, then environment variable.

    Returns:
        str: Path to the LF data directory (e.g., ~/.llamafarm or /var/lib/llamafarm)
    """
    data_dir = None

    try:
        # Try to import from server/rag settings
        from core.settings import settings
        # Server uses lf_data_dir, RAG uses LF_DATA_DIR
        data_dir = getattr(settings, 'lf_data_dir', None) or getattr(settings, 'LF_DATA_DIR', None)
    except ImportError:
        pass

    # Fall back to environment variable or default if settings didn't provide a value
    if not data_dir:
        data_dir = os.getenv("LF_DATA_DIR", str(Path.home() / ".llamafarm"))

    return data_dir


def validate_file_path(file_path: str, parent_dir: str, description: str = "file") -> None:
    """
    Validate that a file path is contained within its parent directory.

    Uses normalized absolute paths to prevent path traversal attacks.

    Args:
        file_path: The file path to validate
        parent_dir: The expected parent directory
        description: Description of the file (for error messages)

    Raises:
        ValueError: If the file path escapes the parent directory
    """
    normalized_file = os.path.normpath(os.path.abspath(file_path))
    normalized_parent = os.path.normpath(os.path.abspath(parent_dir))

    # Security: Ensure file stays within parent directory
    if not normalized_file.startswith(normalized_parent + os.sep):
        raise ValueError(
            f"Invalid {description} path: path escapes parent directory"
        )


def get_project_path(namespace: str, project: str) -> str:
    """
    Get the project directory path with security validation.

    Follows the same pattern as ProjectService.get_project_dir():
    - Validates input components for safe characters
    - Uses os.path.join() for path construction
    - Uses os.normpath() for normalization
    - Validates against path traversal with startswith() check

    Args:
        namespace: Project namespace
        project: Project name

    Returns:
        str: Validated absolute path to the project directory

    Raises:
        ValueError: If path traversal is detected or invalid characters found
    """
    # Security: Validate inputs before path construction
    validate_path_component(namespace, "namespace")
    validate_path_component(project, "project")

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
