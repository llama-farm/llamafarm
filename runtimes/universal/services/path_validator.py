"""Unified path validation service for Universal Runtime.

Consolidates path validation logic that was previously duplicated across:
- server.py (_validate_path_within_directory, _sanitize_model_name, _sanitize_filename)
- models/anomaly_model.py (_validate_model_path)
- models/classifier_model.py (_validate_model_path)

Security: This module provides path traversal protection for all file operations.
"""

import os
import re
from pathlib import Path

# Base data directory - uses standard LlamaFarm data directory
# ~/.llamafarm/ (or LF_DATA_DIR if set)
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))

# Safe directories for model storage
MODELS_BASE_DIR = (_LF_DATA_DIR / "models").resolve()
ANOMALY_MODELS_DIR = (MODELS_BASE_DIR / "anomaly").resolve()
CLASSIFIER_MODELS_DIR = (MODELS_BASE_DIR / "classifier").resolve()


class PathValidationError(ValueError):
    """Raised when path validation fails due to security concerns."""

    pass


def sanitize_model_name(name: str) -> str:
    """Sanitize model name to create a safe filename.

    Only allows alphanumeric characters, hyphens, and underscores.
    This prevents path traversal and ensures consistent naming.

    Args:
        name: Raw model name

    Returns:
        Sanitized name safe for use in file paths
    """
    return "".join(c for c in name if c.isalnum() or c in "-_")


def sanitize_filename(name: str) -> str:
    """Sanitize a filename, preserving extension dots.

    Only allows alphanumeric characters, hyphens, underscores, and dots.
    This prevents path traversal while allowing file extensions like .joblib

    Args:
        name: Raw filename

    Returns:
        Sanitized filename safe for use in file paths
    """
    return "".join(c for c in name if c.isalnum() or c in "-_.")


def validate_path_within_directory(path: Path, safe_dir: Path) -> Path:
    """Validate that a path is within the allowed directory.

    This is a security function to prevent path traversal attacks.
    Returns the resolved (absolute) path if valid.

    Args:
        path: Path to validate
        safe_dir: Directory that path must be within

    Returns:
        Resolved (absolute) path

    Raises:
        PathValidationError: If path is outside the allowed directory
    """
    resolved = path.resolve()
    safe_resolved = safe_dir.resolve()

    try:
        resolved.relative_to(safe_resolved)
    except ValueError:
        raise PathValidationError(
            f"Security error: Path '{path}' resolves outside allowed directory '{safe_dir}'"
        ) from None

    return resolved


def validate_model_path(model_path: Path, model_type: str) -> Path:
    """Validate that model path is within the appropriate safe directory.

    Security: This function prevents path traversal attacks by ensuring
    the model path resolves to a location within the appropriate model directory.

    Args:
        model_path: Path to validate
        model_type: Type of model ('anomaly' or 'classifier')

    Returns:
        Validated, resolved path

    Raises:
        PathValidationError: If path is outside the safe directory or uses path traversal
        ValueError: If model_type is unknown
    """
    # Determine safe directory based on model type
    if model_type == "anomaly":
        safe_dir = ANOMALY_MODELS_DIR
    elif model_type == "classifier":
        safe_dir = CLASSIFIER_MODELS_DIR
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Check for path traversal patterns
    path_str = str(model_path)
    if ".." in path_str:
        raise PathValidationError(
            f"Security error: Model path '{model_path}' contains '..' - "
            "Path traversal is not allowed."
        )

    # Resolve to absolute path
    resolved_path = model_path.resolve()

    # Verify the resolved path is within the safe directory
    try:
        resolved_path.relative_to(safe_dir)
    except ValueError:
        raise PathValidationError(
            f"Security error: Model path '{model_path}' is outside the allowed "
            f"directory '{safe_dir}'. Path traversal is not allowed."
        ) from None

    return resolved_path


def get_model_path(model_name: str, backend: str, model_type: str) -> Path:
    """Get the path for a model file based on name and backend.

    The path is always within the appropriate models directory - users cannot control it.

    Args:
        model_name: Name of the model
        backend: Model backend (e.g., 'isolation_forest', 'setfit')
        model_type: Type of model ('anomaly' or 'classifier')

    Returns:
        Path to the model file (without extension)
    """
    safe_name = sanitize_model_name(model_name)
    safe_backend = sanitize_model_name(backend)

    if model_type == "anomaly":
        filename = f"{safe_name}_{safe_backend}"
        return ANOMALY_MODELS_DIR / filename
    elif model_type == "classifier":
        return CLASSIFIER_MODELS_DIR / safe_name
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def ensure_model_directories() -> None:
    """Ensure all model directories exist."""
    ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_model_name(name: str) -> bool:
    """Check if a model name is valid (no path traversal characters).

    Args:
        name: Model name to validate

    Returns:
        True if valid, False otherwise
    """
    # Must not contain path separators or traversal patterns
    if "/" in name or "\\" in name or ".." in name:
        return False

    # Must contain at least one alphanumeric character
    return bool(re.search(r"[a-zA-Z0-9]", name))
