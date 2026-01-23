"""
MLModelService - Handle model storage, versioning, and resolution.

Provides:
- Versioned model storage in ~/.llamafarm/models/
- {base-name}_{timestamp} versioning when overwrite=False
- {base-name}-latest resolution to find most recent version
- Description metadata storage in metadata.json files
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class MLModelService:
    """Service for managing ML model storage and versioning."""

    # Base directory for all models
    MODELS_DIR = Path.home() / ".llamafarm" / "models"

    # Subdirectories by model type
    CLASSIFIER_DIR = "classifier"
    ANOMALY_DIR = "anomaly"

    # Timestamp format for versioning
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    # Pattern to match versioned model names: base_name_YYYYMMDD_HHMMSS
    VERSION_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{6})$")

    # Known anomaly detection backends (used to parse filenames)
    ANOMALY_BACKENDS = [
        "isolation_forest",
        "one_class_svm",
        "local_outlier_factor",
        "autoencoder",
    ]

    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensure model directories exist."""
        (cls.MODELS_DIR / cls.CLASSIFIER_DIR).mkdir(parents=True, exist_ok=True)
        (cls.MODELS_DIR / cls.ANOMALY_DIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_dir(cls, model_type: str) -> Path:
        """Get the directory for a model type.

        Args:
            model_type: 'classifier' or 'anomaly'

        Returns:
            Path to the model type directory
        """
        cls.ensure_dirs()
        if model_type == "classifier":
            return cls.MODELS_DIR / cls.CLASSIFIER_DIR
        elif model_type == "anomaly":
            return cls.MODELS_DIR / cls.ANOMALY_DIR
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    @classmethod
    def get_versioned_name(cls, base_name: str, overwrite: bool) -> str:
        """Get a model name, optionally versioned with timestamp.

        Args:
            base_name: The base model name
            overwrite: If True, return base_name as-is; if False, append timestamp

        Returns:
            Model name (versioned if overwrite=False)
        """
        if overwrite:
            return base_name

        timestamp = datetime.now().strftime(cls.TIMESTAMP_FORMAT)
        return f"{base_name}_{timestamp}"

    @classmethod
    def resolve_model_name(cls, model_type: str, name: str) -> str:
        """Resolve a model name, handling -latest suffix.

        Args:
            model_type: 'classifier' or 'anomaly'
            name: Model name, possibly ending in '-latest'

        Returns:
            Resolved model name (most recent version if -latest)
        """
        if not name.endswith("-latest"):
            return name

        # Extract base name by removing -latest suffix
        base_name = name[:-7]  # len("-latest") == 7

        # Find the latest version
        latest = cls.find_latest_version(model_type, base_name)
        if latest:
            logger.info(f"Resolved {name} to {latest}")
            return latest

        # No versioned model found, try the base name itself
        logger.info(f"No versioned model found for {base_name}, using base name")
        return base_name

    @classmethod
    def find_latest_version(cls, model_type: str, base_name: str) -> str | None:
        """Find the most recent versioned model.

        Args:
            model_type: 'classifier' or 'anomaly'
            base_name: Base model name (without version suffix)

        Returns:
            Name of the most recent version, or None if not found
        """
        versions = cls.list_versions(model_type, base_name)

        if not versions:
            return None

        # Versions are sorted by timestamp (newest last)
        return versions[-1]

    @classmethod
    def list_versions(cls, model_type: str, base_name: str) -> list[str]:
        """List all versions of a model, sorted by timestamp.

        Args:
            model_type: 'classifier' or 'anomaly'
            base_name: Base model name

        Returns:
            List of version names, sorted oldest to newest
        """
        model_dir = cls.get_model_dir(model_type)
        versions = []

        # Pattern to match this base name's versions
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{8}}_\d{{6}})")

        if model_type == "classifier":
            # Classifiers are directories
            for item in model_dir.iterdir():
                if item.is_dir():
                    match = pattern.match(item.name)
                    if match:
                        versions.append((match.group(1), item.name))
                    elif item.name == base_name:
                        # Non-versioned (overwrite=True) version - use actual mtime
                        # Format as timestamp so it sorts correctly with versioned models
                        from datetime import datetime

                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        ts = mtime.strftime("%Y%m%d_%H%M%S")
                        versions.append((ts, item.name))
        else:
            # Anomaly models are files like: {model}_{backend}.joblib
            # or versioned: {model}_{YYYYMMDD_HHMMSS}_{backend}.joblib
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix == ".joblib":
                    # Filename format: model_name_backend.joblib
                    # Extract name without .joblib extension
                    name_without_ext = item.stem

                    # Remove known backend suffix to get model name
                    model_part = None
                    for backend in cls.ANOMALY_BACKENDS:
                        suffix = f"_{backend}"
                        if name_without_ext.endswith(suffix):
                            model_part = name_without_ext[: -len(suffix)]
                            break

                    if model_part is None:
                        # Unknown backend format, skip
                        continue

                    # Check if model_part matches our base_name pattern
                    match = pattern.match(model_part)
                    if match:
                        versions.append((match.group(1), model_part))
                    elif model_part == base_name:
                        # Non-versioned (overwrite=True) version - use actual mtime
                        from datetime import datetime

                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        ts = mtime.strftime("%Y%m%d_%H%M%S")
                        versions.append((ts, model_part))

        # Sort by timestamp and return just the names
        versions.sort(key=lambda x: x[0])
        return [v[1] for v in versions]

    @classmethod
    def list_all_models(cls, model_type: str) -> list[dict]:
        """List all models of a type with their metadata.

        Args:
            model_type: 'classifier' or 'anomaly'

        Returns:
            List of model info dicts
        """
        model_dir = cls.get_model_dir(model_type)
        models = []

        if model_type == "classifier":
            for item in model_dir.iterdir():
                if item.is_dir():
                    # Parse version info
                    match = cls.VERSION_PATTERN.match(item.name)
                    if match:
                        base_name, timestamp = match.groups()
                        created = datetime.strptime(timestamp, cls.TIMESTAMP_FORMAT)
                    else:
                        base_name = item.name
                        created = datetime.fromtimestamp(item.stat().st_mtime)

                    models.append(
                        {
                            "name": item.name,
                            "base_name": base_name,
                            "path": str(item),
                            "created": created.isoformat(),
                            "is_versioned": match is not None,
                        }
                    )
        else:
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix in (".joblib", ".pkl", ".pt"):
                    name_without_ext = item.stem

                    # Parse backend from name using known backends
                    # Filename format: {model}_{backend}.joblib
                    model_name = name_without_ext
                    backend = "unknown"
                    for known_backend in cls.ANOMALY_BACKENDS:
                        suffix = f"_{known_backend}"
                        if name_without_ext.endswith(suffix):
                            model_name = name_without_ext[: -len(suffix)]
                            backend = known_backend
                            break

                    # Parse version info
                    match = cls.VERSION_PATTERN.match(model_name)
                    if match:
                        base_name, timestamp = match.groups()
                        created = datetime.strptime(timestamp, cls.TIMESTAMP_FORMAT)
                    else:
                        base_name = model_name
                        created = datetime.fromtimestamp(item.stat().st_mtime)

                    models.append(
                        {
                            "name": model_name,
                            "filename": item.name,
                            "base_name": base_name,
                            "backend": backend,
                            "path": str(item),
                            "size_bytes": item.stat().st_size,
                            "created": created.isoformat(),
                            "is_versioned": match is not None,
                        }
                    )

        # Sort by creation time, newest first
        models.sort(key=lambda x: x["created"], reverse=True)
        return models

    @classmethod
    def get_model_path(cls, model_type: str, name: str) -> Path:
        """Get the full path for a model.

        Args:
            model_type: 'classifier' or 'anomaly'
            name: Model name

        Returns:
            Full path to the model
        """
        return cls.get_model_dir(model_type) / name

    @classmethod
    def _validate_path(cls, model_dir: Path, name: str) -> Path:
        """Validate that a model path is within the model directory.

        Prevents path traversal attacks by ensuring the resolved path
        stays within the expected model directory.

        Args:
            model_dir: The base model directory
            name: The model name to validate

        Returns:
            The validated, resolved path

        Raises:
            ValueError: If the path would escape the model directory
        """
        # Reject names with path separators
        if "/" in name or "\\" in name:
            raise ValueError(f"Invalid model name: {name}")

        path = model_dir / name
        resolved = path.resolve()

        # Ensure resolved path is within model_dir
        try:
            resolved.relative_to(model_dir.resolve())
        except ValueError:
            raise ValueError(f"Invalid model name: {name}") from None

        return resolved

    @classmethod
    def delete_model(cls, model_type: str, name: str) -> bool:
        """Delete a model.

        Args:
            model_type: 'classifier' or 'anomaly'
            name: Model name or filename

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If the model name is invalid (e.g., path traversal)
        """
        import shutil

        model_dir = cls.get_model_dir(model_type)

        if model_type == "classifier":
            path = cls._validate_path(model_dir, name)
            if path.is_dir():
                shutil.rmtree(path)
                logger.info(f"Deleted classifier model: {name}")
                return True
        else:
            # For anomaly, name might be just the model name or the full filename
            path = cls._validate_path(model_dir, name)
            if path.is_file():
                # Also delete associated metadata
                cls._delete_metadata(model_type, name)
                path.unlink()
                logger.info(f"Deleted anomaly model: {name}")
                return True

            # Try with various extensions used by anomaly models
            for ext in (".joblib", ".pkl", ".pt"):
                try:
                    path = cls._validate_path(model_dir, f"{name}{ext}")
                    if path.is_file():
                        cls._delete_metadata(model_type, name)
                        path.unlink()
                        logger.info(f"Deleted anomaly model: {name}{ext}")
                        return True
                except ValueError:
                    continue

        return False

    # =========================================================================
    # Metadata Management (descriptions, etc.)
    # =========================================================================

    @classmethod
    def _get_metadata_path(cls, model_type: str, model_name: str) -> Path:
        """Get the path to a model's metadata file.

        For classifiers: ~/.llamafarm/models/classifier/{model_name}/metadata.json
        For anomaly: ~/.llamafarm/models/anomaly/{model_name}.metadata.json

        Raises:
            ValueError: If the model name is invalid (e.g., path traversal)
        """
        model_dir = cls.get_model_dir(model_type)

        # Validate model name to prevent path traversal
        cls._validate_path(model_dir, model_name)

        if model_type == "classifier":
            return model_dir / model_name / "metadata.json"
        else:
            # For anomaly models, store metadata alongside the model file
            return model_dir / f"{model_name}.metadata.json"

    @classmethod
    def save_description(
        cls, model_type: str, model_name: str, description: str | None
    ) -> None:
        """Save a description for a model.

        Args:
            model_type: 'classifier' or 'anomaly'
            model_name: The model name (without file extension for anomaly)
            description: The description text, or None to skip
        """
        if not description:
            return

        metadata_path = cls._get_metadata_path(model_type, model_name)

        # Load existing metadata or create new
        metadata = cls._load_metadata(metadata_path)
        metadata["description"] = description

        # Ensure parent directory exists
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        # Save metadata
        try:
            metadata_path.write_text(json.dumps(metadata, indent=2))
            logger.info(f"Saved description for {model_type} model: {model_name}")
        except Exception as e:
            logger.warning(f"Failed to save metadata for {model_name}: {e}")

    @classmethod
    def get_description(cls, model_type: str, model_name: str) -> str | None:
        """Get the description for a model.

        Args:
            model_type: 'classifier' or 'anomaly'
            model_name: The model name

        Returns:
            The description string, or None if not set
        """
        metadata_path = cls._get_metadata_path(model_type, model_name)
        metadata = cls._load_metadata(metadata_path)
        return metadata.get("description")

    @classmethod
    def _load_metadata(cls, metadata_path: Path) -> dict:
        """Load metadata from a file, returning empty dict if not found."""
        if metadata_path.exists():
            try:
                return json.loads(metadata_path.read_text())
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load metadata from {metadata_path}: {e}")
        return {}

    @classmethod
    def _delete_metadata(cls, model_type: str, model_name: str) -> None:
        """Delete metadata file for a model if it exists."""
        metadata_path = cls._get_metadata_path(model_type, model_name)
        if metadata_path.exists():
            try:
                metadata_path.unlink()
                logger.info(f"Deleted metadata for {model_name}")
            except OSError as e:
                logger.warning(f"Failed to delete metadata for {model_name}: {e}")
