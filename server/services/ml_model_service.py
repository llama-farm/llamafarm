"""
MLModelService - Handle model storage, versioning, and resolution.

Provides:
- Versioned model storage in ~/.llamafarm/models/
- {base-name}_{timestamp} versioning when overwrite=False
- {base-name}-latest resolution to find most recent version
- Description metadata storage in metadata.json files

Supports all ML model types:
- classifier: SetFit text classification models (directory-based)
- anomaly: Anomaly detection models (PyOD backends)
- timeseries: Time-series forecasting models (Darts/Chronos)
- adtk: ADTK time-series anomaly detection
- drift: Data drift detection models
- shap: SHAP explainers
- catboost: CatBoost gradient boosting models
"""

import json
import logging
import os
import re
import uuid
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _safe_home() -> Path:
    """Return the user's home directory with fallback for embedded Python."""
    try:
        return Path.home()
    except RuntimeError:
        fb = (
            os.environ.get("USERPROFILE")
            or os.environ.get("APPDATA")
            or os.environ.get("LOCALAPPDATA")
        )
        return Path(fb) if fb else Path.cwd()


class MLModelService:
    """Service for managing ML model storage and versioning."""

    # Base directory for all models (uses LF_DATA_DIR if set, with safe home fallback)
    MODELS_DIR = Path(
        os.environ.get("LF_DATA_DIR", _safe_home() / ".llamafarm")
    ) / "models"

    # All supported model types
    MODEL_TYPES = [
        "classifier",
        "anomaly",
        "timeseries",
        "adtk",
        "drift",
        "shap",
        "catboost",
    ]

    # Model types that use directory-based storage (vs file-based)
    DIRECTORY_BASED_TYPES = ["classifier"]

    # Model types that don't include backend in filename
    NO_BACKEND_TYPES = ["classifier", "shap", "catboost"]

    # Timestamp format for versioning
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"

    # Pattern to match versioned model names: base_name_YYYYMMDD_HHMMSS
    VERSION_PATTERN = re.compile(r"^(.+)_(\d{8}_\d{6})$")

    # Known backends by model type (used to parse filenames)
    KNOWN_BACKENDS: dict[str, list[str]] = {
        "anomaly": [
            "isolation_forest",
            "one_class_svm",
            "local_outlier_factor",
            "autoencoder",
            # PyOD backends
            "ecod",
            "copod",
            "hbos",
            "knn",
            "lof",
            "abod",
            "cblof",
            "cof",
            "sod",
            "iforest",
            "inne",
            "lscp",
            "mcd",
            "ocsvm",
            "pca",
            "rod",
            "sampling",
        ],
        "timeseries": [
            "arima",
            "exponential_smoothing",
            "theta",
            "chronos",
            "chronos-bolt",
        ],
        "adtk": [
            "threshold",
            "quantile",
            "inter_quartile_range",
            "generalized_esd",
            "persist",
            "level_shift",
            "volatility_shift",
            "seasonal",
            "autoregressive",
        ],
        "drift": [
            "kolmogorov_smirnov",
            "chi_squared",
            "population_stability",
            "jensen_shannon",
        ],
    }

    # Legacy alias for backwards compatibility
    ANOMALY_BACKENDS = KNOWN_BACKENDS.get("anomaly", [])

    @classmethod
    def ensure_dirs(cls) -> None:
        """Ensure all model directories exist."""
        for model_type in cls.MODEL_TYPES:
            (cls.MODELS_DIR / model_type).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_dir(cls, model_type: str) -> Path:
        """Get the directory for a model type.

        Args:
            model_type: One of MODEL_TYPES (classifier, anomaly, timeseries, etc.)

        Returns:
            Path to the model type directory

        Raises:
            ValueError: If model_type is not recognized
        """
        if model_type not in cls.MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Valid types: {cls.MODEL_TYPES}"
            )
        model_dir = cls.MODELS_DIR / model_type
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir

    @classmethod
    def generate_model_name(cls, model_type: str) -> str:
        """Generate a unique model name if none provided.

        Args:
            model_type: Type of model (for prefix)

        Returns:
            Generated name like "timeseries-a1b2c3d4"
        """
        return f"{model_type}-{uuid.uuid4().hex[:8]}"

    @classmethod
    def get_backends_for_type(cls, model_type: str) -> list[str]:
        """Get known backends for a model type.

        Args:
            model_type: The model type

        Returns:
            List of known backend names, or empty list if none defined
        """
        return cls.KNOWN_BACKENDS.get(model_type, [])

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
            model_type: One of MODEL_TYPES
            base_name: Base model name

        Returns:
            List of version names, sorted oldest to newest
        """
        model_dir = cls.get_model_dir(model_type)
        versions = []

        # Pattern to match this base name's versions
        pattern = re.compile(rf"^{re.escape(base_name)}_(\d{{8}}_\d{{6}})")

        # Directory-based models (classifier, etc.)
        if model_type in cls.DIRECTORY_BASED_TYPES:
            for item in model_dir.iterdir():
                if item.is_dir():
                    match = pattern.match(item.name)
                    if match:
                        versions.append((match.group(1), item.name))
                    elif item.name == base_name:
                        # Non-versioned (overwrite=True) version - use actual mtime
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        ts = mtime.strftime("%Y%m%d_%H%M%S")
                        versions.append((ts, item.name))
        # Models that don't include backend in filename (shap, catboost)
        elif model_type in cls.NO_BACKEND_TYPES and model_type not in cls.DIRECTORY_BASED_TYPES:
            # Get file extension based on model type
            ext = ".cbm" if model_type == "catboost" else ".joblib"
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix == ext:
                    name_without_ext = item.stem
                    match = pattern.match(name_without_ext)
                    if match:
                        versions.append((match.group(1), name_without_ext))
                    elif name_without_ext == base_name:
                        mtime = datetime.fromtimestamp(item.stat().st_mtime)
                        ts = mtime.strftime("%Y%m%d_%H%M%S")
                        versions.append((ts, name_without_ext))
        else:
            # File-based models with backend in filename (anomaly, timeseries, adtk, drift)
            # Filename format: {model}_{backend}.joblib
            known_backends = cls.get_backends_for_type(model_type)
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix == ".joblib":
                    name_without_ext = item.stem

                    # Remove known backend suffix to get model name
                    model_part = None
                    for backend in known_backends:
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
            model_type: One of MODEL_TYPES

        Returns:
            List of model info dicts
        """
        model_dir = cls.get_model_dir(model_type)
        models = []

        # Directory-based models (classifier)
        if model_type in cls.DIRECTORY_BASED_TYPES:
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
                            "model_type": model_type,
                            "path": str(item),
                            "created": created.isoformat(),
                            "is_versioned": match is not None,
                        }
                    )
        # Models without backend in filename (shap, catboost)
        elif model_type in cls.NO_BACKEND_TYPES and model_type not in cls.DIRECTORY_BASED_TYPES:
            ext = ".cbm" if model_type == "catboost" else ".joblib"
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix == ext:
                    name_without_ext = item.stem

                    # Parse version info
                    match = cls.VERSION_PATTERN.match(name_without_ext)
                    if match:
                        base_name, timestamp = match.groups()
                        created = datetime.strptime(timestamp, cls.TIMESTAMP_FORMAT)
                    else:
                        base_name = name_without_ext
                        created = datetime.fromtimestamp(item.stat().st_mtime)

                    models.append(
                        {
                            "name": name_without_ext,
                            "filename": item.name,
                            "base_name": base_name,
                            "model_type": model_type,
                            "path": str(item),
                            "size_bytes": item.stat().st_size,
                            "created": created.isoformat(),
                            "is_versioned": match is not None,
                        }
                    )
        else:
            # File-based models with backend in filename (anomaly, timeseries, adtk, drift)
            known_backends = cls.get_backends_for_type(model_type)
            for item in model_dir.iterdir():
                if item.is_file() and item.suffix in (".joblib", ".pkl", ".pt"):
                    name_without_ext = item.stem

                    # Parse backend from name using known backends
                    # Filename format: {model}_{backend}.joblib
                    model_name = name_without_ext
                    backend = "unknown"
                    for known_backend in known_backends:
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
                            "model_type": model_type,
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
            model_type: One of MODEL_TYPES
            name: Model name or filename

        Returns:
            True if deleted, False if not found

        Raises:
            ValueError: If the model name is invalid (e.g., path traversal)
        """
        import shutil

        model_dir = cls.get_model_dir(model_type)

        # Directory-based models (classifier)
        if model_type in cls.DIRECTORY_BASED_TYPES:
            path = cls._validate_path(model_dir, name)
            if path.is_dir():
                shutil.rmtree(path)
                logger.info(f"Deleted {model_type} model: {name}")
                return True
        else:
            # File-based models - name might be just the model name or the full filename
            path = cls._validate_path(model_dir, name)
            if path.is_file():
                # Also delete associated metadata
                cls._delete_metadata(model_type, name)
                path.unlink()
                logger.info(f"Deleted {model_type} model: {name}")
                return True

            # Try with various extensions used by models
            extensions = [".cbm"] if model_type == "catboost" else [".joblib", ".pkl", ".pt"]
            for ext in extensions:
                try:
                    path = cls._validate_path(model_dir, f"{name}{ext}")
                    if path.is_file():
                        cls._delete_metadata(model_type, name)
                        path.unlink()
                        logger.info(f"Deleted {model_type} model: {name}{ext}")
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

        For directory-based models (classifier):
            ~/.llamafarm/models/classifier/{model_name}/metadata.json
        For file-based models (anomaly, timeseries, etc.):
            ~/.llamafarm/models/{type}/{model_name}.metadata.json

        Raises:
            ValueError: If the model name is invalid (e.g., path traversal)
        """
        model_dir = cls.get_model_dir(model_type)

        # Validate model name to prevent path traversal
        cls._validate_path(model_dir, model_name)

        if model_type in cls.DIRECTORY_BASED_TYPES:
            return model_dir / model_name / "metadata.json"
        else:
            # For file-based models, store metadata alongside the model file
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
