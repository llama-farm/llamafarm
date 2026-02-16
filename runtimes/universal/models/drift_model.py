"""Alibi Detect drift detection model.

This module provides data drift monitoring to detect when production data
distribution differs from training data. Critical for ML operations to
know when models need retraining.

Drift Types:
- Covariate drift: Input feature distribution changed
- Concept drift: Relationship between X and Y changed
- Prior drift: Target distribution changed

Detectors:
- KS (Kolmogorov-Smirnov): Fast, univariate numeric
- MMD (Maximum Mean Discrepancy): Medium speed, multivariate
- Chi-squared: Fast, categorical data
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from alibi_detect.cd import ChiSquareDrift, KSDrift, MMDDrift

from services.path_validator import (
    DRIFT_MODELS_DIR,
    get_model_path,
    sanitize_model_name,
    validate_model_path,
)

logger = logging.getLogger(__name__)

# Detector types and their configurations
DETECTOR_TYPES = {
    "ks": {
        "description": "Kolmogorov-Smirnov test for univariate drift detection",
        "class": KSDrift,
        "multivariate": False,
        "default_params": {"p_val": 0.05},
    },
    "mmd": {
        "description": "Maximum Mean Discrepancy for multivariate drift detection",
        "class": MMDDrift,
        "multivariate": True,
        "default_params": {"p_val": 0.05},
    },
    "chi2": {
        "description": "Chi-squared test for categorical drift detection",
        "class": ChiSquareDrift,
        "multivariate": False,
        "default_params": {"p_val": 0.05},
    },
}

# All available detectors
ALL_DETECTORS = list(DETECTOR_TYPES.keys())


@dataclass
class DriftResult:
    """Result from drift detection."""

    is_drift: bool
    p_value: float
    threshold: float
    distance: float | None = None
    p_values: list[float] | None = None  # Per-feature p-values for univariate tests


@dataclass
class DriftStatus:
    """Status of a drift detector."""

    detector_id: str
    detector_type: str
    is_fitted: bool
    reference_size: int
    detection_count: int
    drift_count: int
    last_detection: DriftResult | None


@dataclass
class DriftModelInfo:
    """Information about a saved drift model."""

    name: str
    detector: str
    created_at: str
    description: str | None = None
    is_fitted: bool = False
    reference_size: int = 0


def get_detectors_info() -> list[dict[str, Any]]:
    """Get information about all available detectors.

    Returns:
        List of detector info dictionaries
    """
    return [
        {
            "name": name,
            "description": info["description"],
            "multivariate": info["multivariate"],
            "default_params": info["default_params"],
        }
        for name, info in DETECTOR_TYPES.items()
    ]


def is_valid_detector(detector: str) -> bool:
    """Check if detector name is valid."""
    return detector in DETECTOR_TYPES


def get_all_detectors() -> list[str]:
    """Get list of all detector names."""
    return ALL_DETECTORS


def list_saved_models() -> list[DriftModelInfo]:
    """List all saved drift models.

    Returns:
        List of model info objects
    """
    models = []
    if not DRIFT_MODELS_DIR.exists():
        return models

    for model_file in DRIFT_MODELS_DIR.glob("*.joblib"):
        name = model_file.stem
        # Parse name_detector format
        parts = name.rsplit("_", 1)
        if len(parts) == 2:
            model_name, detector = parts
        else:
            model_name = name
            detector = "unknown"

        # Try to load metadata
        metadata_file = model_file.with_suffix(".metadata.json")
        description = None
        created_at = datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
        is_fitted = True
        reference_size = 0

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    description = metadata.get("description")
                    created_at = metadata.get("created_at", created_at)
                    is_fitted = metadata.get("is_fitted", True)
                    reference_size = metadata.get("reference_size", 0)
            except Exception:
                pass

        models.append(
            DriftModelInfo(
                name=model_name,
                detector=detector,
                created_at=created_at,
                description=description,
                is_fitted=is_fitted,
                reference_size=reference_size,
            )
        )

    return models


def delete_model(model_name: str) -> bool:
    """Delete a saved drift model.

    Args:
        model_name: Name of the model to delete

    Returns:
        True if deleted, False if not found
    """
    if not DRIFT_MODELS_DIR.exists():
        return False

    safe_name = sanitize_model_name(model_name)
    deleted = False

    # Find and delete matching files
    for model_file in DRIFT_MODELS_DIR.glob(f"{safe_name}_*.joblib"):
        try:
            model_file.unlink()
            deleted = True
            # Also delete metadata if exists
            metadata_file = model_file.with_suffix(".metadata.json")
            if metadata_file.exists():
                metadata_file.unlink()
        except Exception as e:
            logger.error(f"Failed to delete {model_file}: {e}")

    return deleted


class DriftModel:
    """Alibi Detect-based data drift detector.

    Monitors for distribution changes between reference and production data:
    - KS test: Univariate numeric drift
    - MMD: Multivariate drift
    - Chi-squared: Categorical drift
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        detector: str = "ks",
        **kwargs: Any,
    ):
        """Initialize drift model.

        Args:
            model_id: Unique identifier for this model
            device: Device to use (drift detection is CPU-based)
            detector: Detector type (ks, mmd, chi2)
            **kwargs: Additional detector parameters
        """
        if detector not in DETECTOR_TYPES:
            raise ValueError(
                f"Unknown detector: {detector}. "
                f"Valid detectors: {list(DETECTOR_TYPES.keys())}"
            )

        self.model_id = model_id
        self.device = device
        self.detector_type = detector
        self.detector_config = DETECTOR_TYPES[detector]
        self.kwargs = kwargs

        # Detector instance (created on fit)
        self._detector = None
        self._is_fitted = False
        self._reference_data = None
        self._detection_count = 0
        self._drift_count = 0
        self._last_result = None

    @property
    def model_type(self) -> str:
        """Get the model type string."""
        return f"drift_{self.detector_type}"

    @property
    def is_fitted(self) -> bool:
        """Check if the detector is fitted."""
        return self._is_fitted

    @property
    def reference_size(self) -> int:
        """Get the number of reference samples."""
        if self._reference_data is None:
            return 0
        return len(self._reference_data)

    async def load(self) -> None:
        """Initialize the detector.

        Drift detectors are created during fit, not load.
        This is a no-op.
        """
        logger.info(f"ADTK drift detector ready: {self.detector_type}")

    async def fit(
        self,
        reference_data: list[list[float]] | np.ndarray,
        feature_names: list[str] | None = None,
        autosave: bool = True,
        overwrite: bool = True,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Fit the drift detector on reference data.

        Args:
            reference_data: Reference distribution samples (n_samples x n_features)
            feature_names: Optional feature names
            autosave: Whether to auto-save after fitting
            overwrite: Whether to overwrite existing model
            description: Optional model description

        Returns:
            Fit result dictionary
        """
        start_time = time.time()

        # Convert to numpy array
        if not isinstance(reference_data, np.ndarray):
            reference_data = np.array(reference_data, dtype=np.float32)

        self._reference_data = reference_data
        self._feature_names = feature_names

        # Get detector parameters
        detector_class = self.detector_config["class"]
        params = self.detector_config["default_params"].copy()
        params.update(self.kwargs)

        # Create and fit detector
        try:
            self._detector = detector_class(reference_data, **params)
            self._is_fitted = True
        except Exception as e:
            logger.error(f"Failed to fit drift detector: {e}")
            raise

        training_time_ms = (time.time() - start_time) * 1000

        result = {
            "model": self.model_id,
            "detector": self.detector_type,
            "reference_size": len(reference_data),
            "n_features": reference_data.shape[1] if len(reference_data.shape) > 1 else 1,
            "training_time_ms": training_time_ms,
        }

        # Auto-save after fit
        if autosave:
            save_path = await self.save(overwrite=overwrite, description=description)
            result["saved_path"] = str(save_path)

        return result

    async def detect(
        self,
        data: list[list[float]] | np.ndarray,
    ) -> DriftResult:
        """Check for drift in new data.

        Args:
            data: New data to check (n_samples x n_features)

        Returns:
            DriftResult with detection outcome
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted first. Call fit() with reference data.")

        # Convert to numpy array
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        # Run drift detection
        try:
            result = self._detector.predict(data)
        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise

        # Extract results
        # For univariate detectors (KS, Chi2), is_drift and p_val are arrays (one per feature)
        # For multivariate (MMD), they are scalars
        is_drift_raw = result["data"]["is_drift"]
        p_val_raw = result["data"]["p_val"]
        threshold_raw = result["data"]["threshold"]

        # Handle array vs scalar results
        # Check if it's array-like with len() > 1
        p_values = None  # Per-feature p-values for univariate tests
        try:
            if hasattr(p_val_raw, "__len__") and len(p_val_raw) > 1:
                # Univariate: drift if ANY feature shows drift
                is_drift = bool(np.any(is_drift_raw))
                # Use minimum p-value (most significant)
                p_value = float(np.min(p_val_raw))
                # Store per-feature p-values
                p_values = [float(p) for p in p_val_raw]
                threshold = float(threshold_raw) if np.isscalar(threshold_raw) else float(np.array(threshold_raw).flat[0])
            else:
                # Scalar or single-element array - handle numpy arrays safely
                if isinstance(is_drift_raw, np.ndarray):
                    is_drift = bool(is_drift_raw.item())
                else:
                    is_drift = bool(is_drift_raw)
                p_value = float(np.array(p_val_raw).flat[0]) if hasattr(p_val_raw, "__len__") else float(p_val_raw)
                threshold = float(np.array(threshold_raw).flat[0]) if hasattr(threshold_raw, "__len__") else float(threshold_raw)
        except (TypeError, ValueError):
            # Fallback for scalar values
            if isinstance(is_drift_raw, np.ndarray):
                is_drift = bool(is_drift_raw.item())
            else:
                is_drift = bool(is_drift_raw)
            p_value = float(p_val_raw)
            threshold = float(threshold_raw)

        # Distance may not always be present
        distance = None
        if "distance" in result["data"]:
            dist_raw = result["data"]["distance"]
            if dist_raw is not None:
                if isinstance(dist_raw, np.ndarray):
                    distance = float(np.mean(dist_raw))
                else:
                    distance = float(dist_raw)

        # Update stats
        self._detection_count += 1
        if is_drift:
            self._drift_count += 1

        drift_result = DriftResult(
            is_drift=is_drift,
            p_value=p_value,
            threshold=threshold,
            distance=distance,
            p_values=p_values,
        )
        self._last_result = drift_result

        return drift_result

    def get_status(self) -> DriftStatus:
        """Get the current status of the detector.

        Returns:
            DriftStatus with detection statistics
        """
        return DriftStatus(
            detector_id=self.model_id,
            detector_type=self.detector_type,
            is_fitted=self._is_fitted,
            reference_size=self.reference_size,
            detection_count=self._detection_count,
            drift_count=self._drift_count,
            last_detection=self._last_result,
        )

    async def reset(self) -> None:
        """Reset the detector state (clear reference and stats)."""
        self._detector = None
        self._is_fitted = False
        self._reference_data = None
        self._detection_count = 0
        self._drift_count = 0
        self._last_result = None
        logger.info(f"Reset drift detector: {self.model_id}")

    async def save(
        self,
        path: Path | None = None,
        overwrite: bool = True,
        description: str | None = None,
    ) -> Path:
        """Save the detector to disk.

        Args:
            path: Optional custom path. If None, uses standard location.
            overwrite: Whether to overwrite existing file
            description: Optional model description

        Returns:
            Path where model was saved
        """
        if path is None:
            path = get_model_path(self.model_id, self.detector_type, "drift")

        # Ensure parent directory exists
        DRIFT_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Add .joblib extension
        if not str(path).endswith(".joblib"):
            path = Path(str(path) + ".joblib")

        # Validate path is within safe directory
        validate_model_path(path, "drift")

        # Handle versioning if not overwriting
        if not overwrite and path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = path.stem
            path = path.with_stem(f"{base}_{timestamp}")

        # Save model state
        state = {
            "detector": self._detector,
            "detector_type": self.detector_type,
            "is_fitted": self._is_fitted,
            "reference_data": self._reference_data,
            "detection_count": self._detection_count,
            "drift_count": self._drift_count,
            "last_result": self._last_result,
            "kwargs": self.kwargs,
        }

        joblib.dump(state, path)
        logger.info(f"Saved drift model to {path}")

        # Save metadata
        metadata = {
            "model_id": self.model_id,
            "detector": self.detector_type,
            "is_fitted": self._is_fitted,
            "reference_size": self.reference_size,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "detection_count": self._detection_count,
            "drift_count": self._drift_count,
        }
        metadata_path = path.with_suffix(".metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return path

    async def load_from_path(self, path: Path) -> None:
        """Load model state from a saved file.

        Args:
            path: Path to the saved model
        """
        # Validate path
        validate_model_path(path, "drift")

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = joblib.load(path)
        self._detector = state["detector"]
        self.detector_type = state["detector_type"]
        self.detector_config = DETECTOR_TYPES[self.detector_type]
        self._is_fitted = state["is_fitted"]
        self._reference_data = state.get("reference_data")
        self._detection_count = state.get("detection_count", 0)
        self._drift_count = state.get("drift_count", 0)
        self._last_result = state.get("last_result")
        self.kwargs = state.get("kwargs", {})

        logger.info(f"Loaded drift model from {path}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        self._detector = None
        self._is_fitted = False
        self._reference_data = None
        self._detection_count = 0
        self._drift_count = 0
        self._last_result = None
        logger.info(f"Unloaded drift model: {self.model_id}")
