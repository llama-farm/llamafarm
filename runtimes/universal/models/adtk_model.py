"""ADTK (Anomaly Detection Toolkit) model for time-series anomaly detection.

This module provides time-series specific anomaly detection that complements PyOD:
- Level Shifts: Sudden changes in baseline
- Seasonal Anomalies: Unusual patterns within expected cycles
- Spikes/Dips: Short-term outliers
- Volatility Shifts: Changes in variance
- Persist: Stuck sensor values

ADTK specializes in temporal patterns that point anomaly detectors miss.
"""

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from adtk.detector import (
    InterQuartileRangeAD,
    LevelShiftAD,
    PersistAD,
    SeasonalAD,
    ThresholdAD,
    VolatilityShiftAD,
)

from services.path_validator import (
    ADTK_MODELS_DIR,
    get_model_path,
    sanitize_model_name,
    validate_model_path,
)

logger = logging.getLogger(__name__)

# Detector types and their configurations
DETECTOR_TYPES = {
    "level_shift": {
        "description": "Detects sudden changes in baseline level",
        "class": LevelShiftAD,
        "requires_training": False,
        "default_params": {"c": 6.0, "side": "both", "window": 5},
    },
    "seasonal": {
        "description": "Detects anomalies relative to seasonal patterns",
        "class": SeasonalAD,
        "requires_training": True,
        "default_params": {"c": 3.0, "side": "both"},
    },
    "spike": {
        "description": "Detects short-term spikes and dips (IQR-based)",
        "class": InterQuartileRangeAD,
        "requires_training": False,
        "default_params": {"c": 1.5},
    },
    "volatility_shift": {
        "description": "Detects changes in variance/volatility",
        "class": VolatilityShiftAD,
        "requires_training": False,
        "default_params": {"c": 6.0, "side": "both", "window": 30},
    },
    "persist": {
        "description": "Detects stuck/constant values (sensor failure)",
        "class": PersistAD,
        "requires_training": False,
        "default_params": {"c": 3.0, "side": "positive", "window": 5},
    },
    "threshold": {
        "description": "Simple threshold-based detection",
        "class": ThresholdAD,
        "requires_training": False,
        "default_params": {"low": None, "high": None},
    },
}

# All available detectors
ALL_DETECTORS = list(DETECTOR_TYPES.keys())


@dataclass
class ADTKAnomaly:
    """Represents a detected time-series anomaly."""

    timestamp: str
    value: float
    anomaly_type: str
    score: float | None = None


@dataclass
class ADTKDetectResult:
    """Result from ADTK detection."""

    detector: str
    anomalies: list[ADTKAnomaly]
    total_points: int
    anomaly_count: int
    detection_time_ms: float


@dataclass
class ADTKModelInfo:
    """Information about a saved ADTK model."""

    name: str
    detector: str
    created_at: str
    description: str | None = None
    is_fitted: bool = False


def get_detectors_info() -> list[dict[str, Any]]:
    """Get information about all available detectors.

    Returns:
        List of detector info dictionaries
    """
    return [
        {
            "name": name,
            "description": info["description"],
            "requires_training": info["requires_training"],
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


def list_saved_models() -> list[ADTKModelInfo]:
    """List all saved ADTK models.

    Returns:
        List of model info objects
    """
    models = []
    if not ADTK_MODELS_DIR.exists():
        return models

    for model_file in ADTK_MODELS_DIR.glob("*.joblib"):
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

        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                    description = metadata.get("description")
                    created_at = metadata.get("created_at", created_at)
                    is_fitted = metadata.get("is_fitted", True)
            except Exception:
                pass

        models.append(
            ADTKModelInfo(
                name=model_name,
                detector=detector,
                created_at=created_at,
                description=description,
                is_fitted=is_fitted,
            )
        )

    return models


def delete_model(model_name: str) -> bool:
    """Delete a saved ADTK model.

    Args:
        model_name: Name of the model to delete

    Returns:
        True if deleted, False if not found
    """
    if not ADTK_MODELS_DIR.exists():
        return False

    safe_name = sanitize_model_name(model_name)
    deleted = False

    # Find and delete matching files
    for model_file in ADTK_MODELS_DIR.glob(f"{safe_name}_*.joblib"):
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


class ADTKModel:
    """ADTK-based time-series anomaly detector.

    Provides time-series specific anomaly detection:
    - Level shifts (sudden baseline changes)
    - Seasonal anomalies (pattern violations)
    - Spikes/dips (IQR-based outliers)
    - Volatility shifts (variance changes)
    - Persist anomalies (stuck values)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        detector: str = "level_shift",
        **kwargs: Any,
    ):
        """Initialize ADTK model.

        Args:
            model_id: Unique identifier for this model
            device: Device to use (ADTK is CPU-only)
            detector: Detector type (level_shift, seasonal, spike, etc.)
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

        # Detector instance (created on load/fit)
        self._detector = None
        self._is_fitted = False
        self._training_series = None

    @property
    def model_type(self) -> str:
        """Get the model type string."""
        return f"adtk_{self.detector_type}"

    @property
    def requires_training(self) -> bool:
        """Check if this detector requires training."""
        return self.detector_config["requires_training"]

    @property
    def is_fitted(self) -> bool:
        """Check if the detector is fitted."""
        return self._is_fitted

    async def load(self) -> None:
        """Load/initialize the detector.

        For detectors that don't require training, this creates the detector.
        For seasonal detector, this is a no-op until fit is called.
        """
        detector_class = self.detector_config["class"]
        default_params = self.detector_config["default_params"].copy()

        # Override defaults with user params
        default_params.update(self.kwargs)

        # Remove None values for threshold detector
        if self.detector_type == "threshold":
            default_params = {k: v for k, v in default_params.items() if v is not None}

        try:
            self._detector = detector_class(**default_params)
            logger.info(
                f"Loaded ADTK detector: {self.detector_type} with params {default_params}"
            )
        except Exception as e:
            logger.error(f"Failed to create detector: {e}")
            raise

    async def fit(
        self,
        data: list[dict[str, Any]],
        autosave: bool = True,
        overwrite: bool = True,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Fit the detector on time-series data.

        Args:
            data: List of dicts with 'timestamp' and 'value' keys
            autosave: Whether to auto-save after fitting
            overwrite: Whether to overwrite existing model
            description: Optional model description

        Returns:
            Fit result dictionary
        """
        if self._detector is None:
            await self.load()

        start_time = time.time()

        # Convert to pandas Series with DatetimeIndex
        series = self._to_pandas_series(data)
        self._training_series = series

        # Fit the detector (some detectors don't need this)
        if self.requires_training:
            try:
                self._detector.fit(series)
            except Exception as e:
                logger.error(f"Failed to fit detector: {e}")
                raise

        self._is_fitted = True
        training_time_ms = (time.time() - start_time) * 1000

        result = {
            "model": self.model_id,
            "detector": self.detector_type,
            "samples_fitted": len(data),
            "training_time_ms": training_time_ms,
            "requires_training": self.requires_training,
        }

        # Auto-save after fit
        if autosave:
            save_path = await self.save(overwrite=overwrite, description=description)
            result["saved_path"] = str(save_path)

        return result

    async def detect(self, data: list[dict[str, Any]]) -> ADTKDetectResult:
        """Detect anomalies in time-series data.

        Args:
            data: List of dicts with 'timestamp' and 'value' keys

        Returns:
            ADTKDetectResult with detected anomalies
        """
        if self._detector is None:
            await self.load()

        start_time = time.time()

        # Convert to pandas Series with DatetimeIndex
        series = self._to_pandas_series(data)

        # Detect anomalies - use appropriate method based on detector type
        # ThresholdAD only has detect(), not fit_detect()
        # PersistAD and others use fit_detect()
        try:
            if self.detector_type == "threshold":
                # ThresholdAD only needs detect() - no fitting required
                anomaly_flags = self._detector.detect(series)
            else:
                # Other detectors use fit_detect() which handles both
                anomaly_flags = self._detector.fit_detect(series)
            self._is_fitted = True
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

        # Convert results to anomaly list
        anomalies = []
        for timestamp, is_anomaly in anomaly_flags.items():
            if pd.notna(is_anomaly) and is_anomaly:
                value = float(series.loc[timestamp])
                anomalies.append(
                    ADTKAnomaly(
                        timestamp=timestamp.isoformat(),
                        value=value,
                        anomaly_type=self.detector_type,
                        score=None,  # ADTK doesn't provide scores
                    )
                )

        detection_time_ms = (time.time() - start_time) * 1000

        return ADTKDetectResult(
            detector=self.detector_type,
            anomalies=anomalies,
            total_points=len(data),
            anomaly_count=len(anomalies),
            detection_time_ms=detection_time_ms,
        )

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
            path = get_model_path(self.model_id, self.detector_type, "adtk")

        # Ensure parent directory exists
        ADTK_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Add .joblib extension
        if not str(path).endswith(".joblib"):
            path = Path(str(path) + ".joblib")

        # Validate path is within safe directory
        validate_model_path(path, "adtk")

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
            "training_series": self._training_series,
            "kwargs": self.kwargs,
        }

        joblib.dump(state, path)
        logger.info(f"Saved ADTK model to {path}")

        # Save metadata
        metadata = {
            "model_id": self.model_id,
            "detector": self.detector_type,
            "is_fitted": self._is_fitted,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "samples_count": len(self._training_series) if self._training_series is not None else 0,
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
        validate_model_path(path, "adtk")

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        state = joblib.load(path)
        self._detector = state["detector"]
        self.detector_type = state["detector_type"]
        self.detector_config = DETECTOR_TYPES[self.detector_type]
        self._is_fitted = state["is_fitted"]
        self._training_series = state.get("training_series")
        self.kwargs = state.get("kwargs", {})

        logger.info(f"Loaded ADTK model from {path}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        self._detector = None
        self._is_fitted = False
        self._training_series = None
        logger.info(f"Unloaded ADTK model: {self.model_id}")

    def _to_pandas_series(self, data: list[dict[str, Any]]) -> pd.Series:
        """Convert data list to pandas Series with DatetimeIndex.

        ADTK requires a pandas Series with DatetimeIndex.

        Args:
            data: List of dicts with 'timestamp' and 'value' keys

        Returns:
            pandas Series with DatetimeIndex
        """
        # Extract timestamps and values
        timestamps = []
        values = []

        for point in data:
            ts = point.get("timestamp")
            val = point.get("value")

            if ts is None or val is None:
                continue

            # Parse timestamp
            if isinstance(ts, str):
                ts = pd.to_datetime(ts)
            timestamps.append(ts)
            values.append(float(val))

        # Create Series with DatetimeIndex
        series = pd.Series(values, index=pd.DatetimeIndex(timestamps))
        series = series.sort_index()

        return series
