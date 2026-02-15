"""Pydantic models for ADTK (time-series anomaly detection) endpoints.

ADTK detects temporal anomalies that point anomaly detectors miss:
- Level shifts (sudden baseline changes)
- Seasonal anomalies (pattern violations)
- Spikes/dips (IQR-based outliers)
- Volatility shifts (variance changes)
- Persist anomalies (stuck values)
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Detector types
ADTKDetectorType = Literal[
    "level_shift",
    "seasonal",
    "spike",
    "volatility_shift",
    "persist",
    "threshold",
]


class ADTKDataPoint(BaseModel):
    """A single time-series data point."""

    timestamp: str = Field(..., description="ISO timestamp")
    value: float = Field(..., description="Data value")


class ADTKFitRequest(BaseModel):
    """Request to fit an ADTK detector on time-series data."""

    model: str | None = Field(
        default=None,
        description="Model name. Auto-generated if not provided.",
    )
    detector: ADTKDetectorType = Field(
        default="level_shift",
        description="Detector type to use",
    )
    data: list[ADTKDataPoint | dict[str, Any]] = Field(
        ...,
        description="Time-series data [{timestamp, value}, ...]",
    )
    overwrite: bool = Field(
        default=True,
        description="Overwrite existing model or create versioned copy",
    )
    description: str | None = Field(
        default=None,
        description="Optional model description",
    )
    # Detector-specific parameters
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific parameters (e.g., c, window, side)",
    )


class ADTKFitResponse(BaseModel):
    """Response from fitting an ADTK detector."""

    model: str = Field(..., description="Model name (generated if not provided)")
    detector: str = Field(..., description="Detector type used")
    saved_path: str = Field(..., description="Path where model was saved")
    training_time_ms: float = Field(..., description="Training time in milliseconds")
    samples_fitted: int = Field(..., description="Number of samples used for fitting")
    requires_training: bool = Field(
        ...,
        description="Whether this detector type requires training",
    )


class ADTKDetectRequest(BaseModel):
    """Request to detect anomalies in time-series data."""

    model: str | None = Field(
        default=None,
        description="Model name. If None, uses default detector settings.",
    )
    detector: ADTKDetectorType = Field(
        default="level_shift",
        description="Detector type to use (ignored if model is specified)",
    )
    data: list[ADTKDataPoint | dict[str, Any]] = Field(
        ...,
        description="Time-series data [{timestamp, value}, ...]",
    )
    # Detector-specific parameters (used if no model specified)
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Detector-specific parameters (e.g., c, window, side)",
    )


class ADTKAnomaly(BaseModel):
    """A detected time-series anomaly."""

    timestamp: str = Field(..., description="ISO timestamp of anomaly")
    value: float = Field(..., description="Value at anomaly point")
    anomaly_type: str = Field(..., description="Type of anomaly detected")
    score: float | None = Field(
        default=None,
        description="Anomaly score (not available for all detectors)",
    )


class ADTKDetectResponse(BaseModel):
    """Response from ADTK anomaly detection."""

    model: str | None = Field(
        default=None,
        description="Model name used (if loaded from saved model)",
    )
    detector: str = Field(..., description="Detector type used")
    anomalies: list[ADTKAnomaly] = Field(
        default_factory=list,
        description="List of detected anomalies",
    )
    total_points: int = Field(..., description="Total number of data points")
    anomaly_count: int = Field(..., description="Number of anomalies detected")
    detection_time_ms: float = Field(..., description="Detection time in milliseconds")


class ADTKLoadRequest(BaseModel):
    """Request to load a saved ADTK model."""

    model: str = Field(
        ...,
        description="Model name to load. Supports '-latest' suffix.",
    )


class ADTKLoadResponse(BaseModel):
    """Response from loading an ADTK model."""

    model: str = Field(..., description="Model name loaded")
    detector: str = Field(..., description="Detector type")
    is_fitted: bool = Field(..., description="Whether model is fitted")


class ADTKDetectorInfo(BaseModel):
    """Information about an ADTK detector type."""

    name: str = Field(..., description="Detector name")
    description: str = Field(..., description="Human-readable description")
    requires_training: bool = Field(..., description="Whether training is required")
    default_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameters for this detector",
    )


class ADTKDetectorsResponse(BaseModel):
    """Response listing available ADTK detectors."""

    detectors: list[ADTKDetectorInfo] = Field(
        ...,
        description="Available detector types",
    )


class ADTKModelInfo(BaseModel):
    """Information about a saved ADTK model."""

    name: str = Field(..., description="Model name")
    detector: str = Field(..., description="Detector type")
    created_at: str = Field(..., description="ISO timestamp when created")
    description: str | None = Field(
        default=None,
        description="Optional model description",
    )
    is_fitted: bool = Field(
        default=False,
        description="Whether model is fitted",
    )


class ADTKModelsResponse(BaseModel):
    """Response listing saved ADTK models."""

    models: list[ADTKModelInfo] = Field(
        default_factory=list,
        description="List of saved models",
    )


class ADTKDeleteResponse(BaseModel):
    """Response from deleting an ADTK model."""

    model: str = Field(..., description="Model name that was deleted")
    deleted: bool = Field(..., description="Whether the model was deleted")
