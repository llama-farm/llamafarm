"""Pydantic models for Alibi Detect drift detection endpoints.

Data drift monitoring detects when production data distribution differs from
training data. Critical for ML operations to know when models need retraining.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# Detector types
DriftDetectorType = Literal["ks", "mmd", "chi2"]


class DriftFitRequest(BaseModel):
    """Request to fit a drift detector on reference data."""

    model: str | None = Field(
        default=None,
        description="Model name. Auto-generated if not provided.",
    )
    detector: DriftDetectorType = Field(
        default="ks",
        description="Detector type: ks (univariate), mmd (multivariate), chi2 (categorical)",
    )
    reference_data: list[list[float]] = Field(
        ...,
        description="Reference distribution samples (n_samples x n_features)",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Optional feature names",
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
        description="Detector-specific parameters (e.g., p_val)",
    )


class DriftFitResponse(BaseModel):
    """Response from fitting a drift detector."""

    model: str = Field(..., description="Model name (generated if not provided)")
    detector: str = Field(..., description="Detector type used")
    saved_path: str = Field(..., description="Path where model was saved")
    training_time_ms: float = Field(..., description="Training time in milliseconds")
    reference_size: int = Field(..., description="Number of reference samples")
    n_features: int = Field(..., description="Number of features")


class DriftDetectRequest(BaseModel):
    """Request to check for drift in new data."""

    model: str = Field(
        ...,
        description="Model name to use. Supports '-latest' suffix.",
    )
    data: list[list[float]] = Field(
        ...,
        description="Data to check for drift (n_samples x n_features)",
    )


class DriftResult(BaseModel):
    """Result from drift detection."""

    is_drift: bool = Field(..., description="Whether drift was detected")
    p_value: float = Field(..., description="P-value from statistical test (min across features for univariate)")
    threshold: float = Field(..., description="Threshold used for decision")
    distance: float | None = Field(
        default=None,
        description="Distance metric (for some detectors)",
    )
    p_values: list[float] | None = Field(
        default=None,
        description="Per-feature p-values (for univariate detectors like KS)",
    )


class DriftDetectResponse(BaseModel):
    """Response from drift detection."""

    model: str = Field(..., description="Model name used")
    detector: str = Field(..., description="Detector type used")
    result: DriftResult = Field(..., description="Detection result")
    detection_time_ms: float = Field(..., description="Detection time in milliseconds")


class DriftLoadRequest(BaseModel):
    """Request to load a saved drift model."""

    model: str = Field(
        ...,
        description="Model name to load. Supports '-latest' suffix.",
    )


class DriftLoadResponse(BaseModel):
    """Response from loading a drift model."""

    model: str = Field(..., description="Model name loaded")
    detector: str = Field(..., description="Detector type")
    is_fitted: bool = Field(..., description="Whether model is fitted")
    reference_size: int = Field(..., description="Number of reference samples")


class DriftStatusRequest(BaseModel):
    """Request to get drift detector status."""

    model: str = Field(
        ...,
        description="Model name to check status",
    )


class DriftStatus(BaseModel):
    """Status of a drift detector."""

    model: str = Field(..., description="Model identifier")
    detector: str = Field(..., description="Detector type")
    is_fitted: bool = Field(..., description="Whether detector is fitted")
    reference_size: int = Field(..., description="Number of reference samples")
    detection_count: int = Field(..., description="Total detections performed")
    drift_count: int = Field(..., description="Number of drift detections")
    last_detection: DriftResult | None = Field(
        default=None,
        description="Result of last detection",
    )


class DriftDetectorInfo(BaseModel):
    """Information about a drift detector type."""

    name: str = Field(..., description="Detector name")
    description: str = Field(..., description="Human-readable description")
    multivariate: bool = Field(..., description="Whether detector handles multivariate data")
    default_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Default parameters for this detector",
    )


class DriftDetectorsResponse(BaseModel):
    """Response listing available drift detectors."""

    detectors: list[DriftDetectorInfo] = Field(
        ...,
        description="Available detector types",
    )


class DriftModelInfo(BaseModel):
    """Information about a saved drift model."""

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
    reference_size: int = Field(
        default=0,
        description="Number of reference samples",
    )


class DriftModelsResponse(BaseModel):
    """Response listing saved drift models."""

    models: list[DriftModelInfo] = Field(
        default_factory=list,
        description="List of saved models",
    )


class DriftDeleteResponse(BaseModel):
    """Response from deleting a drift model."""

    model: str = Field(..., description="Model name that was deleted")
    deleted: bool = Field(..., description="Whether the model was deleted")


class DriftResetResponse(BaseModel):
    """Response from resetting a drift detector."""

    model: str = Field(..., description="Model name that was reset")
    success: bool = Field(..., description="Whether reset succeeded")
