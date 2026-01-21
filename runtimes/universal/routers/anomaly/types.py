"""Pydantic models for anomaly detection endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class AnomalyScoreRequest(BaseModel):
    """Anomaly scoring request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (varies by backend)
    """

    model: str = "default"  # Model identifier
    backend: str = "isolation_forest"  # isolation_forest, one_class_svm, local_outlier_factor, autoencoder, copod, hbos, ecod
    data: list[list[float]] | list[dict]  # Data points (numeric arrays or dicts)
    schema: dict[str, str] | None = (
        None  # Feature encoding schema (required for dict data)
    )
    threshold: float | None = Field(
        default=None, ge=0, le=1, description="Anomaly threshold (0-1)"
    )
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    scaler_type: Literal["robust", "standard"] = "robust"


class AnomalyFitRequest(BaseModel):
    """Anomaly model fitting request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}

    Schema encoding types:
    - numeric: Pass through as-is (int/float)
    - hash: MD5 hash to integer (good for high-cardinality like user_agent)
    - label: Category -> integer mapping (learned from training data)
    - onehot: One-hot encoding (for low-cardinality categoricals)
    - binary: Boolean-like values (yes/no, true/false -> 0/1)
    - frequency: Encode as occurrence frequency from training data

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (varies by backend)
    """

    model: str = "default"  # Model identifier (for caching)
    backend: str = "isolation_forest"  # Backend to use
    data: list[list[float]] | list[dict] | None = None  # Training data
    training_file: str | None = None  # File reference ID from upload-training-data
    schema: dict[str, str] | None = None  # Feature encoding schema
    contamination: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description="Expected proportion of anomalies (0-0.5]",
    )
    epochs: int = Field(
        default=100, ge=1, description="Training epochs (autoencoder only)"
    )
    batch_size: int = Field(
        default=32, ge=1, description="Batch size (autoencoder only)"
    )
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    scaler_type: Literal["robust", "standard"] = "robust"
    # VAE / Early Stopping parameters
    validation_split: float = Field(
        default=0.1, ge=0, le=0.5, description="Fraction of data for validation"
    )
    patience: int = Field(
        default=10, ge=1, description="Epochs without improvement before stopping"
    )
    min_delta: float = Field(
        default=1e-4, ge=0, description="Minimum improvement threshold"
    )


class AnomalyDetectRequest(BaseModel):
    """Anomaly detection request - combines fit and score for convenience."""

    model: str = "default"
    backend: str = "isolation_forest"
    data: list[list[float]] | list[dict]  # Data to analyze
    schema: dict[str, str] | None = None
    contamination: float = Field(default=0.1, gt=0, le=0.5)
    threshold: float | None = Field(default=None, ge=0, le=1)
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    scaler_type: Literal["robust", "standard"] = "robust"
    epochs: int = Field(default=100, ge=1)
    batch_size: int = Field(default=32, ge=1)
    validation_split: float = Field(default=0.1, ge=0, le=0.5)
    patience: int = Field(default=10, ge=1)
    min_delta: float = Field(default=1e-4, ge=0)


class AnomalySaveRequest(BaseModel):
    """Request to save a fitted anomaly model."""

    model: str  # Model identifier (must be fitted)
    backend: str = "isolation_forest"
    normalization: str = (
        "standardization"  # Must match the normalization used during fit
    )
    scaler_type: str = "robust"  # Must match the scaler_type used during fit


class AnomalyLoadRequest(BaseModel):
    """Request to load a pre-trained anomaly model."""

    model: str  # Model identifier to load/cache as
    backend: str = "isolation_forest"


class AnomalyExplainRequest(BaseModel):
    """Anomaly explanation request using SHAP values."""

    model_id: str  # ID of trained anomaly model
    data: list[list[float]]  # Data points to explain
    feature_names: list[str] | None = None  # Optional feature names
    background_samples: int = Field(default=100, ge=1)  # Number of background samples (must be >= 1)
    nsamples: int = Field(default=100, ge=1)  # Number of SHAP samples (must be >= 1)
    backend: str = "isolation_forest"  # Backend used when training
    normalization: str = "standardization"  # Normalization method
    scaler_type: str = "robust"  # Scaler type
