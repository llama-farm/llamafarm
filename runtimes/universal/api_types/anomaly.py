"""Anomaly detection types for fit, score, detect, save, and load endpoints."""

from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Request Types
# =============================================================================


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
    backend: str = "isolation_forest"  # isolation_forest, one_class_svm, local_outlier_factor, autoencoder
    data: list[list[float]] | list[dict]  # Data points (numeric arrays or dicts)
    schema_: dict[str, str] | None = Field(
        default=None, alias="schema"
    )  # Feature encoding schema
    threshold: float | None = None  # Override default threshold
    normalization: str = "standardization"  # standardization, zscore, or raw

    model_config = {"populate_by_name": True}


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
    data: list[list[float]] | list[dict]  # Training data (numeric arrays or dicts)
    schema_: dict[str, str] | None = Field(
        default=None, alias="schema"
    )  # Feature encoding schema
    contamination: float = 0.1  # Expected proportion of anomalies
    epochs: int = 100  # Training epochs (autoencoder only)
    batch_size: int = 32  # Batch size (autoencoder only)
    normalization: str = "standardization"  # standardization, zscore, or raw
    overwrite: bool = True  # If True, overwrite existing model; if False, version it

    model_config = {"populate_by_name": True}


class AnomalySaveRequest(BaseModel):
    """Anomaly model save request."""

    model: str  # Model identifier
    backend: str = "isolation_forest"  # Backend type
    normalization: str = "standardization"  # Score normalization method


class AnomalyLoadRequest(BaseModel):
    """Anomaly model load request."""

    model: str  # Model identifier or path
    backend: str = "isolation_forest"  # Backend type


# =============================================================================
# Response Types
# =============================================================================


class AnomalyScoreResult(BaseModel):
    """Single anomaly score result."""

    index: int
    score: float  # Normalized score (0-1 for standardization)
    is_anomaly: bool
    raw_score: float  # Backend-specific raw score


class AnomalyScoreResponse(BaseModel):
    """Anomaly scoring response."""

    object: Literal["list"] = "list"
    data: list[AnomalyScoreResult]
    model: str
    backend: str
    normalization: str
    threshold: float


class AnomalyFitResponse(BaseModel):
    """Anomaly fit response."""

    status: str = "success"
    model: str
    backend: str
    samples_fitted: int
    training_time_ms: float
    model_params: dict[str, Any]
    encoder_info: dict[str, Any] | None = None  # Info about feature encoding if used
    saved_path: str | None = None  # Path where model was auto-saved


class AnomalySaveResponse(BaseModel):
    """Anomaly save response."""

    status: str = "success"
    model: str
    path: str


class AnomalyLoadResponse(BaseModel):
    """Anomaly load response."""

    status: str = "success"
    model: str
    backend: str
    path: str


class AnomalyModelInfo(BaseModel):
    """Information about a saved anomaly model."""

    name: str
    filename: str
    base_name: str
    backend: str
    path: str
    size_bytes: int
    created: str
    is_versioned: bool
    description: str | None = None


class AnomalyModelsResponse(BaseModel):
    """List of saved anomaly models."""

    object: Literal["list"] = "list"
    data: list[AnomalyModelInfo]
    total: int


class AnomalyDeleteResponse(BaseModel):
    """Anomaly model deletion response."""

    deleted: bool
    model: str
