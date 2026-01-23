"""Anomaly detection types for fit, score, detect, save, and load endpoints.

All anomaly detection is powered by PyOD, providing 12+ algorithms.
Legacy backend names are mapped to PyOD equivalents for backward compatibility.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Backend Type
# =============================================================================

# All supported anomaly detection backends
AnomalyBackendType = Literal[
    # Legacy backends (mapped to PyOD equivalents)
    "isolation_forest",      # PyOD IForest - tree-based ensemble
    "one_class_svm",         # PyOD OCSVM - support vector machine
    "local_outlier_factor",  # PyOD LOF - density-based
    "autoencoder",           # PyOD AutoEncoder - neural network
    # Fast backends (parameter-free or minimal tuning)
    "ecod",   # Empirical CDF - fast, parameter-free
    "hbos",   # Histogram-based - fastest
    "copod",  # Copula-based - fast, parameter-free
    # Distance-based backends
    "knn",    # K-Nearest Neighbors
    "mcd",    # Minimum Covariance Determinant
    # Clustering backend
    "cblof",  # Clustering-Based LOF
    # Ensemble backend
    "suod",   # Scalable ensemble
    # Streaming backend
    "loda",   # Lightweight Online Detector
]


# =============================================================================
# Request Types
# =============================================================================


class AnomalyScoreRequest(BaseModel):
    """Anomaly scoring request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}

    All backends are powered by PyOD. See GET /v1/anomaly/backends for full list.

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (higher = more anomalous)
    """

    model: str = "default"  # Model identifier
    backend: AnomalyBackendType = "isolation_forest"  # Algorithm to use
    data: list[list[float]] | list[dict]  # Data points (numeric arrays or dicts)
    schema_: dict[str, str] | None = Field(
        default=None, alias="schema"
    )  # Feature encoding schema
    threshold: float | None = None  # Override default threshold
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"

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

    All backends are powered by PyOD. See GET /v1/anomaly/backends for full list.
    Popular choices:
    - isolation_forest: Fast, works well out of the box (recommended legacy)
    - ecod: Fast and parameter-free (recommended for new projects)
    - hbos: Fastest algorithm, good for high dimensions
    """

    model: str = "default"  # Model identifier (for caching)
    backend: AnomalyBackendType = "isolation_forest"  # Algorithm to use
    data: list[list[float]] | list[dict]  # Training data (numeric arrays or dicts)
    schema_: dict[str, str] | None = Field(
        default=None, alias="schema"
    )  # Feature encoding schema
    contamination: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description="Expected proportion of anomalies (0-0.5]",
    )
    epochs: int = 100  # Training epochs (autoencoder only)
    batch_size: int = 32  # Batch size (autoencoder only)
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    overwrite: bool = True  # If True, overwrite existing model; if False, version it

    model_config = {"populate_by_name": True}


class AnomalySaveRequest(BaseModel):
    """Anomaly model save request."""

    model: str  # Model identifier
    backend: AnomalyBackendType = "isolation_forest"
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"


class AnomalyLoadRequest(BaseModel):
    """Anomaly model load request."""

    model: str  # Model identifier or path
    backend: AnomalyBackendType = "isolation_forest"


# =============================================================================
# Response Types
# =============================================================================


class AnomalyScoreResult(BaseModel):
    """Single anomaly score result."""

    index: int
    score: float  # Normalized score (0-1 for standardization)
    is_anomaly: bool
    raw_score: float  # PyOD-native raw score


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


# =============================================================================
# Backend Info Types (for /v1/anomaly/backends endpoint)
# =============================================================================


class AnomalyBackendInfo(BaseModel):
    """Information about a single anomaly detection backend."""

    backend: str  # Backend identifier (e.g., "isolation_forest", "ecod")
    name: str  # Human-readable name
    description: str  # What the algorithm does
    category: Literal["legacy", "fast", "distance", "clustering", "ensemble", "streaming", "deep_learning"]
    speed: Literal["very_fast", "fast", "medium", "slow"]
    memory: Literal["low", "medium", "high"]
    parameters: list[str]  # Configurable parameters
    best_for: str  # Use case recommendation
    is_legacy: bool  # True for backward-compatible backends


class AnomalyBackendsResponse(BaseModel):
    """Response for GET /v1/anomaly/backends."""

    object: Literal["list"] = "list"
    data: list[AnomalyBackendInfo]
    total: int
    categories: list[str]  # Available categories
