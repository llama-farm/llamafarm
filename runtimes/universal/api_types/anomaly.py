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


class PolarsConfig(BaseModel):
    """Configuration for Polars-based feature engineering.

    When enabled, data is processed through a Polars buffer that can compute
    rolling statistics and lag features automatically.

    Example:
        polars_config = {
            "enabled": True,
            "rolling_windows": [5, 10],
            "include_rolling_stats": ["mean", "std"],
            "include_lags": True,
            "lag_periods": [1, 2]
        }
    """

    enabled: bool = Field(
        default=False,
        description="Enable Polars-based feature engineering",
    )
    rolling_windows: list[int] | None = Field(
        default=None,
        description="Rolling window sizes (e.g., [5, 10, 20])",
    )
    include_rolling_stats: list[Literal["mean", "std", "min", "max"]] | None = Field(
        default=None,
        description="Which rolling statistics to compute (default: all)",
    )
    include_lags: bool = Field(
        default=False,
        description="Include lag features",
    )
    lag_periods: list[int] | None = Field(
        default=None,
        description="Lag periods to compute (e.g., [1, 2, 3])",
    )


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

    Polars Integration:
    - Set polars_config.enabled=True to use Polars for feature engineering
    - Configure rolling_windows for automatic rolling statistics
    - Configure lag_periods for temporal features
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
    # Polars configuration
    polars_config: PolarsConfig | None = Field(
        default=None,
        description="Optional Polars-based feature engineering configuration",
    )

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


# =============================================================================
# Streaming Anomaly Detection Types
# =============================================================================


class AnomalyStreamRequest(BaseModel):
    """Streaming anomaly detection request.

    Process data through an auto-rolling detector that:
    - Uses Polars internally as the data substrate (automatic)
    - Computes rolling features if configured
    - Auto-retrains after retrain_interval samples
    - Maintains a sliding window of window_size samples

    The detector handles cold start automatically - collecting min_samples
    before the first model training.

    Data format:
    - Single point: data = {"amount": 100.0, "count": 5}
    - Batch: data = [{"amount": 100.0}, {"amount": 200.0}]
    """

    model: str = Field(
        default="default-stream",
        description="Unique identifier for this streaming detector",
    )
    data: dict | list[dict] = Field(
        ...,
        description="Single data point or batch of data points",
    )
    backend: AnomalyBackendType = Field(
        default="ecod",
        description="PyOD backend (ecod recommended for streaming)",
    )
    min_samples: int = Field(
        default=50,
        ge=10,
        description="Minimum samples before first model training (cold start)",
    )
    retrain_interval: int = Field(
        default=100,
        ge=10,
        description="Retrain model after this many new samples",
    )
    window_size: int = Field(
        default=1000,
        ge=50,
        description="Sliding window size (keeps most recent N samples)",
    )
    contamination: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description="Expected proportion of anomalies",
    )
    threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Anomaly score threshold",
    )
    # Optional rolling feature configuration
    rolling_windows: list[int] | None = Field(
        default=None,
        description="Rolling window sizes for automatic feature computation (e.g., [5, 10, 20])",
    )
    include_lags: bool = Field(
        default=False,
        description="Include lag features in rolling computation",
    )
    lag_periods: list[int] | None = Field(
        default=None,
        description="Lag periods if include_lags is True (e.g., [1, 2, 3])",
    )


class AnomalyStreamResultItem(BaseModel):
    """Single streaming result item."""

    index: int
    score: float | None = None  # None during cold start
    is_anomaly: bool | None = None  # None during cold start
    raw_score: float | None = None
    samples_until_ready: int


class AnomalyStreamResponse(BaseModel):
    """Streaming anomaly detection response.

    Status values:
    - collecting: Cold start phase, collecting min_samples
    - ready: Model trained and scoring data
    - retraining: Background retraining in progress (still scoring)
    """

    object: Literal["streaming_result"] = "streaming_result"
    model: str
    status: Literal["collecting", "ready", "retraining"]
    results: list[AnomalyStreamResultItem]
    model_version: int
    samples_collected: int
    samples_until_ready: int
    threshold: float


class AnomalyStreamDetectorInfo(BaseModel):
    """Information about a streaming detector."""

    model_id: str
    backend: str
    status: Literal["collecting", "ready", "retraining"]
    model_version: int
    samples_collected: int
    total_processed: int
    samples_since_retrain: int
    min_samples: int
    retrain_interval: int
    window_size: int
    threshold: float
    is_ready: bool


class AnomalyStreamDetectorsResponse(BaseModel):
    """List of active streaming detectors."""

    object: Literal["list"] = "list"
    data: list[AnomalyStreamDetectorInfo]
    total: int


# =============================================================================
# Polars Buffer Types (for direct buffer access)
# =============================================================================


class PolarsBufferCreateRequest(BaseModel):
    """Request to create a named Polars buffer."""

    buffer_id: str = Field(..., description="Unique buffer identifier")
    window_size: int = Field(
        default=1000,
        ge=10,
        description="Maximum records to keep (sliding window)",
    )


class PolarsBufferAppendRequest(BaseModel):
    """Request to append data to a Polars buffer."""

    buffer_id: str = Field(..., description="Buffer identifier")
    data: dict | list[dict] = Field(
        ...,
        description="Single record or batch of records to append",
    )


class PolarsBufferFeaturesRequest(BaseModel):
    """Request to compute features from a Polars buffer."""

    buffer_id: str = Field(..., description="Buffer identifier")
    rolling_windows: list[int] | None = Field(
        default=None,
        description="Rolling window sizes (default: [5, 10, 20])",
    )
    include_rolling_stats: list[Literal["mean", "std", "min", "max"]] | None = Field(
        default=None,
        description="Which rolling stats to compute (default: all)",
    )
    include_lags: bool = Field(
        default=True,
        description="Include lag features",
    )
    lag_periods: list[int] | None = Field(
        default=None,
        description="Lag periods (default: [1, 2, 3])",
    )
    tail: int | None = Field(
        default=None,
        description="Return only last N rows (optional)",
    )


class PolarsBufferStats(BaseModel):
    """Statistics about a Polars buffer."""

    buffer_id: str
    size: int
    window_size: int
    columns: list[str]
    numeric_columns: list[str]
    memory_bytes: int
    append_count: int
    avg_append_ms: float


class PolarsBufferDataResponse(BaseModel):
    """Response containing buffer data."""

    object: Literal["polars_data"] = "polars_data"
    buffer_id: str
    rows: int
    columns: list[str]
    data: list[dict]


class PolarsBuffersListResponse(BaseModel):
    """List of active Polars buffers."""

    object: Literal["list"] = "list"
    data: list[PolarsBufferStats]
    total: int
