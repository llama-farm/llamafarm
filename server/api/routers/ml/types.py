"""
Pydantic models for ML endpoints.

These models mirror the Universal Runtime's request/response schemas
to provide a consistent API experience.

Note: OCR and Document extraction types have moved to vision/types.py
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# SetFit Classifier Types
# =============================================================================


class ClassifierFitRequest(BaseModel):
    """Request to fit a text classifier."""

    model: str  # Model identifier (for caching/saving)
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    training_data: list[dict[str, str]]  # List of {"text": "...", "label": "..."}
    num_iterations: int = 20
    batch_size: int = 16
    overwrite: bool = (
        True  # If True, overwrite existing; if False, version with timestamp
    )
    description: str | None = None  # Optional model description


class ClassifierPredictRequest(BaseModel):
    """Request to classify texts."""

    model: str  # Model identifier (must be fitted or loaded)
    texts: list[str]


class ClassifierSaveRequest(BaseModel):
    """Request to save a fitted classifier."""

    model: str  # Model identifier (must be fitted)
    description: str | None = None  # Optional model description


class ClassifierLoadRequest(BaseModel):
    """Request to load a pre-trained classifier."""

    model: str  # Model identifier to load


# =============================================================================
# Anomaly Detection Types
# =============================================================================

# All supported anomaly detection backends (powered by PyOD)
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


class AnomalyFitRequest(BaseModel):
    """Anomaly model fitting request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}

    All backends are powered by PyOD. See GET /v1/anomaly/backends for full list.
    Popular choices:
    - isolation_forest: Fast, works well out of the box (recommended legacy)
    - ecod: Fast and parameter-free (recommended for new projects)
    - hbos: Fastest algorithm, good for high dimensions

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (higher = more anomalous)
    """

    model: str = "default"  # Model identifier (for caching)
    backend: AnomalyBackendType = "isolation_forest"  # Algorithm to use
    data: list[list[float]] | list[dict[str, Any]]  # Training data
    schema: dict[str, str] | None = None  # Feature encoding schema
    contamination: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description="Expected proportion of anomalies (0-0.5]",
    )
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    epochs: int = 100  # Training epochs (autoencoder only)
    batch_size: int = 32  # Batch size (autoencoder only)
    overwrite: bool = (
        True  # If True, overwrite existing; if False, version with timestamp
    )
    description: str | None = None  # Optional model description


class AnomalyScoreRequest(BaseModel):
    """Anomaly scoring request.

    All backends are powered by PyOD. See GET /v1/anomaly/backends for full list.

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (higher = more anomalous)
    """

    model: str = "default"  # Model identifier
    backend: AnomalyBackendType = "isolation_forest"  # Algorithm to use
    data: list[list[float]] | list[dict[str, Any]]  # Data points
    schema: dict[str, str] | None = None  # Feature encoding schema
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    threshold: float | None = None  # Override default threshold


class AnomalySaveRequest(BaseModel):
    """Request to save a fitted anomaly model."""

    model: str  # Model identifier (must be fitted)
    backend: AnomalyBackendType = "isolation_forest"
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    description: str | None = None  # Optional model description


class AnomalyLoadRequest(BaseModel):
    """Request to load a pre-trained anomaly model."""

    model: str  # Model identifier to load/cache as
    backend: AnomalyBackendType = "isolation_forest"


# =============================================================================
# Anomaly Backend Info Types (for /v1/anomaly/backends endpoint)
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
    """

    model: str = Field(
        default="default-stream",
        description="Unique identifier for this streaming detector",
    )
    data: dict[str, Any] | list[dict[str, Any]] = Field(
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
        description="Rolling window sizes for automatic feature computation",
    )
    include_lags: bool = Field(
        default=False,
        description="Include lag features in rolling computation",
    )
    lag_periods: list[int] | None = Field(
        default=None,
        description="Lag periods if include_lags is True",
    )


class AnomalyStreamResultItem(BaseModel):
    """Single streaming result item."""

    index: int
    score: float | None = None  # None during cold start
    is_anomaly: bool | None = None  # None during cold start
    raw_score: float | None = None
    samples_until_ready: int


class AnomalyStreamResponse(BaseModel):
    """Streaming anomaly detection response."""

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
# Polars Buffer Types
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
