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
