"""
Request/response types for anomaly detection endpoints.
"""

from pydantic import BaseModel


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
    schema: dict[str, str] | None = (
        None  # Feature encoding schema (required for dict data)
    )
    threshold: float | None = None  # Override default threshold
    normalization: str = "standardization"  # standardization, zscore, or raw


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
    schema: dict[str, str] | None = (
        None  # Feature encoding schema (required for dict data)
    )
    contamination: float = 0.1  # Expected proportion of anomalies
    epochs: int = 100  # Training epochs (autoencoder only)
    batch_size: int = 32  # Batch size (autoencoder only)
    normalization: str = "standardization"  # standardization, zscore, or raw


class AnomalySaveRequest(BaseModel):
    """Request to save a fitted anomaly model."""

    model: str  # Model identifier (must be fitted)
    backend: str = "isolation_forest"
    normalization: str = (
        "standardization"  # Must match the normalization used during fit
    )
    # Note: filename is auto-generated from model name, no user control over paths


class AnomalyLoadRequest(BaseModel):
    """Request to load a pre-trained anomaly model."""

    model: str  # Model identifier to load/cache as
    backend: str = "isolation_forest"
    # Note: filename is derived from model name, no user control over paths
