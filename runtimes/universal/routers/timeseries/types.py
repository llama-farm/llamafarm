"""Pydantic models for TimeSeries endpoints."""

from pydantic import BaseModel

# =============================================================================
# Time-Series Forecasting
# =============================================================================


class TimeSeriesForecastRequest(BaseModel):
    """Time-series forecast request."""

    values: list[float]  # Historical time-series values
    horizon: int = 7  # Number of future steps to forecast
    quantiles: list[float] | None = None  # Quantile levels (default: [0.1, 0.5, 0.9])
    num_samples: int = 20  # Number of samples for uncertainty
    model: str = "amazon/chronos-t5-small"  # HuggingFace model name


class TimeSeriesForecastBatchRequest(BaseModel):
    """Batch time-series forecast request."""

    series: list[list[float]]  # List of time-series
    horizon: int = 7
    quantiles: list[float] | None = None
    num_samples: int = 20
    model: str = "amazon/chronos-t5-small"


# =============================================================================
# Change Point Detection
# =============================================================================


class ChangePointRequest(BaseModel):
    """Change point detection request."""

    values: list[float]  # Time-series values
    n_changepoints: int | None = None  # Exact number (if known)
    penalty: float | None = None  # Penalty for regularization (higher = fewer points)
    algorithm: str = "pelt"  # pelt, binseg, window, bottomup
    model: str = "rbf"  # l1, l2, rbf, normal, ar
    min_size: int = 2  # Minimum segment size


class ChangePointBatchRequest(BaseModel):
    """Batch change point detection request."""

    series: list[list[float]]  # List of time-series
    n_changepoints: int | None = None
    penalty: float | None = None
    algorithm: str = "pelt"
    model: str = "rbf"
    min_size: int = 2


# =============================================================================
# Drift Detection
# =============================================================================


class DriftDetectionRequest(BaseModel):
    """Concept drift detection request."""

    values: list[float]  # Data stream values
    algorithm: str = "adwin"  # adwin, page_hinkley, kswin, ddm
    delta: float | None = None  # Sensitivity parameter (for ADWIN, PageHinkley)
    threshold: float | None = None  # Threshold (for PageHinkley)
    alpha: float | None = None  # Significance level (for KSWIN)
    window_size: int | None = None  # Window size (for KSWIN)


class DriftDetectorCreateRequest(BaseModel):
    """Create drift detector request."""

    algorithm: str = "adwin"
    detector_id: str | None = None
    delta: float | None = None
    threshold: float | None = None
    alpha: float | None = None
    window_size: int | None = None
