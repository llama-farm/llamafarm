"""
Pydantic models for ML endpoints.

These models mirror the Universal Runtime's request/response schemas
to provide a consistent API experience.

Note: OCR and Document extraction types have moved to vision/types.py
"""

from typing import Any

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
        False  # If False, version with timestamp; if True, overwrite existing
    )


class ClassifierPredictRequest(BaseModel):
    """Request to classify texts."""

    model: str  # Model identifier (must be fitted or loaded)
    texts: list[str]


class ClassifierSaveRequest(BaseModel):
    """Request to save a fitted classifier."""

    model: str  # Model identifier (must be fitted)


class ClassifierLoadRequest(BaseModel):
    """Request to load a pre-trained classifier."""

    model: str  # Model identifier to load


# =============================================================================
# Anomaly Detection Types
# =============================================================================


class AnomalyFitRequest(BaseModel):
    """Anomaly model fitting request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}
    """

    model: str = "default"  # Model identifier (for caching)
    backend: str = "isolation_forest"  # Backend to use
    data: list[list[float]] | list[dict[str, Any]]  # Training data
    schema: dict[str, str] | None = None  # Feature encoding schema
    contamination: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description="Expected proportion of anomalies (0-0.5]",
    )
    epochs: int = 100  # Training epochs (autoencoder only)
    batch_size: int = 32  # Batch size (autoencoder only)
    overwrite: bool = (
        False  # If False, version with timestamp; if True, overwrite existing
    )


class AnomalyScoreRequest(BaseModel):
    """Anomaly scoring request."""

    model: str = "default"  # Model identifier
    backend: str = "isolation_forest"  # Backend
    data: list[list[float]] | list[dict[str, Any]]  # Data points
    schema: dict[str, str] | None = None  # Feature encoding schema
    threshold: float | None = None  # Override default threshold


class AnomalySaveRequest(BaseModel):
    """Request to save a fitted anomaly model."""

    model: str  # Model identifier (must be fitted)
    backend: str = "isolation_forest"


class AnomalyLoadRequest(BaseModel):
    """Request to load a pre-trained anomaly model."""

    model: str  # Model identifier to load/cache as
    backend: str = "isolation_forest"
