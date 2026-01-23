"""Classifier types for SetFit-based text classification endpoints."""

from typing import Literal

from pydantic import BaseModel

# =============================================================================
# Request Types
# =============================================================================


class ClassifierFitRequest(BaseModel):
    """Classifier fitting request using SetFit few-shot learning.

    SetFit enables training text classifiers with as few as 8-16 examples per class.
    """

    model: str  # Model identifier (for caching/saving)
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # Base transformer
    training_data: list[dict[str, str]]  # List of {"text": ..., "label": ...}
    num_iterations: int = 20  # Training iterations
    batch_size: int = 16  # Training batch size
    overwrite: bool = True  # If True, overwrite existing model; if False, version it


class ClassifierPredictRequest(BaseModel):
    """Classifier prediction request."""

    model: str  # Model identifier
    texts: list[str]  # Texts to classify


class ClassifierSaveRequest(BaseModel):
    """Classifier save request."""

    model: str  # Model identifier


class ClassifierLoadRequest(BaseModel):
    """Classifier load request."""

    model: str  # Model identifier or path


# =============================================================================
# Response Types
# =============================================================================


class ClassifierPrediction(BaseModel):
    """Single classification prediction."""

    text: str
    label: str
    score: float
    all_scores: dict[str, float]


class ClassifierPredictResponse(BaseModel):
    """Classifier prediction response."""

    object: Literal["list"] = "list"
    data: list[ClassifierPrediction]
    model: str


class ClassifierFitResponse(BaseModel):
    """Classifier fit response."""

    status: str = "success"
    model: str
    samples_fitted: int
    num_classes: int
    labels: list[str]
    training_time_ms: float
    base_model: str
    saved_path: str | None = None  # Path where model was auto-saved


class ClassifierSaveResponse(BaseModel):
    """Classifier save response."""

    status: str = "success"
    model: str
    path: str


class ClassifierLoadResponse(BaseModel):
    """Classifier load response."""

    status: str = "success"
    model: str
    labels: list[str]
    path: str


class ClassifierModelInfo(BaseModel):
    """Information about a saved classifier model."""

    name: str
    base_name: str
    path: str
    created: str
    is_versioned: bool
    labels: list[str] = []
    description: str | None = None


class ClassifierModelsResponse(BaseModel):
    """List of saved classifier models."""

    object: Literal["list"] = "list"
    data: list[ClassifierModelInfo]
    total: int


class ClassifierDeleteResponse(BaseModel):
    """Classifier model deletion response."""

    deleted: bool
    model: str
