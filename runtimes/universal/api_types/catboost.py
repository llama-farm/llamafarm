"""Pydantic models for CatBoost classification/regression endpoints.

CatBoost provides gradient boosting with unique features:
- Native categorical support: No one-hot encoding needed
- Incremental learning: Update model without full retrain
- GPU acceleration: Fast training on NVIDIA GPUs

Use Cases:
- Tabular data classification and regression
- Real-time model updates with streaming data
- Handling mixed numeric/categorical features
"""

from typing import Any

from pydantic import BaseModel, Field


class CatBoostFitRequest(BaseModel):
    """Request to train a CatBoost model."""

    model_id: str | None = Field(
        default=None,
        description="Model identifier. Auto-generated if not provided.",
    )
    model_type: str = Field(
        default="classifier",
        description="Type of model: 'classifier' or 'regressor'",
    )
    data: list[list[float | str]] = Field(
        ...,
        description="Training data (n_samples x n_features). Can include categorical values as strings.",
    )
    labels: list[int | float | str] = Field(
        ...,
        description="Training labels. Integers/strings for classification, floats for regression.",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Optional feature names",
    )
    cat_features: list[int] | list[str] | None = Field(
        default=None,
        description="Indices or names of categorical features",
    )
    iterations: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Number of boosting iterations",
    )
    learning_rate: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Learning rate for gradient descent",
    )
    depth: int = Field(
        default=6,
        ge=1,
        le=16,
        description="Depth of individual trees",
    )
    early_stopping_rounds: int | None = Field(
        default=None,
        ge=1,
        description="Stop if no improvement for N rounds (requires validation set)",
    )
    validation_fraction: float | None = Field(
        default=None,
        gt=0,
        lt=1,
        description="Fraction of data to use for validation (for early stopping)",
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility",
    )


class CatBoostFitResponse(BaseModel):
    """Response from CatBoost training."""

    model_id: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Type of model (classifier/regressor)")
    samples_fitted: int = Field(..., description="Number of training samples")
    n_features: int = Field(..., description="Number of features")
    iterations: int = Field(..., description="Number of trees in the model")
    best_iteration: int | None = Field(
        default=None,
        description="Best iteration if early stopping was used",
    )
    classes: list[Any] | None = Field(
        default=None,
        description="Class labels for classifier",
    )
    saved_path: str = Field(..., description="Path where model was saved")
    fit_time_ms: float = Field(..., description="Training time in milliseconds")


class CatBoostPredictRequest(BaseModel):
    """Request for CatBoost prediction."""

    model_id: str = Field(..., description="Model identifier")
    data: list[list[float | str]] = Field(
        ...,
        description="Data to predict (n_samples x n_features)",
    )
    return_proba: bool = Field(
        default=False,
        description="Return class probabilities instead of labels (classifier only)",
    )


class CatBoostPrediction(BaseModel):
    """Single prediction result."""

    sample_index: int = Field(..., description="Index in the input data")
    prediction: int | float | str = Field(..., description="Predicted label or value")
    probabilities: dict[str, float] | None = Field(
        default=None,
        description="Class probabilities if return_proba=True",
    )


class CatBoostPredictResponse(BaseModel):
    """Response from CatBoost prediction."""

    model_id: str = Field(..., description="Model identifier")
    predictions: list[CatBoostPrediction] = Field(..., description="Predictions for each sample")
    predict_time_ms: float = Field(..., description="Prediction time in milliseconds")


class CatBoostUpdateRequest(BaseModel):
    """Request for incremental model update."""

    model_id: str = Field(..., description="Model identifier")
    data: list[list[float | str]] = Field(
        ...,
        description="New training data (n_samples x n_features)",
    )
    labels: list[int | float | str] = Field(
        ...,
        description="New training labels",
    )
    sample_weight: list[float] | None = Field(
        default=None,
        description="Optional sample weights",
    )


class CatBoostUpdateResponse(BaseModel):
    """Response from incremental update."""

    model_id: str = Field(..., description="Model identifier")
    samples_added: int = Field(..., description="Number of new samples used")
    trees_before: int = Field(..., description="Number of trees before update")
    trees_after: int = Field(..., description="Number of trees after update")
    update_time_ms: float = Field(..., description="Update time in milliseconds")


class CatBoostSaveRequest(BaseModel):
    """Request to save a CatBoost model."""

    model_id: str = Field(..., description="Model identifier to save")


class CatBoostSaveResponse(BaseModel):
    """Response from saving a model."""

    model_id: str = Field(..., description="Model identifier")
    saved_path: str = Field(..., description="Path where model was saved")


class CatBoostLoadRequest(BaseModel):
    """Request to load a CatBoost model."""

    model_id: str = Field(..., description="Model identifier to load")


class CatBoostLoadResponse(BaseModel):
    """Response from loading a model."""

    model_id: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Type of model (classifier/regressor)")
    n_features: int = Field(..., description="Number of features")
    classes: list[Any] | None = Field(default=None, description="Class labels for classifier")


class CatBoostModelInfo(BaseModel):
    """Information about a saved CatBoost model."""

    model_id: str = Field(..., description="Model identifier")
    model_type: str = Field(..., description="Type of model")
    n_features: int | None = Field(default=None, description="Number of features")
    iterations: int | None = Field(default=None, description="Number of trees")
    path: str = Field(..., description="Model file path")


class CatBoostModelsResponse(BaseModel):
    """Response listing available CatBoost models."""

    models: list[CatBoostModelInfo] = Field(..., description="List of saved models")


class CatBoostDeleteResponse(BaseModel):
    """Response from deleting a model."""

    model_id: str = Field(..., description="Model identifier that was deleted")
    deleted: bool = Field(..., description="Whether deletion was successful")


class CatBoostFeatureImportance(BaseModel):
    """Feature importance entry."""

    feature: str = Field(..., description="Feature name")
    importance: float = Field(..., description="Importance score")


class CatBoostFeatureImportanceResponse(BaseModel):
    """Response with feature importance."""

    model_id: str = Field(..., description="Model identifier")
    importances: list[CatBoostFeatureImportance] = Field(
        ...,
        description="Feature importances sorted by importance",
    )
    importance_type: str = Field(
        default="FeatureImportance",
        description="Type of importance computed",
    )


class CatBoostInfoResponse(BaseModel):
    """Response with CatBoost availability and capabilities."""

    available: bool = Field(..., description="Whether CatBoost is installed")
    gpu_available: bool = Field(default=False, description="Whether GPU training is available")
    model_types: list[str] = Field(
        default_factory=lambda: ["classifier", "regressor"],
        description="Supported model types",
    )
    features: list[str | None] = Field(
        default_factory=list,
        description="Available features",
    )
    error: str | None = Field(default=None, description="Error message if not available")
