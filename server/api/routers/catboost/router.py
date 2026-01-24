"""
CatBoost Router - Endpoints for CatBoost gradient boosting models.

Provides access to:
- Classification and regression training
- Native categorical feature support (no need to encode)
- Incremental learning (add new data without retraining)
- Feature importance analysis
- GPU acceleration (when available)
"""

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/catboost", tags=["catboost"])


# =============================================================================
# Request/Response Types
# =============================================================================


class CatBoostFitRequest(BaseModel):
    """Request to train a CatBoost model."""

    model_id: str | None = Field(
        default=None,
        description="Model identifier (auto-generated if not provided)",
    )
    model_type: Literal["classifier", "regressor"] = Field(
        default="classifier",
        description="Model type: classifier or regressor",
    )
    data: list[list[Any]] = Field(
        ...,
        description="Training features (can include strings for categorical features)",
    )
    labels: list[Any] = Field(
        ...,
        description="Training labels (classes for classifier, values for regressor)",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Names for feature columns",
    )
    cat_features: list[int] | None = Field(
        default=None,
        description="Indices of categorical feature columns (auto-detected if not specified)",
    )
    iterations: int = Field(
        default=100,
        ge=1,
        description="Number of boosting iterations",
    )
    learning_rate: float = Field(
        default=0.1,
        gt=0,
        le=1,
        description="Learning rate",
    )
    depth: int = Field(
        default=6,
        ge=1,
        le=16,
        description="Tree depth",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    validation_fraction: float | None = Field(
        default=None,
        description="Fraction of data to use for validation (0-1)",
    )
    early_stopping_rounds: int | None = Field(
        default=None,
        description="Stop training if validation score doesn't improve for N rounds",
    )


class CatBoostPredictRequest(BaseModel):
    """Request to make predictions."""

    model_id: str = Field(..., description="Model identifier")
    data: list[list[Any]] = Field(
        ...,
        description="Features to predict on",
    )
    return_proba: bool = Field(
        default=False,
        description="Return class probabilities (classifier only)",
    )


class CatBoostUpdateRequest(BaseModel):
    """Request to incrementally update a model."""

    model_id: str = Field(..., description="Model identifier")
    data: list[list[Any]] = Field(
        ...,
        description="New training features",
    )
    labels: list[Any] = Field(
        ...,
        description="New training labels",
    )
    sample_weight: list[float] | None = Field(
        default=None,
        description="Optional sample weights",
    )


class CatBoostLoadRequest(BaseModel):
    """Request to load a model."""

    model_id: str = Field(..., description="Model identifier")


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/info")
async def get_info() -> dict[str, Any]:
    """Get CatBoost availability and capabilities.

    Returns:
    - available: Whether CatBoost is installed
    - version: CatBoost version
    - gpu_available: Whether GPU acceleration is available
    - supported_tasks: List of supported task types
    """
    return await UniversalRuntimeService.catboost_info()


@router.get("/models")
async def list_models() -> dict[str, Any]:
    """List all saved CatBoost models.

    Returns models with metadata:
    - model_id: Model identifier
    - model_type: classifier or regressor
    - path: Path to saved model
    """
    return await UniversalRuntimeService.catboost_list_models()


@router.post("/fit")
async def fit_model(request: CatBoostFitRequest) -> dict[str, Any]:
    """Train a CatBoost model.

    Trains a gradient boosting classifier or regressor. CatBoost
    handles categorical features natively - just pass strings directly.

    Example request:
    ```json
    {
        "model_id": "customer-churn",
        "model_type": "classifier",
        "data": [
            ["premium", 500, 12],
            ["basic", 100, 2],
            ["premium", 750, 24]
        ],
        "labels": [0, 1, 0],
        "feature_names": ["plan", "spend", "tenure"],
        "cat_features": [0],
        "iterations": 100
    }
    ```

    Response includes:
    - model_id: Model identifier
    - model_type: classifier or regressor
    - samples_fitted: Number of samples trained on
    - n_features: Number of features
    - iterations: Actual iterations (may be less with early stopping)
    - best_iteration: Best iteration (with early stopping)
    - classes: Class labels (classifier only)
    - saved_path: Path where model was saved
    - fit_time_ms: Training time
    """
    return await UniversalRuntimeService.catboost_fit(request.model_dump())


@router.post("/predict")
async def predict(request: CatBoostPredictRequest) -> dict[str, Any]:
    """Make predictions with a CatBoost model.

    Example request:
    ```json
    {
        "model_id": "customer-churn",
        "data": [["premium", 600, 18]],
        "return_proba": true
    }
    ```

    Response includes:
    - model_id: Model identifier
    - predictions: List of predictions with:
      - sample_index: Index in input
      - prediction: Predicted class or value
      - probabilities: Class probabilities (if return_proba=true)
    - predict_time_ms: Prediction time
    """
    return await UniversalRuntimeService.catboost_predict(request.model_dump())


@router.post("/update")
async def update_model(request: CatBoostUpdateRequest) -> dict[str, Any]:
    """Incrementally update a CatBoost model with new data.

    Adds new training examples without retraining from scratch.
    Useful for online learning scenarios.

    Example request:
    ```json
    {
        "model_id": "customer-churn",
        "data": [["basic", 50, 1]],
        "labels": [1]
    }
    ```

    Response includes:
    - model_id: Model identifier
    - samples_added: Number of new samples
    - trees_before: Trees before update
    - trees_after: Trees after update
    - update_time_ms: Update time
    """
    return await UniversalRuntimeService.catboost_update(request.model_dump())


@router.post("/load")
async def load_model(request: CatBoostLoadRequest) -> dict[str, Any]:
    """Load a saved CatBoost model.

    Example request:
    ```json
    {
        "model_id": "customer-churn"
    }
    ```

    Response includes:
    - model_id: Model identifier
    - model_type: classifier or regressor
    - n_features: Number of features
    - classes: Class labels (classifier only)
    """
    return await UniversalRuntimeService.catboost_load(request.model_dump())


@router.delete("/{model_id}")
async def delete_model(model_id: str) -> dict[str, Any]:
    """Delete a CatBoost model.

    Removes from cache and disk.

    Args:
        model_id: Model identifier to delete
    """
    # Validate to prevent path traversal
    if "/" in model_id or "\\" in model_id or ".." in model_id:
        raise HTTPException(status_code=400, detail=f"Invalid model ID: {model_id}")

    return await UniversalRuntimeService.catboost_delete(model_id)


@router.get("/{model_id}/importance")
async def get_feature_importance(model_id: str) -> dict[str, Any]:
    """Get feature importance for a CatBoost model.

    Returns features ranked by importance score.

    Args:
        model_id: Model identifier

    Response includes:
    - model_id: Model identifier
    - importances: List of {feature, importance} sorted by importance
    - importance_type: Type of importance computed
    """
    return await UniversalRuntimeService.catboost_importance(model_id)
