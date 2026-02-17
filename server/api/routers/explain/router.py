"""
Explain Router - Endpoints for SHAP model explainability.

Provides SHAP (SHapley Additive exPlanations) for understanding model predictions:
- Local explanations: Why did the model make this specific prediction?
- Global feature importance: Which features are most important overall?

Supports tree, linear, and kernel explainers for different model types.
"""

import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/explain", tags=["explain"])


# =============================================================================
# Request/Response Types
# =============================================================================


class SHAPExplainRequest(BaseModel):
    """Request for SHAP explanation."""

    model_type: str = Field(
        ...,
        description="Type of model: anomaly, classifier, catboost",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    data: list[list[float]] = Field(
        ...,
        description="Data points to explain",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Names for features (improves readability)",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top contributing features to return",
    )
    generate_narrative: bool = Field(
        default=False,
        description="Generate human-readable explanation narrative",
    )


class FeatureImportanceRequest(BaseModel):
    """Request for global feature importance."""

    model_type: str = Field(
        ...,
        description="Type of model: anomaly, classifier, catboost",
    )
    model_id: str = Field(
        ...,
        description="Model identifier",
    )
    data: list[list[float]] = Field(
        ...,
        description="Data to compute importance on",
    )
    feature_names: list[str] | None = Field(
        default=None,
        description="Names for features",
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/explainers")
async def list_explainers() -> dict[str, Any]:
    """List available SHAP explainer types.

    Returns:
    - tree: For tree-based models (RandomForest, XGBoost, CatBoost, IForest)
    - linear: For linear models (LogisticRegression, LinearSVM)
    - kernel: Universal explainer (works with any model, slower)

    The system auto-selects the best explainer based on model type.
    """
    return await UniversalRuntimeService.explain_list_explainers()


@router.post("/shap")
async def explain_shap(request: SHAPExplainRequest) -> dict[str, Any]:
    """Generate SHAP explanation for model predictions.

    Explains why the model made specific predictions by computing
    the contribution of each feature using SHAP values.

    Example request:
    ```json
    {
        "model_type": "anomaly",
        "model_id": "fraud-detector",
        "data": [[100.0, 5, 0.3], [5000.0, 1, 0.9]],
        "feature_names": ["amount", "count", "velocity"],
        "top_k": 3,
        "generate_narrative": true
    }
    ```

    Response includes:
    - explanations: List of per-sample explanations with:
      - sample_index: Index in input data
      - base_value: Model's expected output
      - prediction: Actual prediction for this sample
      - contributions: Top features with SHAP values
    - narrative: Human-readable summary (if requested)
    """
    return await UniversalRuntimeService.explain_shap(request.model_dump())


@router.post("/importance")
async def feature_importance(request: FeatureImportanceRequest) -> dict[str, Any]:
    """Compute global feature importance from SHAP values.

    Returns features ranked by their mean absolute SHAP value,
    indicating their overall importance for predictions.

    Unlike model-specific feature importance, SHAP importance
    is consistent and comparable across different model types.

    Example request:
    ```json
    {
        "model_type": "classifier",
        "model_id": "intent-classifier",
        "data": [[...], [...], [...]],
        "feature_names": ["feature_a", "feature_b", "feature_c"]
    }
    ```

    Response includes:
    - importances: List of {feature, importance} sorted by importance
    - compute_time_ms: Time to compute importance
    """
    return await UniversalRuntimeService.explain_importance(request.model_dump())
