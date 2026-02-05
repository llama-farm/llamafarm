"""SHAP Explainability router.

Provides endpoints for generating SHAP explanations for ML model predictions.
Supports tree, linear, and kernel explainers for different model types.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException

from api_types.explain import (
    ExplainerInfo,
    ExplainersResponse,
    FeatureContribution,
    FeatureImportance,
    FeatureImportanceRequest,
    FeatureImportanceResponse,
    NarrativeExplanation,
    SHAPExplainRequest,
    SHAPExplainResponse,
    SHAPExplanation,
)
from models.shap_explainer import (
    SHAPExplainer,
    get_explainer_types,
)
from services.error_handler import handle_endpoint_errors

logger = logging.getLogger(__name__)

router = APIRouter()

# Type for model getter function
ModelGetterFunc = Callable[[str, str], Awaitable[Any]]

# Module-level state (set by server.py)
_model_getter: ModelGetterFunc | None = None
_model_load_lock: asyncio.Lock | None = None


def set_model_getter(getter: ModelGetterFunc) -> None:
    """Set the function to get models for explanation.

    Args:
        getter: Async function that takes (model_type, model_id) and returns model
    """
    global _model_getter
    _model_getter = getter


def set_explain_state(lock: asyncio.Lock) -> None:
    """Set shared state from server.py."""
    global _model_load_lock
    _model_load_lock = lock


@router.get("/v1/explain/explainers")
@handle_endpoint_errors("explain-explainers")
async def list_explainers() -> ExplainersResponse:
    """List available SHAP explainer types."""
    explainers = get_explainer_types()
    return ExplainersResponse(
        explainers=[
            ExplainerInfo(
                name=e["name"],
                description=e["description"],
                supported_models=e["supported_models"],
            )
            for e in explainers
        ]
    )


@router.post("/v1/explain/shap")
@handle_endpoint_errors("explain-shap")
async def explain_shap(request: SHAPExplainRequest) -> SHAPExplainResponse:
    """Generate SHAP explanation for model predictions.

    This endpoint loads the specified model and computes SHAP values
    to explain how each feature contributed to the prediction.
    """
    if _model_getter is None:
        raise HTTPException(
            status_code=500,
            detail="Explainer not initialized. Model getter not set.",
        )

    start_time = time.time()

    try:
        # Get the model
        model = await _model_getter(request.model_type, request.model_id)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model_type}/{request.model_id}",
            )

        # Get the underlying model object for SHAP
        # Different model types have different ways to access the model
        underlying_model = None
        if hasattr(model, "_model"):
            underlying_model = model._model
        elif hasattr(model, "_detector"):
            underlying_model = model._detector
        elif hasattr(model, "model"):
            underlying_model = model.model
        else:
            underlying_model = model

        if underlying_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model does not have an explainable component",
            )

        # Convert data to numpy
        data = np.array(request.data, dtype=np.float32)

        # Create background data from the input if needed
        # For kernel and linear explainers, we need at least 2 samples for background data
        background_data = data if len(data) > 1 else None

        # Note: SHAP explainer type is auto-detected based on model type
        # Tree models (IForest, CatBoost, XGBoost) use TreeExplainer which doesn't need background data
        # For non-tree models, we'll need background data if using Kernel/Linear explainers

        # Create explainer
        explainer = SHAPExplainer(
            model=underlying_model,
            feature_names=request.feature_names,
            background_data=background_data,
        )

        await explainer.load()

        # Compute SHAP values
        explanations_raw = await explainer.explain(data, top_k=request.top_k)

        # Convert to API types
        explanations = []
        for exp in explanations_raw:
            contributions = [
                FeatureContribution(
                    feature=c.feature,
                    value=c.value,
                    shap_value=c.shap_value,
                    direction=c.direction,
                )
                for c in exp.contributions
            ]
            explanations.append(
                SHAPExplanation(
                    sample_index=exp.sample_index,
                    base_value=exp.base_value,
                    prediction=exp.prediction,
                    contributions=contributions,
                )
            )

        # Generate narrative if requested
        narrative = None
        if request.generate_narrative and explanations:
            # Use first explanation for narrative
            narrative_raw = await explainer.generate_narrative(explanations_raw[0])
            narrative = NarrativeExplanation(
                summary=narrative_raw.summary,
                details=narrative_raw.details,
            )

        await explainer.unload()

        explain_time_ms = (time.time() - start_time) * 1000

        return SHAPExplainResponse(
            model_type=request.model_type,
            model_id=request.model_id,
            explainer_type=explainer.explainer_type,
            explanations=explanations,
            narrative=narrative,
            explain_time_ms=explain_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SHAP explanation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate SHAP explanation: {str(e)}",
        ) from e


@router.post("/v1/explain/importance")
@handle_endpoint_errors("explain-importance")
async def feature_importance(request: FeatureImportanceRequest) -> FeatureImportanceResponse:
    """Compute global feature importance from SHAP values.

    Returns features ranked by their mean absolute SHAP value,
    indicating their overall importance for predictions.
    """
    if _model_getter is None:
        raise HTTPException(
            status_code=500,
            detail="Explainer not initialized. Model getter not set.",
        )

    start_time = time.time()

    try:
        # Get the model
        model = await _model_getter(request.model_type, request.model_id)
        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model not found: {request.model_type}/{request.model_id}",
            )

        # Get underlying model
        underlying_model = None
        if hasattr(model, "_model"):
            underlying_model = model._model
        elif hasattr(model, "_detector"):
            underlying_model = model._detector
        elif hasattr(model, "model"):
            underlying_model = model.model
        else:
            underlying_model = model

        if underlying_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model does not have an explainable component",
            )

        # Convert data
        data = np.array(request.data, dtype=np.float32)

        # Create explainer
        explainer = SHAPExplainer(
            model=underlying_model,
            feature_names=request.feature_names,
            background_data=data,
        )

        await explainer.load()

        # Compute importance
        importance_raw = await explainer.get_feature_importance(data)

        await explainer.unload()

        compute_time_ms = (time.time() - start_time) * 1000

        return FeatureImportanceResponse(
            model_type=request.model_type,
            model_id=request.model_id,
            importances=[
                FeatureImportance(feature=name, importance=imp)
                for name, imp in importance_raw
            ],
            compute_time_ms=compute_time_ms,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature importance computation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compute feature importance: {str(e)}",
        ) from e
