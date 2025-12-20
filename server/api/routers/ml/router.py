"""
ML Router - Endpoints for ML model training and inference.

Provides access to:
- Custom Text Classification (SetFit few-shot learning)
- Anomaly Detection (train and detect anomalies)

Note: OCR and Document extraction have moved to /v1/vision/*
"""

import logging
from typing import Any

from fastapi import APIRouter
from server.services.universal_runtime_service import UniversalRuntimeService

from .types import (
    AnomalyFitRequest,
    AnomalyLoadRequest,
    AnomalySaveRequest,
    AnomalyScoreRequest,
    ClassifierFitRequest,
    ClassifierLoadRequest,
    ClassifierPredictRequest,
    ClassifierSaveRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


# =============================================================================
# Health Check
# =============================================================================


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Check Universal Runtime health.

    Returns the health status of the Universal Runtime service.
    """
    return await UniversalRuntimeService.health_check()


# =============================================================================
# SetFit Classifier Endpoints
# =============================================================================


@router.post("/classifier/fit")
async def fit_classifier(request: ClassifierFitRequest) -> dict[str, Any]:
    """Fit a text classifier using few-shot learning (SetFit).

    Train a classifier with as few as 8-16 examples per class.
    """
    return await UniversalRuntimeService.classifier_fit(
        model=request.model,
        training_data=request.training_data,
        base_model=request.base_model,
        num_iterations=request.num_iterations,
        batch_size=request.batch_size,
    )


@router.post("/classifier/predict")
async def predict_classifier(request: ClassifierPredictRequest) -> dict[str, Any]:
    """Classify texts using a fitted classifier."""
    return await UniversalRuntimeService.classifier_predict(
        model=request.model,
        texts=request.texts,
    )


@router.post("/classifier/save")
async def save_classifier(request: ClassifierSaveRequest) -> dict[str, Any]:
    """Save a fitted classifier to disk for production use."""
    return await UniversalRuntimeService.classifier_save(model=request.model)


@router.post("/classifier/load")
async def load_classifier(request: ClassifierLoadRequest) -> dict[str, Any]:
    """Load a pre-trained classifier from disk."""
    return await UniversalRuntimeService.classifier_load(model=request.model)


@router.get("/classifier/models")
async def list_classifier_models() -> dict[str, Any]:
    """List all saved classifier models available for loading."""
    return await UniversalRuntimeService.classifier_list_models()


@router.delete("/classifier/models/{model_name}")
async def delete_classifier_model(model_name: str) -> dict[str, Any]:
    """Delete a saved classifier model."""
    from fastapi import HTTPException

    if "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

    return await UniversalRuntimeService.classifier_delete_model(model_name)


# =============================================================================
# Anomaly Detection Endpoints
# =============================================================================


@router.post("/anomaly/fit")
async def fit_anomaly_detector(request: AnomalyFitRequest) -> dict[str, Any]:
    """Fit an anomaly detector on training data."""
    return await UniversalRuntimeService.anomaly_fit(
        model=request.model,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        contamination=request.contamination,
        epochs=request.epochs,
        batch_size=request.batch_size,
    )


@router.post("/anomaly/score")
async def score_anomalies(request: AnomalyScoreRequest) -> dict[str, Any]:
    """Score data points for anomalies."""
    return await UniversalRuntimeService.anomaly_score(
        model=request.model,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        threshold=request.threshold,
    )


@router.post("/anomaly/detect")
async def detect_anomalies(request: AnomalyScoreRequest) -> dict[str, Any]:
    """Detect anomalies in data (returns only anomalous points)."""
    return await UniversalRuntimeService.anomaly_detect(
        model=request.model,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        threshold=request.threshold,
    )


@router.post("/anomaly/save")
async def save_anomaly_model(request: AnomalySaveRequest) -> dict[str, Any]:
    """Save a fitted anomaly model to disk for production use."""
    return await UniversalRuntimeService.anomaly_save(
        model=request.model,
        backend=request.backend,
    )


@router.post("/anomaly/load")
async def load_anomaly_model(request: AnomalyLoadRequest) -> dict[str, Any]:
    """Load a pre-trained anomaly model from disk."""
    return await UniversalRuntimeService.anomaly_load(
        model=request.model,
        backend=request.backend,
    )


@router.get("/anomaly/models")
async def list_anomaly_models() -> dict[str, Any]:
    """List all saved anomaly models available for loading."""
    return await UniversalRuntimeService.anomaly_list_models()


@router.delete("/anomaly/models/{filename}")
async def delete_anomaly_model(filename: str) -> dict[str, Any]:
    """Delete a saved anomaly model."""
    from fastapi import HTTPException

    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename}")

    return await UniversalRuntimeService.anomaly_delete_model(filename)
