"""
ML Router - Endpoints for ML model training and inference.

Provides access to:
- Custom Text Classification (SetFit few-shot learning)
- Anomaly Detection (train and detect anomalies)

Note: OCR and Document extraction have moved to /v1/vision/*
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from server.services.ml_model_service import MLModelService
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
# SetFit Classifier Endpoints
# =============================================================================


@router.post("/classifier/fit")
async def fit_classifier(request: ClassifierFitRequest) -> dict[str, Any]:
    """Fit a text classifier using few-shot learning (SetFit).

    Train a classifier with as few as 8-16 examples per class.
    SetFit uses contrastive learning to fine-tune a sentence-transformer,
    then trains a small classification head.

    Models are automatically saved to ~/.llamafarm/models/classifier/ after training.

    Args:
        model: Base name for the model
        overwrite: If True (default), overwrites existing model with same name
                   If False, creates versioned model {model}_{timestamp}

    Example request:
    ```json
    {
        "model": "intent-classifier",
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "training_data": [
            {"text": "I need to book a flight", "label": "booking"},
            {"text": "Cancel my reservation", "label": "cancellation"},
            {"text": "What's the weather?", "label": "weather"}
        ],
        "num_iterations": 20,
        "overwrite": true
    }
    ```

    Model is automatically saved after fitting - no separate save step needed.
    Use /v1/ml/classifier/predict to classify new texts.
    Use "{model}-latest" in predict/load to get the most recent version.
    """
    # Get versioned model name
    versioned_name = MLModelService.get_versioned_name(request.model, request.overwrite)
    logger.info(f"Training classifier: {request.model} -> {versioned_name}")

    result = await UniversalRuntimeService.classifier_fit(
        model=versioned_name,
        training_data=request.training_data,
        base_model=request.base_model,
        num_iterations=request.num_iterations,
        batch_size=request.batch_size,
    )

    # Save description metadata if provided (model auto-saves during fit)
    if request.description:
        MLModelService.save_description("classifier", versioned_name, request.description)

    # Add versioning info to response
    result["base_name"] = request.model
    result["versioned_name"] = versioned_name
    result["overwrite"] = request.overwrite

    return result


@router.post("/classifier/predict")
async def predict_classifier(request: ClassifierPredictRequest) -> dict[str, Any]:
    """Classify texts using a fitted classifier.

    Supports "-latest" suffix to use the most recent version:
    ```json
    {
        "model": "intent-classifier-latest",
        "texts": ["I want to cancel my trip", "Book me a hotel"]
    }
    ```

    Example request:
    ```json
    {
        "model": "intent-classifier",
        "texts": ["I want to cancel my trip", "Book me a hotel"]
    }
    ```

    Returns predictions with confidence scores for each text.
    """
    # Resolve -latest to actual model name
    resolved_model = MLModelService.resolve_model_name("classifier", request.model)

    return await UniversalRuntimeService.classifier_predict(
        model=resolved_model,
        texts=request.texts,
    )


@router.post("/classifier/save")
async def save_classifier(request: ClassifierSaveRequest) -> dict[str, Any]:
    """Save a fitted classifier to disk for production use.

    After fitting a model with /v1/ml/classifier/fit, save it to disk so it
    persists across server restarts.

    Models are saved to ~/.llamafarm/models/classifier/ with auto-generated
    directory names based on the model name.

    Args:
        model: Model identifier to save
        description: Optional description for the model
    """
    result = await UniversalRuntimeService.classifier_save(model=request.model)

    # Save description metadata if provided (after model is saved to disk)
    if request.description:
        MLModelService.save_description(
            "classifier", request.model, request.description
        )

    return result


@router.post("/classifier/load")
async def load_classifier(request: ClassifierLoadRequest) -> dict[str, Any]:
    """Load a pre-trained classifier from disk.

    Load a previously saved model for production inference without
    re-training.

    Supports "-latest" suffix to load the most recent version:
    ```json
    {
        "model": "intent-classifier-latest"
    }
    ```

    Example request:
    ```json
    {
        "model": "intent-classifier"
    }
    ```
    """
    # Resolve -latest to actual model name
    resolved_model = MLModelService.resolve_model_name("classifier", request.model)

    return await UniversalRuntimeService.classifier_load(model=resolved_model)


@router.get("/classifier/models")
async def list_classifier_models() -> dict[str, Any]:
    """List all saved classifier models available for loading.

    Returns models saved in the classifier models directory with rich metadata.

    Response includes:
    - name: Model name (directory name)
    - base_name: Base model name (without version suffix)
    - path: Full path to the model directory
    - created: ISO timestamp of creation/modification
    - is_versioned: Whether this is a versioned model
    - labels: Class labels (loaded from labels.txt if present)
    - description: Model description (if set)
    """
    models = MLModelService.list_all_models("classifier")

    # Also try to load labels and description for each model
    for model in models:
        labels_path = Path(model["path"]) / "labels.txt"
        if labels_path.exists():
            model["labels"] = labels_path.read_text().strip().split("\n")
        else:
            model["labels"] = []

        # Load description from metadata
        description = MLModelService.get_description("classifier", model["name"])
        if description:
            model["description"] = description

    return {
        "object": "list",
        "data": models,
        "total": len(models),
    }


@router.delete("/classifier/models/{model_name}")
async def delete_classifier_model(model_name: str) -> dict[str, Any]:
    """Delete a saved classifier model.

    Removes the model directory from disk. Does not affect cached models.
    """
    # Validate model name to prevent path traversal
    if "/" in model_name or "\\" in model_name or ".." in model_name:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"Invalid model name: {model_name}")

    return await UniversalRuntimeService.classifier_delete_model(model_name)


# =============================================================================
# Anomaly Detection Endpoints
# =============================================================================


@router.post("/anomaly/fit")
async def fit_anomaly_detector(request: AnomalyFitRequest) -> dict[str, Any]:
    """Fit an anomaly detector on training data.

    Train an anomaly detection model on data assumed to be mostly normal.
    The model learns what "normal" looks like and can then detect deviations.

    Models are automatically saved to ~/.llamafarm/models/anomaly/ after training.

    Args:
        model: Base name for the model
        overwrite: If True (default), overwrites existing model with same name
                   If False, creates versioned model {model}_{timestamp}

    Backends:
    - isolation_forest: Fast, works well out of the box (recommended)
    - one_class_svm: Good for small datasets
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Best for complex patterns, requires more data

    Example request (numeric data):
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]],
        "contamination": 0.1,
        "overwrite": true
    }
    ```

    Model is automatically saved after fitting - no separate save step needed.
    Use /v1/ml/anomaly/score or /v1/ml/anomaly/detect for inference.
    Use "{model}-latest" in score/detect/load to get the most recent version.
    """
    # Get versioned model name
    versioned_name = MLModelService.get_versioned_name(request.model, request.overwrite)
    logger.info(f"Training anomaly detector: {request.model} -> {versioned_name}")

    result = await UniversalRuntimeService.anomaly_fit(
        model=versioned_name,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        contamination=request.contamination,
        normalization=request.normalization,
        epochs=request.epochs,
        batch_size=request.batch_size,
    )

    # Save description metadata if provided (model auto-saves during fit)
    if request.description:
        MLModelService.save_description("anomaly", versioned_name, request.description)

    # Add versioning info to response
    result["base_name"] = request.model
    result["versioned_name"] = versioned_name
    result["overwrite"] = request.overwrite

    return result


@router.post("/anomaly/score")
async def score_anomalies(request: AnomalyScoreRequest) -> dict[str, Any]:
    """Score data points for anomalies.

    Detects anomalies in data using various algorithms.
    Returns all points with their anomaly scores.

    Note: Model must be fitted first via /v1/ml/anomaly/fit or loaded from disk.

    Supports "-latest" suffix to use the most recent version:
    ```json
    {
        "model": "sensor-detector-latest",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [100.0, 200.0]],
        "threshold": 0.5
    }
    ```

    Response includes:
    - score: Anomaly score (0-1, higher = more anomalous)
    - is_anomaly: Boolean based on threshold
    - raw_score: Backend-specific raw score
    """
    # Resolve -latest to actual model name
    resolved_model = MLModelService.resolve_model_name("anomaly", request.model)

    return await UniversalRuntimeService.anomaly_score(
        model=resolved_model,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        normalization=request.normalization,
        threshold=request.threshold,
    )


@router.post("/anomaly/detect")
async def detect_anomalies(request: AnomalyScoreRequest) -> dict[str, Any]:
    """Detect anomalies in data (returns only anomalous points).

    Same as /v1/ml/anomaly/score but filters to return only points
    classified as anomalies.

    Supports "-latest" suffix to use the most recent version.

    Example request:
    ```json
    {
        "model": "sensor-detector-latest",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 200.0]],
        "threshold": 0.5
    }
    ```
    """
    # Resolve -latest to actual model name
    resolved_model = MLModelService.resolve_model_name("anomaly", request.model)

    return await UniversalRuntimeService.anomaly_detect(
        model=resolved_model,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        normalization=request.normalization,
        threshold=request.threshold,
    )


@router.post("/anomaly/save")
async def save_anomaly_model(request: AnomalySaveRequest) -> dict[str, Any]:
    """Save a fitted anomaly model to disk for production use.

    After fitting a model with /v1/ml/anomaly/fit, save it to disk so it
    persists across server restarts.

    Models are saved to ~/.llamafarm/models/anomaly/ with auto-generated
    filenames based on the model name and backend.

    Args:
        model: Model identifier to save
        backend: Backend type used for training
        description: Optional description for the model
    """
    result = await UniversalRuntimeService.anomaly_save(
        model=request.model,
        backend=request.backend,
        normalization=request.normalization,
    )

    # Save description metadata if provided (after model is saved to disk)
    if request.description:
        MLModelService.save_description("anomaly", request.model, request.description)

    return result


@router.post("/anomaly/load")
async def load_anomaly_model(request: AnomalyLoadRequest) -> dict[str, Any]:
    """Load a pre-trained anomaly model from disk.

    Load a previously saved model for production inference without
    re-training.

    Supports "-latest" suffix to load the most recent version:
    ```json
    {
        "model": "sensor-detector-latest",
        "backend": "isolation_forest"
    }
    ```

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest"
    }
    ```
    """
    # Resolve -latest to actual model name
    resolved_model = MLModelService.resolve_model_name("anomaly", request.model)

    return await UniversalRuntimeService.anomaly_load(
        model=resolved_model,
        backend=request.backend,
    )


@router.get("/anomaly/models")
async def list_anomaly_models() -> dict[str, Any]:
    """List all saved anomaly models available for loading.

    Returns models saved in the anomaly models directory with rich metadata.

    Response includes:
    - name: Model name (without extension)
    - filename: Full filename on disk
    - base_name: Base model name (without version suffix)
    - backend: Detected backend type
    - path: Full path to model file
    - size_bytes: File size
    - created: ISO timestamp of creation/modification
    - is_versioned: Whether this is a versioned model
    - description: Model description (if set)
    """
    models = MLModelService.list_all_models("anomaly")

    # Load description for each model
    for model in models:
        description = MLModelService.get_description("anomaly", model["name"])
        if description:
            model["description"] = description

    return {
        "object": "list",
        "data": models,
        "total": len(models),
    }


@router.delete("/anomaly/models/{filename}")
async def delete_anomaly_model(filename: str) -> dict[str, Any]:
    """Delete a saved anomaly model.

    Removes the model file from disk. Does not affect cached models.
    """
    # Validate filename to prevent path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        from fastapi import HTTPException

        raise HTTPException(status_code=400, detail=f"Invalid filename: {filename}")

    return await UniversalRuntimeService.anomaly_delete_model(filename)
