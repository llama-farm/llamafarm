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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from server.services.ml_model_service import MLModelService
from server.services.router_storage_service import RouterStorageService
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
    RouterGenerateDataRequest,
    RouterLoadRequest,
    RouterRouteRequest,
    RouterTrainRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


def _validate_model_name(name: str, param_name: str = "model name") -> None:
    """Validate a model name to prevent path traversal attacks.

    Args:
        name: The model name to validate
        param_name: Name of the parameter for error messages

    Raises:
        HTTPException: If the name contains invalid characters
    """
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(status_code=400, detail=f"Invalid {param_name}: {name}")


# =============================================================================
# SetFit Classifier Endpoints
# =============================================================================


@router.post("/classifier/fit")
async def fit_classifier(request: ClassifierFitRequest) -> dict[str, Any]:
    """Fit a text classifier using few-shot learning (SetFit).

    Train a classifier with as few as 8-16 examples per class.
    SetFit uses contrastive learning to fine-tune a sentence-transformer,
    then trains a small classification head.

    Models are stored in ~/.llamafarm/models/classifier/

    Args:
        model: Base name for the model
        overwrite: If False (default), creates versioned model {model}_{timestamp}
                   If True, overwrites existing model with same name

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
        "overwrite": false
    }
    ```

    After fitting, use /v1/ml/classifier/save to persist the model
    (with optional description).
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
    _validate_model_name(model_name)
    return await UniversalRuntimeService.classifier_delete_model(model_name)


# =============================================================================
# Anomaly Detection Endpoints
# =============================================================================


@router.post("/anomaly/fit")
async def fit_anomaly_detector(request: AnomalyFitRequest) -> dict[str, Any]:
    """Fit an anomaly detector on training data.

    Train an anomaly detection model on data assumed to be mostly normal.
    The model learns what "normal" looks like and can then detect deviations.

    Models are stored in ~/.llamafarm/models/anomaly/

    Args:
        model: Base name for the model
        overwrite: If False (default), creates versioned model {model}_{timestamp}
                   If True, overwrites existing model with same name

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
        "overwrite": false
    }
    ```

    After fitting, use /v1/ml/anomaly/save to persist the model
    (with optional description).
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
    _validate_model_name(filename, "filename")
    return await UniversalRuntimeService.anomaly_delete_model(filename)


# =============================================================================
# Semantic Router Endpoints
# =============================================================================


@router.post("/router/train")
async def train_router(request: RouterTrainRequest) -> dict[str, Any]:
    """Train a semantic router with routes and utterances.

    Creates a router that matches queries to target models based on
    semantic similarity to example utterances.

    Routes are automatically saved to the project directory after training.

    Example request:
    ```json
    {
        "model": "customer-router",
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "default_model": "general-assistant",
        "similarity_threshold": 0.7,
        "namespace": "default",
        "project_id": "my_project",
        "routes": [
            {
                "name": "billing",
                "target_model": "billing-specialist",
                "utterances": ["what is my bill", "payment question"]
            },
            {
                "name": "support",
                "target_model": "tech-support",
                "utterances": ["help with login", "password reset"]
            }
        ]
    }
    ```
    """
    logger.info(
        f"Training router: {request.model} "
        f"(namespace={request.namespace}, project={request.project_id})"
    )

    # Get project-specific storage path
    storage_path = str(
        RouterStorageService.get_router_dir(
            request.namespace, request.project_id, request.model
        )
    )

    # Convert routes to dict format for service
    routes_data = [
        {
            "name": route.name,
            "target_model": route.target_model,
            "description": route.description,
            "utterances": route.utterances,
        }
        for route in request.routes
    ]

    result = await UniversalRuntimeService.router_train(
        model=request.model,
        routes=routes_data,
        embedder_model=request.embedder_model,
        default_model=request.default_model,
        similarity_threshold=request.similarity_threshold,
        storage_path=storage_path,
    )

    # Add project context to response
    result["namespace"] = request.namespace
    result["project_id"] = request.project_id
    result["storage_path"] = storage_path

    return result


@router.post("/router/route")
async def route_query(request: RouterRouteRequest) -> dict[str, Any]:
    """Route a query to the appropriate target model.

    Returns the routing decision including:
    - target_model: Model to route to
    - route_name: Matched route (or null for default)
    - similarity_score: Confidence of the match
    - matched_utterance: Best matching training example
    - router_name: Name of the router that handled the request
    - namespace: Project namespace
    - project_id: Project ID

    Example request:
    ```json
    {
        "model": "customer-router",
        "query": "I need help with my account balance",
        "namespace": "default",
        "project_id": "my_project"
    }
    ```
    """
    # Get project-specific storage path
    storage_path = str(
        RouterStorageService.get_router_dir(
            request.namespace, request.project_id, request.model
        )
    )

    result = await UniversalRuntimeService.router_route(
        model=request.model,
        query=request.query,
        storage_path=storage_path,
    )

    # Add routing metadata for clients
    result["router_name"] = request.model
    result["namespace"] = request.namespace
    result["project_id"] = request.project_id

    # Log routing decision for observability
    logger.info(
        "Routing decision",
        extra={
            "router_name": request.model,
            "target_model": result.get("target_model"),
            "route_name": result.get("route_name"),
            "similarity_score": result.get("similarity_score"),
            "query_preview": (
                request.query[:50] + "..."
                if len(request.query) > 50
                else request.query
            ),
            "namespace": request.namespace,
            "project_id": request.project_id,
        },
    )

    return result


@router.post("/router/load")
async def load_router(request: RouterLoadRequest) -> dict[str, Any]:
    """Load a saved router into memory.

    Routers are automatically loaded on first use, but this endpoint
    allows pre-loading for faster first requests.

    Example request:
    ```json
    {
        "model": "customer-router",
        "namespace": "default",
        "project_id": "my_project"
    }
    ```
    """
    # Get project-specific storage path
    storage_path = str(
        RouterStorageService.get_router_dir(
            request.namespace, request.project_id, request.model
        )
    )

    return await UniversalRuntimeService.router_load(
        model=request.model,
        storage_path=storage_path,
    )


class RouterListModelsRequest(BaseModel):
    """Request to list router models with project context."""

    namespace: str = "default"
    project_id: str = "default"


class RouterDeleteRequest(BaseModel):
    """Request to delete a router with project context."""

    namespace: str = "default"
    project_id: str = "default"


@router.get("/router/models")
async def list_router_models() -> dict[str, Any]:
    """List all saved router models (global/legacy).

    Returns routers saved in the global router models directory.
    For project-specific routers, use POST /router/models/list
    with namespace/project_id.

    Returns:
    - name: Router model name
    - num_routes: Number of routes configured
    - routes: List of route names
    - embedder_model: Embedding model used
    - default_model: Fallback model
    - similarity_threshold: Matching threshold
    """
    return await UniversalRuntimeService.router_list_models()


@router.post("/router/models/list")
async def list_router_models_project(
    request: RouterListModelsRequest,
) -> dict[str, Any]:
    """List all saved router models in a project.

    Example request:
    ```json
    {
        "namespace": "default",
        "project_id": "my_project"
    }
    ```

    Returns routers saved in the project directory with metadata:
    - name: Router model name
    - path: Storage path
    - has_embeddings: Whether embeddings file exists
    - config: Full router configuration
    """
    routers = RouterStorageService.list_routers(request.namespace, request.project_id)

    return {
        "object": "list",
        "data": routers,
        "total": len(routers),
        "namespace": request.namespace,
        "project_id": request.project_id,
    }


@router.delete("/router/models/{model_name}")
async def delete_router_model(model_name: str) -> dict[str, Any]:
    """Delete a saved router model (global/legacy).

    Removes the router from the global storage. Does not affect cached routers.
    For project-specific routers, use POST /router/models/{name}/delete.
    """
    _validate_model_name(model_name)
    return await UniversalRuntimeService.router_delete_model(model_name)


@router.post("/router/models/{model_name}/delete")
async def delete_router_model_project(
    model_name: str, request: RouterDeleteRequest
) -> dict[str, Any]:
    """Delete a saved router model from a project.

    Example request:
    ```json
    {
        "namespace": "default",
        "project_id": "my_project"
    }
    ```
    """
    _validate_model_name(model_name)
    deleted = RouterStorageService.delete_router(
        request.namespace, request.project_id, model_name
    )

    return {
        "deleted": deleted,
        "model": model_name,
        "namespace": request.namespace,
        "project_id": request.project_id,
    }


@router.post("/router/generate-data")
async def generate_router_data(request: RouterGenerateDataRequest) -> dict[str, Any]:
    """Generate synthetic training data for router routes.

    Uses an LLM to generate diverse example utterances based on
    route descriptions. Uses local model by default (no API key needed).

    Complexity options:
    - simple: Short, direct questions (5-10 words)
    - complex: Detailed, multi-part questions (15-30 words)
    - mixed: A mix of simple and complex (default)

    Single route generation:
    ```json
    {
        "route_description": "billing and payment inquiries",
        "count": 20,
        "complexity": "simple"
    }
    ```

    Batch generation for multiple routes:
    ```json
    {
        "routes": [
            {"route_name": "billing", "description": "billing inquiries", "count": 10},
            {
                "route_name": "support",
                "description": "tech support",
                "count": 10,
                "complexity": "complex"
            }
        ],
        "complexity": "mixed"
    }
    ```
    """
    # Convert routes to dict format if present
    routes_data = None
    if request.routes:
        routes_data = [
            {
                "route_name": route.route_name,
                "description": route.description,
                "count": route.count,
                "complexity": route.complexity,
            }
            for route in request.routes
        ]

    return await UniversalRuntimeService.router_generate_data(
        route_description=request.route_description,
        count=request.count,
        complexity=request.complexity,
        style=request.style,
        model=request.model,
        api_key=request.api_key,
        base_url=request.base_url,
        routes=routes_data,
    )
