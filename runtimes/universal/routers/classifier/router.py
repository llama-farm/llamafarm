"""Classifier router for SetFit-based text classification endpoints.

This router provides endpoints for:
- Fitting text classifiers with few-shot learning (SetFit)
- Predicting/classifying texts
- Loading pre-trained classifiers from disk
- Listing available classifier models
- Deleting saved classifiers

Note: The /v1/classifier/save endpoint has been removed. Models are auto-saved
after fitting to prevent data loss.
"""

import logging
import shutil
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from api_types.classifier import (
    ClassifierFitRequest,
    ClassifierLoadRequest,
    ClassifierPredictRequest,
)
from services.error_handler import handle_endpoint_errors
from services.path_validator import (
    PathValidationError,
    sanitize_model_name,
    validate_path_within_directory,
)

logger = logging.getLogger(__name__)


# Router with classifier prefix
router = APIRouter(prefix="/v1/classifier", tags=["classifier"])

# =============================================================================
# Dependency Injection
# =============================================================================

# Injected loader function
_load_classifier_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None

# Injected models directory
_CLASSIFIER_MODELS_DIR: Path | None = None

# Injected shared state
_classifiers: dict | None = None
_model_load_lock = None


def set_classifier_loader(
    load_classifier_fn: Callable[..., Coroutine[Any, Any, Any]] | None,
):
    """Set the classifier loading function.

    This allows the router to be decoupled from the server's model loading logic.
    The function should have signature: async def load_classifier(model_id, base_model) -> ClassifierModel
    """
    global _load_classifier_fn
    _load_classifier_fn = load_classifier_fn


def set_models_dir(models_dir: Path):
    """Set the models directory for classifier storage."""
    global _CLASSIFIER_MODELS_DIR
    _CLASSIFIER_MODELS_DIR = models_dir


def set_state(classifiers: dict, model_load_lock):
    """Set shared state for classifier caching.

    Args:
        classifiers: Dict/ModelCache for caching loaded classifiers
        model_load_lock: asyncio.Lock for synchronizing model loads
    """
    global _classifiers, _model_load_lock
    _classifiers = classifiers
    _model_load_lock = model_load_lock


def _get_classifier_loader():
    """Get the classifier loader, raising error if not initialized."""
    if _load_classifier_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Classifier loader not initialized. Server configuration error.",
        )
    return _load_classifier_fn


def _get_models_dir() -> Path:
    """Get the models directory, raising error if not set."""
    if _CLASSIFIER_MODELS_DIR is None:
        raise HTTPException(
            status_code=500,
            detail="Classifier models directory not configured.",
        )
    return _CLASSIFIER_MODELS_DIR


# =============================================================================
# Helper Functions
# =============================================================================


def _make_classifier_cache_key(model_name: str) -> str:
    """Create a cache key for classifier models."""
    return f"classifier:{model_name}"


def _get_classifier_path(model_name: str) -> Path:
    """Get the path for a classifier model directory.

    The path is always within CLASSIFIER_MODELS_DIR - users cannot control it.
    """
    safe_name = sanitize_model_name(model_name)
    return _get_models_dir() / safe_name


async def _auto_save_classifier_model(
    model: Any,
    model_name: str,
) -> dict[str, str | None]:
    """Auto-save classifier model after fit to prevent data loss.

    Models are saved immediately after training to ensure they persist
    across server restarts without requiring an explicit /save call.

    Returns:
        Dict with saved file path
    """
    try:
        models_dir = _get_models_dir()
        models_dir.mkdir(parents=True, exist_ok=True)

        # Generate path from model name
        save_path = _get_classifier_path(model_name)
        await model.save(str(save_path))

        logger.info(f"Auto-saved classifier model to {save_path}")
        return {"model_path": str(save_path)}

    except Exception as e:
        logger.warning(f"Auto-save failed (model still in memory): {e}")
        return {"model_path": None}


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/fit")
@handle_endpoint_errors("fit_classifier")
async def fit_classifier(request: ClassifierFitRequest):
    """
    Fit a text classifier using few-shot learning (SetFit).

    Train a classifier with as few as 8-16 examples per class.
    SetFit uses contrastive learning to fine-tune a sentence-transformer,
    then trains a small classification head.

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
        "num_iterations": 20
    }
    ```

    After fitting, use /v1/classifier/predict to classify new texts.
    Models are automatically saved to disk after fitting.
    """
    # Extract texts and labels from training data
    texts = [item["text"] for item in request.training_data]
    labels = [item["label"] for item in request.training_data]

    if len(texts) < 2:
        raise HTTPException(
            status_code=400,
            detail="At least 2 training examples required",
        )

    load_classifier = _get_classifier_loader()
    model = await load_classifier(
        model_id=request.model,
        base_model=request.base_model,
    )

    # Add model to cache (the loader may or may not do this)
    if _classifiers is not None:
        cache_key = _make_classifier_cache_key(request.model)
        _classifiers[cache_key] = model

    # Fit the classifier
    result = await model.fit(
        texts=texts,
        labels=labels,
        num_iterations=request.num_iterations,
        batch_size=request.batch_size,
    )

    # Auto-save model to prevent data loss on restart
    saved_paths = await _auto_save_classifier_model(
        model=model,
        model_name=request.model,
    )

    return {
        "object": "fit_result",
        "model": request.model,
        "base_model": result.base_model,
        "samples_fitted": result.samples_fitted,
        "num_classes": result.num_classes,
        "labels": result.labels,
        "training_time_ms": result.training_time_ms,
        "status": "fitted",
        "auto_saved": saved_paths["model_path"] is not None,
        "saved_path": saved_paths["model_path"],
    }


@router.post("/predict")
@handle_endpoint_errors("predict_classifier")
async def predict_classifier(request: ClassifierPredictRequest):
    """
    Classify texts using a fitted classifier.

    Example request:
    ```json
    {
        "model": "intent-classifier",
        "texts": ["I want to cancel my trip", "Book me a hotel"]
    }
    ```

    Returns predictions with confidence scores for each text.
    """
    if _classifiers is None:
        raise HTTPException(
            status_code=500,
            detail="Classifier state not initialized. Server configuration error.",
        )

    cache_key = _make_classifier_cache_key(request.model)

    # get() refreshes TTL automatically
    model = _classifiers.get(cache_key)
    if model is None:
        raise HTTPException(
            status_code=404,
            detail=f"Classifier '{request.model}' not found. "
            "Fit with /v1/classifier/fit or load with /v1/classifier/load first.",
        )

    if not model.is_fitted:
        raise HTTPException(
            status_code=400,
            detail="Model not fitted. Call /v1/classifier/fit first.",
        )

    results = await model.classify(request.texts)

    return {
        "object": "list",
        "data": [
            {
                "text": r.text,
                "label": r.label,
                "score": r.score,
                "all_scores": r.all_scores,
            }
            for r in results
        ],
        "total_count": len(results),
        "model": request.model,
    }


@router.post("/load")
@handle_endpoint_errors("load_classifier")
async def load_classifier_endpoint(request: ClassifierLoadRequest):
    """
    Load a pre-trained classifier from disk.

    Load a previously saved model for production inference without
    re-training. The model path is automatically determined from the
    model name - no user control over file paths.

    Example request:
    ```json
    {
        "model": "intent-classifier"
    }
    ```

    The model will be loaded from the classifier models directory and cached
    for subsequent /v1/classifier/predict calls.
    """
    if _classifiers is None or _model_load_lock is None:
        raise HTTPException(
            status_code=500,
            detail="Classifier state not initialized. Server configuration error.",
        )

    # Generate path from model name (no user-controlled paths)
    model_path = _get_classifier_path(request.model)
    models_dir = _get_models_dir()

    if not model_path.exists():
        available = (
            [f.name for f in models_dir.glob("*") if f.is_dir()]
            if models_dir.exists()
            else []
        )
        raise HTTPException(
            status_code=404,
            detail=f"Classifier '{request.model}' not found. "
            f"Available classifiers: {available}",
        )

    # Import here to avoid circular imports (only after existence check passes)
    from models import ClassifierModel
    from utils.device import get_optimal_device

    cache_key = _make_classifier_cache_key(request.model)

    # Remove existing model from cache if present
    if cache_key in _classifiers:
        existing = _classifiers.pop(cache_key)
        if existing:
            await existing.unload()

    async with _model_load_lock:
        logger.info(f"Loading pre-trained classifier: {model_path}")
        device = get_optimal_device()

        model = ClassifierModel(
            model_id=str(model_path),  # Pass path as model_id for loading
            device=device,
        )

        await model.load()
        _classifiers[cache_key] = model

    return {
        "object": "load_result",
        "model": request.model,
        "path": str(model_path),
        "is_fitted": model.is_fitted,
        "labels": model.labels,
        "num_classes": len(model.labels),
        "status": "loaded",
    }


@router.get("/models")
@handle_endpoint_errors("list_classifier_models")
async def list_classifier_models():
    """
    List all saved classifier models available for loading.

    Returns models saved in the classifier models directory.

    Response includes:
    - name: Name of the saved model
    - path: Full path to the model directory
    - labels: Class labels (if labels.txt exists)
    """
    models_dir = _get_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for path in models_dir.glob("*"):
        if path.is_dir():
            # Try to read labels
            labels = []
            labels_file = path / "labels.txt"
            if labels_file.exists():
                labels = labels_file.read_text().strip().split("\n")

            stat = path.stat()
            models.append(
                {
                    "name": path.name,
                    "path": str(path),
                    "labels": labels,
                    "num_classes": len(labels),
                    "modified": stat.st_mtime,
                }
            )

    # Sort by modification time (newest first)
    models.sort(key=lambda x: x["modified"], reverse=True)

    return {
        "object": "list",
        "data": models,
        "models_dir": str(models_dir),
        "total": len(models),
    }


@router.delete("/models/{model_name}")
@handle_endpoint_errors("delete_classifier_model")
async def delete_classifier_model(model_name: str):
    """
    Delete a saved classifier model.

    Removes the model directory from disk. Does not affect cached models.
    """
    models_dir = _get_models_dir()

    # Reject any path separators to prevent traversal attempts
    if "/" in model_name or "\\" in model_name or ".." in model_name:
        raise HTTPException(
            status_code=400,
            detail="Invalid model name: path separators not allowed",
        )

    # _get_classifier_path already sanitizes via sanitize_model_name
    model_path = _get_classifier_path(model_name)

    # Validate the resolved path is still within the safe directory
    try:
        resolved_path = validate_path_within_directory(model_path, models_dir)
    except PathValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not resolved_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Classifier model not found: {model_name}",
        )

    # Remove directory and contents
    shutil.rmtree(resolved_path)

    return {
        "object": "delete_result",
        "model": model_name,
        "deleted": True,
    }
