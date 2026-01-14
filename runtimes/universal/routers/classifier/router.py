"""
SetFit classifier endpoints.

Train and use few-shot text classifiers using SetFit.
"""

import shutil

from fastapi import APIRouter, HTTPException

from core.logging import UniversalRuntimeLogger
from models import ClassifierModel
from state import (
    CLASSIFIER_MODELS_DIR,
    get_classifiers_cache,
    get_device,
    get_model_load_lock,
    validate_path_within_directory,
)

from .service import (
    auto_save_classifier_model,
    get_classifier_path,
    load_classifier,
    make_classifier_cache_key,
)
from .types import (
    ClassifierFitRequest,
    ClassifierLoadRequest,
    ClassifierPredictRequest,
    ClassifierSaveRequest,
)

router = APIRouter()
logger = UniversalRuntimeLogger("universal-runtime.classifier")


@router.post("/v1/classifier/fit")
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
    """
    try:
        # Extract texts and labels from training data
        texts = [item["text"] for item in request.training_data]
        labels = [item["label"] for item in request.training_data]

        if len(texts) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 training examples required",
            )

        model = await load_classifier(
            model_id=request.model,
            base_model=request.base_model,
        )

        # Fit the classifier
        result = await model.fit(
            texts=texts,
            labels=labels,
            num_iterations=request.num_iterations,
            batch_size=request.batch_size,
        )

        # Auto-save model to prevent data loss on restart
        saved_paths = await auto_save_classifier_model(
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fit_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/classifier/predict")
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
    try:
        classifiers_cache = get_classifiers_cache()
        cache_key = make_classifier_cache_key(request.model)

        # get() refreshes TTL automatically
        model = classifiers_cache.get(cache_key)
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/classifier/save")
async def save_classifier(request: ClassifierSaveRequest):
    """
    Save a fitted classifier to disk for production use.

    After fitting a model with /v1/classifier/fit, save it to disk so it
    persists across server restarts.

    Example request:
    ```json
    {
        "model": "intent-classifier"
    }
    ```

    Models are saved to ~/.llamafarm/models/classifier/ with auto-generated
    directory names based on the model name.
    """
    try:
        classifiers_cache = get_classifiers_cache()
        cache_key = make_classifier_cache_key(request.model)

        if cache_key not in classifiers_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.model}' not found in cache. "
                "Fit the model first with /v1/classifier/fit",
            )

        model = classifiers_cache[cache_key]

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/classifier/fit first.",
            )

        # Create models directory if needed
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name (no user-controlled paths)
        save_path = get_classifier_path(request.model)
        await model.save(str(save_path))

        return {
            "object": "save_result",
            "model": request.model,
            "path": str(save_path),
            "labels": model.labels,
            "status": "saved",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/classifier/load")
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

    The model will be loaded from ~/.llamafarm/models/classifier/ and cached
    for subsequent /v1/classifier/predict calls.
    """
    try:
        classifiers_cache = get_classifiers_cache()
        model_load_lock = get_model_load_lock()

        # Generate path from model name (no user-controlled paths)
        model_path = get_classifier_path(request.model)

        if not model_path.exists():
            available = (
                [f.name for f in CLASSIFIER_MODELS_DIR.glob("*") if f.is_dir()]
                if CLASSIFIER_MODELS_DIR.exists()
                else []
            )
            raise HTTPException(
                status_code=404,
                detail=f"Classifier '{request.model}' not found. "
                f"Available classifiers: {available}",
            )

        cache_key = make_classifier_cache_key(request.model)

        # Remove existing model from cache if present
        if cache_key in classifiers_cache:
            existing = classifiers_cache.pop(cache_key)
            if existing:
                await existing.unload()

        async with model_load_lock:
            logger.info(f"Loading pre-trained classifier: {model_path}")
            device = get_device()

            model = ClassifierModel(
                model_id=str(model_path),  # Pass path as model_id for loading
                device=device,
            )

            await model.load()
            classifiers_cache[cache_key] = model

        return {
            "object": "load_result",
            "model": request.model,
            "path": str(model_path),
            "is_fitted": model.is_fitted,
            "labels": model.labels,
            "num_classes": len(model.labels),
            "status": "loaded",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/classifier/models")
async def list_classifier_models():
    """
    List all saved classifier models available for loading.

    Returns models saved in the CLASSIFIER_MODELS_DIR directory.

    Response includes:
    - name: Name of the saved model
    - path: Full path to the model directory
    - labels: Class labels (if labels.txt exists)
    """
    try:
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        models = []
        for path in CLASSIFIER_MODELS_DIR.glob("*"):
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
            "models_dir": str(CLASSIFIER_MODELS_DIR),
            "total": len(models),
        }

    except Exception as e:
        logger.error(f"Error in list_classifier_models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/v1/classifier/models/{model_name}")
async def delete_classifier_model(model_name: str):
    """
    Delete a saved classifier model.

    Removes the model directory from disk. Does not affect cached models.
    """
    try:
        # Reject any path separators to prevent traversal attempts
        if "/" in model_name or "\\" in model_name or ".." in model_name:
            raise HTTPException(
                status_code=400,
                detail="Invalid model name: path separators not allowed",
            )

        # get_classifier_path already sanitizes via sanitize_model_name
        model_path = get_classifier_path(model_name)

        # Validate the resolved path is still within the safe directory
        try:
            resolved_path = validate_path_within_directory(
                model_path, CLASSIFIER_MODELS_DIR
            )
        except ValueError as e:
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

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_classifier_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
