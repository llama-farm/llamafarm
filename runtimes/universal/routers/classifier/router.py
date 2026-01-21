"""Classifier API router with 6 endpoints.

Endpoints:
- POST /v1/classifier/fit - Train a text classifier
- POST /v1/classifier/predict - Classify texts
- POST /v1/classifier/save - Save classifier to disk
- POST /v1/classifier/load - Load classifier from disk
- GET /v1/classifier/models - List saved models
- DELETE /v1/classifier/models/{name} - Delete saved model
"""

import logging

from fastapi import APIRouter, HTTPException

from .service import CLASSIFIER_MODELS_DIR, classifier_service
from .types import (
    ClassifierFitRequest,
    ClassifierLoadRequest,
    ClassifierPredictRequest,
    ClassifierSaveRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
        texts = [item["text"] for item in request.training_data]
        labels = [item["label"] for item in request.training_data]

        if len(texts) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 training examples required",
            )

        result = await classifier_service.fit(
            model_id=request.model,
            texts=texts,
            labels=labels,
            base_model=request.base_model,
            num_iterations=request.num_iterations,
            batch_size=request.batch_size,
        )

        # Auto-save model
        model = await classifier_service.get_model(request.model)
        saved_paths = await classifier_service.auto_save(request.model, model)

        return {
            "object": "fit_result",
            "model": result["model"],
            "base_model": result["base_model"],
            "samples_fitted": result["samples_fitted"],
            "num_classes": result["num_classes"],
            "labels": result["labels"],
            "training_time_ms": result["training_time_ms"],
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
        result = await classifier_service.predict(request.model, request.texts)

        return {
            "object": "classification",
            "model": result["model"],
            "results": result["results"],
        }

    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"{e}. Fit with /v1/classifier/fit or load with /v1/classifier/load first.",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{e}. Call /v1/classifier/fit first.",
        ) from e
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
    """
    try:
        result = await classifier_service.save(request.model)

        return {
            "object": "save_result",
            "model": result["model"],
            "path": result["path"],
            "is_fitted": result["is_fitted"],
            "labels": result["labels"],
            "num_classes": result["num_classes"],
            "status": "saved",
        }

    except KeyError as e:
        raise HTTPException(
            status_code=404,
            detail=f"{e}. Fit the model first with /v1/classifier/fit",
        ) from e
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"{e}. Call /v1/classifier/fit first.",
        ) from e
    except Exception as e:
        logger.error(f"Error in save_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/classifier/load")
async def load_classifier_endpoint(request: ClassifierLoadRequest):
    """
    Load a pre-trained classifier from disk.

    Load a previously saved model for production inference without
    re-training.

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
        result = await classifier_service.load_from_disk(request.model)

        return {
            "object": "load_result",
            "model": result["model"],
            "path": result["path"],
            "is_fitted": result["is_fitted"],
            "labels": result["labels"],
            "num_classes": result["num_classes"],
            "status": "loaded",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error in load_classifier_endpoint: {e}", exc_info=True)
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
    - num_classes: Number of classes
    """
    try:
        models = classifier_service.list_models()

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
        result = classifier_service.delete_model(model_name)

        return {
            "object": "delete_result",
            "model": result["model"],
            "path": result["path"],
            "status": "deleted",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error in delete_classifier_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
