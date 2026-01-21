"""Vision API router with 17 endpoints for image processing.

Endpoints:
- Zero-shot classification (CLIP): 2 endpoints
- Few-shot classification: 8 endpoints
- Object detection (YOLOS): 2 endpoints
- Open-vocabulary detection (OWL-ViT): 3 endpoints
- Background removal (RMBG): 2 endpoints
"""

import logging

from fastapi import APIRouter, HTTPException

from .service import FEW_SHOT_MODELS_DIR, vision_service
from .types import (
    BackgroundRemovalBatchRequest,
    BackgroundRemovalRequest,
    FewShotLoadRequest,
    FewShotPredictBatchRequest,
    FewShotPredictRequest,
    FewShotRefineRequest,
    FewShotTrainRequest,
    ObjectDetectionBatchRequest,
    ObjectDetectionRequest,
    OpenVocabDetectImageRequest,
    OpenVocabDetectTextBatchRequest,
    OpenVocabDetectTextRequest,
    ZeroShotClassifyBatchRequest,
    ZeroShotClassifyRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# =============================================================================
# Zero-Shot Classification (CLIP)
# =============================================================================


@router.post("/v1/vision/classify-zero-shot")
async def classify_zero_shot(request: ZeroShotClassifyRequest):
    """
    Zero-shot image classification using CLIP.

    Classify images into arbitrary categories without training. Simply provide
    a list of text labels and the model will predict probabilities for each.

    Example request:
    ```json
    {
        "image": "<base64 encoded image>",
        "labels": ["cat", "dog", "bird"],
        "model": "openai/clip-vit-base-patch32"
    }
    ```

    Response:
    ```json
    {
        "object": "classification",
        "label": "cat",
        "score": 0.87,
        "all_scores": {"cat": 0.87, "dog": 0.10, "bird": 0.03},
        "model": "openai/clip-vit-base-patch32"
    }
    ```
    """
    try:
        if not request.image or not request.image.strip():
            raise HTTPException(
                status_code=400, detail="Image data is required"
            )

        if not request.labels:
            raise HTTPException(
                status_code=400, detail="At least one label is required"
            )

        result = await vision_service.classify_zero_shot(
            request.image, request.labels, request.model
        )

        return {
            "object": "classification",
            "label": result["label"],
            "score": result["score"],
            "all_scores": result["all_scores"],
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_zero_shot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/classify-zero-shot/batch")
async def classify_zero_shot_batch(request: ZeroShotClassifyBatchRequest):
    """
    Batch zero-shot image classification using CLIP.

    Classify multiple images into arbitrary categories without training.
    More efficient than calling single endpoint multiple times.
    """
    try:
        if not request.labels:
            raise HTTPException(
                status_code=400, detail="At least one label is required"
            )
        if not request.images:
            raise HTTPException(
                status_code=400, detail="At least one image is required"
            )

        results = await vision_service.classify_zero_shot_batch(
            request.images, request.labels, request.model
        )

        return {
            "object": "list",
            "data": results,
            "model": request.model,
            "total_count": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in classify_zero_shot_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Few-Shot Classification
# =============================================================================


@router.post("/v1/vision/classify/fit")
async def train_few_shot_classifier(request: FewShotTrainRequest):
    """
    Train a few-shot image classifier using CLIP embeddings with linear probe.

    Creates a custom classifier that can distinguish between your specific
    categories using just 5-50 images per class.

    Example request:
    ```json
    {
        "classifier_id": "cat-dog-classifier",
        "images": ["<base64 cat1>", "<base64 cat2>", "<base64 dog1>", "<base64 dog2>"],
        "labels": ["cat", "cat", "dog", "dog"],
        "epochs": 100
    }
    ```
    """
    try:
        if len(request.images) != len(request.labels):
            raise HTTPException(
                status_code=400,
                detail=f"Number of images ({len(request.images)}) must match labels ({len(request.labels)})",
            )

        if len(request.images) < 2:
            raise HTTPException(
                status_code=400,
                detail="Need at least 2 images to train a classifier",
            )

        unique_classes = set(request.labels)
        if len(unique_classes) < 2:
            raise HTTPException(
                status_code=400,
                detail=f"Training requires at least 2 distinct classes. Found {len(unique_classes)}",
            )

        result = await vision_service.train_few_shot(
            classifier_id=request.classifier_id,
            images=request.images,
            labels=request.labels,
            model_name=request.model,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
        )

        return {
            "object": "few_shot_classifier",
            "classifier_id": request.classifier_id,
            **result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in train_few_shot_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/classify/refine")
async def refine_few_shot_classifier(request: FewShotRefineRequest):
    """
    Refine a few-shot classifier with additional training data.

    Add more examples to an existing classifier to improve accuracy
    or add new classes.
    """
    try:
        if len(request.images) != len(request.labels):
            raise HTTPException(
                status_code=400,
                detail=f"Number of images ({len(request.images)}) must match labels ({len(request.labels)})",
            )

        result = await vision_service.refine_few_shot(
            classifier_id=request.classifier_id,
            images=request.images,
            labels=request.labels,
            model_name=request.model,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
        )

        return {
            "object": "few_shot_classifier",
            "classifier_id": request.classifier_id,
            **result,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in refine_few_shot_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/vision/classify/models")
async def list_few_shot_classifiers():
    """
    List all saved few-shot classifiers available for loading.

    Returns classifiers saved in the models directory.
    """
    try:
        classifiers = vision_service.list_few_shot_classifiers()
        return {
            "object": "list",
            "classifiers": classifiers,
            "models_dir": str(FEW_SHOT_MODELS_DIR),
            "total": len(classifiers),
        }

    except Exception as e:
        logger.error(f"Error in list_few_shot_classifiers: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/classify/load")
async def load_few_shot_classifier_endpoint(request: FewShotLoadRequest):
    """
    Load a previously saved few-shot classifier.

    After loading, use /v1/vision/classify/predict to classify images.
    """
    try:
        classifier = await vision_service.load_saved_few_shot_classifier(
            request.classifier_id, request.model
        )
        model_path = vision_service._get_few_shot_path(request.classifier_id)

        return {
            "object": "load_result",
            "classifier_id": request.classifier_id,
            "path": str(model_path),
            "classes": classifier.classes,
            "loaded": True,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_few_shot_classifier_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/classify/predict")
async def predict_few_shot(request: FewShotPredictRequest):
    """
    Classify an image using a trained few-shot classifier.

    Example response:
    ```json
    {
        "object": "classification",
        "classifier_id": "cat-dog-classifier",
        "label": "cat",
        "score": 0.92,
        "all_scores": {"cat": 0.92, "dog": 0.08}
    }
    ```
    """
    try:
        result = await vision_service.predict_few_shot(
            request.classifier_id, request.image, request.model
        )

        return {
            "object": "classification",
            "classifier_id": request.classifier_id,
            **result,
        }

    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(status_code=404, detail=detail) from e
        raise HTTPException(status_code=400, detail=detail) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_few_shot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/classify/predict/batch")
async def predict_few_shot_batch(request: FewShotPredictBatchRequest):
    """
    Classify multiple images using a trained few-shot classifier.

    More efficient than calling the single-image endpoint multiple times.
    """
    try:
        if not request.images:
            raise HTTPException(
                status_code=400, detail="At least one image is required"
            )

        results = await vision_service.predict_few_shot_batch(
            request.classifier_id, request.images, request.model
        )

        return {
            "object": "list",
            "classifier_id": request.classifier_id,
            "data": results,
            "total_count": len(results),
        }

    except ValueError as e:
        detail = str(e)
        if "not found" in detail.lower():
            raise HTTPException(status_code=404, detail=detail) from e
        raise HTTPException(status_code=400, detail=detail) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in predict_few_shot_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/vision/classify/info/{classifier_id}")
async def get_few_shot_classifier_info(
    classifier_id: str,
    model: str = "openai/clip-vit-base-patch32",
):
    """
    Get information about a few-shot classifier.

    Returns details about whether it's loaded, trained, and its classes.
    """
    try:
        # First check if loaded in memory
        info = vision_service.get_few_shot_classifier_info(classifier_id, model)

        if info is not None:
            return {
                "object": "few_shot_classifier_info",
                "classifier_id": classifier_id,
                **info,
            }

        # Not loaded - return basic info
        return {
            "object": "few_shot_classifier_info",
            "classifier_id": classifier_id,
            "is_loaded": False,
            "is_trained": False,
            "classes": [],
            "num_classes": 0,
            "model": model,
            "message": "Classifier not found",
        }

    except Exception as e:
        logger.error(f"Error in get_few_shot_classifier_info: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/classify/{classifier_id}/unload")
async def unload_few_shot_classifier(
    classifier_id: str,
    model: str = "openai/clip-vit-base-patch32",
):
    """
    Unload a few-shot classifier from memory.

    This does NOT delete the saved model file - only frees memory.
    """
    try:
        unloaded = await vision_service.unload_few_shot_classifier(classifier_id, model)

        return {
            "object": "unload",
            "classifier_id": classifier_id,
            "unloaded": unloaded,
            **({"message": "Classifier not loaded in memory"} if not unloaded else {}),
        }

    except Exception as e:
        logger.error(f"Error in unload_few_shot_classifier: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Object Detection (YOLOS)
# =============================================================================


@router.post("/v1/vision/detect-objects")
async def detect_objects(request: ObjectDetectionRequest):
    """
    Detect objects in an image using YOLOS.

    YOLOS detects 80 COCO classes including person, car, dog, cat, etc.

    Example response:
    ```json
    {
        "object": "object_detection",
        "objects": [
            {"label": "person", "score": 0.95, "box": {"x1": 10, "y1": 20, "x2": 100, "y2": 200}}
        ],
        "count": 1
    }
    ```
    """
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="Image data is required")

        result = await vision_service.detect_objects(
            request.image, request.threshold, request.labels, request.model
        )

        return {
            "object": "object_detection",
            "objects": result["objects"],
            "count": result["count"],
            "image_size": result["image_size"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/detect-objects/batch")
async def detect_objects_batch(request: ObjectDetectionBatchRequest):
    """Detect objects in multiple images."""
    try:
        if not request.images:
            raise HTTPException(
                status_code=400, detail="At least one image is required"
            )

        results = await vision_service.detect_objects_batch(
            request.images, request.threshold, request.labels, request.model
        )

        return {
            "object": "object_detection_batch",
            "results": [
                {
                    "objects": r["objects"],
                    "count": r["count"],
                    "image_size": r["image_size"],
                }
                for r in results
            ],
            "total_images": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_objects_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Open-Vocabulary Detection (OWL-ViT)
# =============================================================================


@router.post("/v1/vision/detect-open")
async def detect_open_vocabulary(request: OpenVocabDetectTextRequest):
    """
    Detect objects using natural language text queries.

    OWL-ViT enables open-vocabulary object detection - find any object
    described in natural language, without retraining.

    Tips for better results:
    - Use descriptive queries: "a photo of a cat" works better than "cat"
    - Lower threshold (0.05-0.2) for recall, higher (0.3-0.5) for precision
    """
    try:
        if not request.image or not request.image.strip():
            raise HTTPException(
                status_code=400, detail="Image data is required"
            )

        if not request.queries:
            raise HTTPException(
                status_code=400, detail="At least one text query is required"
            )

        result = await vision_service.detect_by_text(
            request.image,
            request.queries,
            request.threshold,
            request.top_k,
            request.model,
        )

        return {
            "object": "open_vocab_detection",
            **result,
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_open_vocabulary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/detect-open/batch")
async def detect_open_vocabulary_batch(request: OpenVocabDetectTextBatchRequest):
    """Detect objects in multiple images using text queries."""
    try:
        if not request.images:
            raise HTTPException(
                status_code=400, detail="At least one image is required"
            )
        if not request.queries:
            raise HTTPException(
                status_code=400, detail="At least one text query is required"
            )

        results = await vision_service.detect_by_text_batch(
            request.images,
            request.queries,
            request.threshold,
            request.top_k,
            request.model,
        )

        return {
            "object": "open_vocab_detection_batch",
            "results": results,
            "total_images": len(results),
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_open_vocabulary_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/detect-open/by-image")
async def detect_by_reference_image(request: OpenVocabDetectImageRequest):
    """
    Detect objects similar to reference images (few-shot detection).

    Use this when you have example images of what you want to find.
    """
    try:
        if not request.image or not request.image.strip():
            raise HTTPException(
                status_code=400, detail="Target image is required"
            )

        if not request.query_images:
            raise HTTPException(
                status_code=400, detail="At least one query image is required"
            )

        result = await vision_service.detect_by_image(
            request.image,
            request.query_images,
            request.threshold,
            request.top_k,
            request.model,
        )

        return {
            "object": "image_guided_detection",
            **result,
            "model": request.model,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_by_reference_image: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Background Removal (RMBG)
# =============================================================================


@router.post("/v1/vision/remove-background")
async def remove_background(request: BackgroundRemovalRequest):
    """
    Remove background from an image using RMBG.

    Returns a PNG image with transparent background (alpha channel).

    With `return_mask: true`, also returns the grayscale alpha mask.
    """
    try:
        if not request.image:
            raise HTTPException(status_code=400, detail="Image data is required")

        result = await vision_service.remove_background(
            request.image, request.return_mask, request.model
        )

        response = {
            "object": "background_removal",
            "image": result["image"],
            "width": result["width"],
            "height": result["height"],
        }

        if request.return_mask and "mask" in result:
            response["mask"] = result["mask"]

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in remove_background: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/vision/remove-background/batch")
async def remove_background_batch(request: BackgroundRemovalBatchRequest):
    """Remove background from multiple images."""
    try:
        if not request.images:
            raise HTTPException(
                status_code=400, detail="At least one image is required"
            )

        results = await vision_service.remove_background_batch(
            request.images, request.return_mask, request.model
        )

        return {
            "object": "background_removal_batch",
            "results": [
                {
                    "image": r["image"],
                    "width": r["width"],
                    "height": r["height"],
                    **(
                        {"mask": r["mask"]}
                        if request.return_mask and "mask" in r
                        else {}
                    ),
                }
                for r in results
            ],
            "total_images": len(results),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in remove_background_batch: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
