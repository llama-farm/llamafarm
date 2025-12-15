"""
ML Router - Proxy endpoints to Universal Runtime's specialized ML capabilities.

Provides access to:
- OCR (text extraction from images/PDFs)
- Document Extraction (structured data from forms/invoices)
- Custom Text Classification (SetFit few-shot learning)
- Anomaly Detection (train and detect anomalies)
"""

import logging
from typing import Any

from fastapi import APIRouter, Form, UploadFile
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
    DocumentExtractRequest,
    OCRRequest,
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
# File Management
# =============================================================================


@router.post("/files")
async def upload_file(
    file: UploadFile,
    convert_pdf: bool = Form(default=True),
    pdf_dpi: int = Form(default=150),
) -> dict[str, Any]:
    """Upload a file for use with OCR, document extraction, or image generation.

    Uploaded files are stored temporarily (5 minutes TTL) and can be referenced
    by their file ID in subsequent API calls.

    For PDFs, pages are automatically converted to images for OCR/document processing.

    Args:
        file: The file to upload (images, PDFs supported, max 100MB)
        convert_pdf: If True, convert PDF pages to images (default: True)
        pdf_dpi: DPI for PDF to image conversion (default: 150)

    Returns:
        File metadata including ID for referencing in other endpoints
    """
    return await UniversalRuntimeService.upload_file(
        file=file,
        convert_pdf=convert_pdf,
        pdf_dpi=pdf_dpi,
    )


@router.get("/files")
async def list_files() -> dict[str, Any]:
    """List all uploaded files with their metadata."""
    return await UniversalRuntimeService.list_files()


@router.get("/files/{file_id}")
async def get_file(file_id: str) -> dict[str, Any]:
    """Get metadata for a specific uploaded file."""
    return await UniversalRuntimeService.get_file(file_id)


@router.get("/files/{file_id}/images")
async def get_file_images(file_id: str) -> dict[str, Any]:
    """Get base64-encoded images for a file.

    For PDFs, returns one image per page.
    For image files, returns the image itself.
    """
    return await UniversalRuntimeService.get_file_images(file_id)


@router.delete("/files/{file_id}")
async def delete_file(file_id: str) -> dict[str, Any]:
    """Delete an uploaded file."""
    return await UniversalRuntimeService.delete_file(file_id)


# =============================================================================
# OCR Endpoints
# =============================================================================


@router.post("/ocr")
async def extract_text(request: OCRRequest) -> dict[str, Any]:
    """OCR endpoint for text extraction from images.

    Supports multiple OCR backends:
    - surya: Best accuracy, transformer-based, layout-aware (recommended)
    - easyocr: Good multilingual support (80+ languages), widely used
    - paddleocr: Fast, optimized for production, excellent for Asian languages
    - tesseract: Classic OCR engine, CPU-only, widely deployed

    You can provide images either as:
    1. Base64-encoded strings in the `images` field
    2. A file ID from a previous upload via `file_id` field

    Example with file_id (from /v1/ml/files upload):
    ```json
    {
        "model": "surya",
        "file_id": "file_abc123_def456",
        "languages": ["en"]
    }
    ```
    """
    return await UniversalRuntimeService.ocr(
        model=request.model,
        images=request.images,
        file_id=request.file_id,
        languages=request.languages,
        return_boxes=request.return_boxes,
    )


# =============================================================================
# Document Extraction Endpoints
# =============================================================================


@router.post("/documents/extract")
async def extract_from_documents(request: DocumentExtractRequest) -> dict[str, Any]:
    """Document understanding endpoint.

    Extract structured information from documents using vision-language models.
    Supports forms, invoices, receipts, and other document types.

    Model types:
    - Donut models: End-to-end, no OCR needed (naver-clova-ix/donut-*)
    - LayoutLM models: Uses OCR + layout features (microsoft/layoutlmv3-*)

    Tasks:
    - extraction: Extract key-value pairs from documents
    - vqa: Answer questions about document content
    - classification: Classify document types

    Example with file_id:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-cord-v2",
        "file_id": "file_abc123_def456",
        "task": "extraction"
    }
    ```

    For VQA, include prompts:
    ```json
    {
        "model": "naver-clova-ix/donut-base-finetuned-docvqa",
        "file_id": "file_abc123_def456",
        "prompts": ["What is the total amount?"],
        "task": "vqa"
    }
    ```
    """
    return await UniversalRuntimeService.extract_documents(
        model=request.model,
        images=request.images,
        file_id=request.file_id,
        prompts=request.prompts,
        task=request.task,
    )


# =============================================================================
# SetFit Classifier Endpoints
# =============================================================================


@router.post("/classifier/fit")
async def fit_classifier(request: ClassifierFitRequest) -> dict[str, Any]:
    """Fit a text classifier using few-shot learning (SetFit).

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

    After fitting, use /v1/ml/classifier/predict to classify new texts.
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
    """Classify texts using a fitted classifier.

    Example request:
    ```json
    {
        "model": "intent-classifier",
        "texts": ["I want to cancel my trip", "Book me a hotel"]
    }
    ```

    Returns predictions with confidence scores for each text.
    """
    return await UniversalRuntimeService.classifier_predict(
        model=request.model,
        texts=request.texts,
    )


@router.post("/classifier/save")
async def save_classifier(request: ClassifierSaveRequest) -> dict[str, Any]:
    """Save a fitted classifier to disk for production use.

    After fitting a model with /v1/ml/classifier/fit, save it to disk so it
    persists across server restarts.

    Models are saved to ~/.llamafarm/models/classifier/ with auto-generated
    directory names based on the model name.
    """
    return await UniversalRuntimeService.classifier_save(model=request.model)


@router.post("/classifier/load")
async def load_classifier(request: ClassifierLoadRequest) -> dict[str, Any]:
    """Load a pre-trained classifier from disk.

    Load a previously saved model for production inference without
    re-training.

    Example request:
    ```json
    {
        "model": "intent-classifier"
    }
    ```
    """
    return await UniversalRuntimeService.classifier_load(model=request.model)


@router.get("/classifier/models")
async def list_classifier_models() -> dict[str, Any]:
    """List all saved classifier models available for loading.

    Returns models saved in the classifier models directory.

    Response includes:
    - name: Name of the saved model
    - path: Full path to the model directory
    - labels: Class labels (if labels.txt exists)
    """
    return await UniversalRuntimeService.classifier_list_models()


@router.delete("/classifier/models/{model_name}")
async def delete_classifier_model(model_name: str) -> dict[str, Any]:
    """Delete a saved classifier model.

    Removes the model directory from disk. Does not affect cached models.
    """
    return await UniversalRuntimeService.classifier_delete_model(model_name)


# =============================================================================
# Anomaly Detection Endpoints
# =============================================================================


@router.post("/anomaly/fit")
async def fit_anomaly_detector(request: AnomalyFitRequest) -> dict[str, Any]:
    """Fit an anomaly detector on training data.

    Train an anomaly detection model on data assumed to be mostly normal.
    The model learns what "normal" looks like and can then detect deviations.

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
        "contamination": 0.1
    }
    ```

    Example request (dict data with schema):
    ```json
    {
        "model": "api-monitor",
        "backend": "isolation_forest",
        "data": [
            {"response_time_ms": 100, "bytes": 1024, "method": "GET"},
            {"response_time_ms": 105, "bytes": 1100, "method": "POST"}
        ],
        "schema": {
            "response_time_ms": "numeric",
            "bytes": "numeric",
            "method": "label"
        },
        "contamination": 0.1
    }
    ```

    After fitting, use /v1/ml/anomaly/score or /v1/ml/anomaly/detect.
    """
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
    """Score data points for anomalies.

    Detects anomalies in data using various algorithms.
    Returns all points with their anomaly scores.

    Note: Model must be fitted first via /v1/ml/anomaly/fit or loaded from disk.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 200.0]],
        "threshold": 0.5
    }
    ```

    Response includes:
    - score: Anomaly score (0-1, higher = more anomalous)
    - is_anomaly: Boolean based on threshold
    - raw_score: Backend-specific raw score
    """
    return await UniversalRuntimeService.anomaly_score(
        model=request.model,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        threshold=request.threshold,
    )


@router.post("/anomaly/detect")
async def detect_anomalies(request: AnomalyScoreRequest) -> dict[str, Any]:
    """Detect anomalies in data (returns only anomalous points).

    Same as /v1/ml/anomaly/score but filters to return only points
    classified as anomalies.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [100.0, 200.0]],
        "threshold": 0.5
    }
    ```
    """
    return await UniversalRuntimeService.anomaly_detect(
        model=request.model,
        data=request.data,
        backend=request.backend,
        schema=request.schema,
        threshold=request.threshold,
    )


@router.post("/anomaly/save")
async def save_anomaly_model(request: AnomalySaveRequest) -> dict[str, Any]:
    """Save a fitted anomaly model to disk for production use.

    After fitting a model with /v1/ml/anomaly/fit, save it to disk so it
    persists across server restarts.

    Models are saved to ~/.llamafarm/models/anomaly/ with auto-generated
    filenames based on the model name and backend.
    """
    return await UniversalRuntimeService.anomaly_save(
        model=request.model,
        backend=request.backend,
    )


@router.post("/anomaly/load")
async def load_anomaly_model(request: AnomalyLoadRequest) -> dict[str, Any]:
    """Load a pre-trained anomaly model from disk.

    Load a previously saved model for production inference without
    re-training.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest"
    }
    ```
    """
    return await UniversalRuntimeService.anomaly_load(
        model=request.model,
        backend=request.backend,
    )


@router.get("/anomaly/models")
async def list_anomaly_models() -> dict[str, Any]:
    """List all saved anomaly models available for loading.

    Returns models saved in the anomaly models directory.

    Response includes:
    - filename: Name of the saved model file
    - size_bytes: File size
    - modified: Last modification timestamp
    - backend: Detected backend type (from file extension)
    """
    return await UniversalRuntimeService.anomaly_list_models()


@router.delete("/anomaly/models/{filename}")
async def delete_anomaly_model(filename: str) -> dict[str, Any]:
    """Delete a saved anomaly model.

    Removes the model file from disk. Does not affect cached models.
    """
    return await UniversalRuntimeService.anomaly_delete_model(filename)
