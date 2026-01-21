"""
ML Router - Endpoints for ML model training and inference.

Provides access to:
- Custom Text Classification (SetFit few-shot learning)
- Anomaly Detection (train, detect, explain anomalies)
- Embeddings (sentence-transformers)
- NLP: Language detection, keyword extraction, PII detection/redaction
- Time Series: Forecasting, change point detection
- Vision: Object detection, background removal
- Analysis: Table QA, dataset audit, drift detection

Note: OCR and Document extraction have moved to /v1/vision/*
"""

import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from server.services.ml_model_service import MLModelService
from server.services.universal_runtime_service import UniversalRuntimeService

from .types import (
    AnomalyExplainRequest,
    AnomalyFitRequest,
    AnomalyLoadRequest,
    AnomalySaveRequest,
    AnomalyScoreRequest,
    AudioTranscriptionRequest,
    AudioTranslationRequest,
    ChangePointBatchRequest,
    ChangePointRequest,
    ClassifierFitRequest,
    ClassifierLoadRequest,
    ClassifierPredictRequest,
    ClassifierSaveRequest,
    DatasetAuditRequest,
    DriftDetectRequest,
    EmbeddingsRequest,
    KeywordExtractBatchRequest,
    KeywordExtractRequest,
    LanguageDetectBatchRequest,
    LanguageDetectRequest,
    PIIDetectRequest,
    PIIRedactRequest,
    TableQABatchRequest,
    TableQARequest,
    TimeSeriesForecastBatchRequest,
    TimeSeriesForecastRequest,
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

    After fitting, use /v1/ml/classifier/save to persist the model (with optional description).
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

    After fitting, use /v1/ml/anomaly/save to persist the model (with optional description).
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
        scaler_type=request.scaler_type,
        epochs=request.epochs,
        batch_size=request.batch_size,
        validation_split=request.validation_split,
        patience=request.patience,
        min_delta=request.min_delta,
        training_file=request.training_file,
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
        scaler_type=request.scaler_type,
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
        scaler_type=request.scaler_type,
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


@router.post("/anomaly/explain")
async def explain_anomaly(request: AnomalyExplainRequest) -> dict[str, Any]:
    """Explain why data points are flagged as anomalies using SHAP.

    Returns feature contributions that explain the anomaly score.

    Example request:
    ```json
    {
        "model_id": "sensor-detector",
        "data": [[95.0, 95.0, 200.0]],
        "feature_names": ["cpu", "memory", "latency"],
        "backend": "isolation_forest"
    }
    ```
    """
    return await UniversalRuntimeService.explain_anomaly(
        model_id=request.model_id,
        data=request.data,
        feature_names=request.feature_names,
        backend=request.backend,
        background_samples=request.background_samples,
        nsamples=request.nsamples,
    )


# =============================================================================
# Embeddings Endpoints
# =============================================================================


@router.post("/embeddings")
async def create_embeddings(request: EmbeddingsRequest) -> dict[str, Any]:
    """Generate embeddings for text.

    Uses sentence-transformers models to generate vector embeddings.

    Example request:
    ```json
    {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "input": ["Hello world", "How are you?"],
        "encoding_format": "float"
    }
    ```
    """
    return await UniversalRuntimeService.embeddings(
        model=request.model,
        input=request.input,
        encoding_format=request.encoding_format,
    )


# =============================================================================
# NLP Endpoints
# =============================================================================


@router.post("/nlp/language")
async def detect_language(request: LanguageDetectRequest) -> dict[str, Any]:
    """Detect the language of text.

    Example request:
    ```json
    {
        "text": "Bonjour, comment allez-vous?"
    }
    ```
    """
    return await UniversalRuntimeService.detect_language(
        text=request.text,
        model=request.model,
    )


@router.post("/nlp/language/batch")
async def detect_language_batch(request: LanguageDetectBatchRequest) -> dict[str, Any]:
    """Detect the language of multiple texts."""
    return await UniversalRuntimeService.detect_language_batch(
        texts=request.texts,
        model=request.model,
    )


@router.post("/nlp/keywords")
async def extract_keywords(request: KeywordExtractRequest) -> dict[str, Any]:
    """Extract keywords and keyphrases from text.

    Example request:
    ```json
    {
        "text": "Machine learning enables computers to learn from data.",
        "top_k": 5,
        "diversity": 0.5
    }
    ```
    """
    return await UniversalRuntimeService.extract_keywords(
        text=request.text,
        top_k=request.top_k,
        diversity=request.diversity,
        ngram_range=request.ngram_range,
    )


@router.post("/nlp/keywords/batch")
async def extract_keywords_batch(
    request: KeywordExtractBatchRequest,
) -> dict[str, Any]:
    """Extract keywords from multiple texts."""
    return await UniversalRuntimeService.extract_keywords_batch(
        texts=request.texts,
        top_k=request.top_k,
        diversity=request.diversity,
        ngram_range=request.ngram_range,
    )


@router.post("/nlp/pii/detect")
async def detect_pii(request: PIIDetectRequest) -> dict[str, Any]:
    """Detect Personally Identifiable Information (PII) in text.

    Example request:
    ```json
    {
        "text": "Contact John Smith at john@email.com or call 555-1234."
    }
    ```
    """
    return await UniversalRuntimeService.detect_pii(
        text=request.text,
        entity_types=request.entity_types,
        threshold=request.threshold,
        use_regex=request.use_regex,
    )


@router.post("/nlp/pii/redact")
async def redact_pii(request: PIIRedactRequest) -> dict[str, Any]:
    """Redact PII from text.

    Example request:
    ```json
    {
        "text": "My SSN is 123-45-6789 and email is test@example.com",
        "replacement": "[REDACTED]"
    }
    ```
    """
    return await UniversalRuntimeService.redact_pii(
        text=request.text,
        entity_types=request.entity_types,
        replacement=request.replacement,
        replacement_map=request.replacement_map,
        threshold=request.threshold,
        use_regex=request.use_regex,
    )


# =============================================================================
# Time Series Endpoints
# =============================================================================


@router.post("/timeseries/forecast")
async def forecast_timeseries(request: TimeSeriesForecastRequest) -> dict[str, Any]:
    """Forecast future values in a time series.

    Uses Chronos models for probabilistic forecasting.

    Example request:
    ```json
    {
        "data": [100, 102, 105, 103, 107, 110, 108, 112],
        "horizon": 5
    }
    ```
    """
    return await UniversalRuntimeService.forecast_timeseries(
        data=request.data,
        horizon=request.horizon,
        model=request.model,
    )


@router.post("/timeseries/changepoints")
async def detect_changepoints(request: ChangePointRequest) -> dict[str, Any]:
    """Detect change points in time series data.

    Uses Ruptures library for change point detection.

    Example request:
    ```json
    {
        "data": [10, 11, 10, 12, 25, 26, 24, 27, 15, 16],
        "algorithm": "binseg",
        "n_changepoints": 2
    }
    ```
    """
    return await UniversalRuntimeService.detect_changepoints(
        data=request.data,
        algorithm=request.algorithm,
        n_changepoints=request.n_changepoints,
        penalty=request.penalty,
    )


# =============================================================================
# Analysis Endpoints
# =============================================================================
# Note: Vision endpoints have been moved to /v1/vision/*


@router.post("/analysis/table-qa")
async def table_qa(request: TableQARequest) -> dict[str, Any]:
    """Answer questions about tabular data.

    Uses TAPAS model for table question answering.

    Example request:
    ```json
    {
        "table": [
            {"name": "Alice", "age": 30},
            {"name": "Bob", "age": 25}
        ],
        "query": "What is the average age?"
    }
    ```
    """
    return await UniversalRuntimeService.table_qa(
        table=request.table,
        query=request.query,
        model=request.model,
    )


@router.post("/analysis/dataset-audit")
async def audit_dataset(request: DatasetAuditRequest) -> dict[str, Any]:
    """Audit dataset for label quality issues.

    Uses Cleanlab to identify potentially mislabeled examples.

    Example request:
    ```json
    {
        "labels": [0, 1, 0, 1, 0],
        "pred_probs": [[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7], [0.7, 0.3]],
        "label_names": ["cat", "dog"]
    }
    ```
    """
    return await UniversalRuntimeService.audit_dataset(
        labels=request.labels,
        pred_probs=request.pred_probs,
        label_names=request.label_names,
    )


@router.post("/analysis/drift")
async def detect_drift(request: DriftDetectRequest) -> dict[str, Any]:
    """Detect drift between reference and current data.

    Uses River library for drift detection.

    Example request:
    ```json
    {
        "reference_data": [5.0, 5.1, 4.9, 5.2],
        "current_data": [10.0, 10.1, 9.9, 10.2],
        "algorithm": "adwin"
    }
    ```
    """
    return await UniversalRuntimeService.detect_drift(
        reference_data=request.reference_data,
        current_data=request.current_data,
        algorithm=request.algorithm,
    )


# =============================================================================
# Audio Endpoints
# =============================================================================


@router.post("/audio/transcriptions")
async def transcribe_audio(request: AudioTranscriptionRequest) -> dict[str, Any]:
    """Transcribe audio to text (speech-to-text).

    Uses Whisper models for high-quality transcription.

    Example request:
    ```json
    {
        "file": "<base64-encoded-audio>",
        "model": "openai/whisper-base",
        "language": "en",
        "response_format": "json"
    }
    ```
    """
    return await UniversalRuntimeService.transcribe_audio(
        file=request.file,
        model=request.model,
        language=request.language,
        prompt=request.prompt,
        response_format=request.response_format,
        temperature=request.temperature,
    )


@router.post("/audio/translations")
async def translate_audio(request: AudioTranslationRequest) -> dict[str, Any]:
    """Translate audio to English text.

    Uses Whisper models to translate any language to English.

    Example request:
    ```json
    {
        "file": "<base64-encoded-audio>",
        "model": "openai/whisper-base",
        "response_format": "json"
    }
    ```
    """
    return await UniversalRuntimeService.translate_audio(
        file=request.file,
        model=request.model,
        prompt=request.prompt,
        response_format=request.response_format,
        temperature=request.temperature,
    )


# =============================================================================
# Batch Endpoints
# =============================================================================


@router.post("/timeseries/forecast/batch")
async def forecast_timeseries_batch(
    request: TimeSeriesForecastBatchRequest,
) -> dict[str, Any]:
    """Forecast multiple time series in batch.

    Example request:
    ```json
    {
        "data": [
            [100, 102, 105, 103, 107],
            [200, 202, 205, 203, 207]
        ],
        "horizon": 5,
        "model": "amazon/chronos-t5-tiny"
    }
    ```
    """
    return await UniversalRuntimeService.forecast_timeseries_batch(
        data=request.data,
        horizon=request.horizon,
        model=request.model,
    )


@router.post("/timeseries/changepoints/batch")
async def detect_changepoints_batch(
    request: ChangePointBatchRequest,
) -> dict[str, Any]:
    """Detect change points in multiple time series.

    Example request:
    ```json
    {
        "data": [
            [10, 11, 10, 12, 25, 26, 24, 27],
            [5, 5, 5, 20, 20, 20, 5, 5]
        ],
        "algorithm": "binseg",
        "n_changepoints": 2
    }
    ```
    """
    return await UniversalRuntimeService.detect_changepoints_batch(
        data=request.data,
        algorithm=request.algorithm,
        n_changepoints=request.n_changepoints,
        penalty=request.penalty,
    )


@router.post("/analysis/table-qa/batch")
async def table_qa_batch(request: TableQABatchRequest) -> dict[str, Any]:
    """Answer questions about multiple tables in batch.

    Example request:
    ```json
    {
        "tables": [
            [{"name": "Alice", "age": 30}],
            [{"name": "Bob", "age": 25}]
        ],
        "queries": ["What is the age?", "Who is this?"],
        "model": "google/tapas-base-finetuned-wtq"
    }
    ```
    """
    return await UniversalRuntimeService.table_qa_batch(
        tables=request.tables,
        queries=request.queries,
        model=request.model,
    )
