"""Anomaly detection API router with 9 endpoints.

Endpoints:
- Score anomalies: POST /v1/anomaly/score
- Fit detector: POST /v1/anomaly/fit
- Detect anomalies: POST /v1/anomaly/detect
- Save model: POST /v1/anomaly/save
- Load model: POST /v1/anomaly/load
- List models: GET /v1/anomaly/models
- Delete model: DELETE /v1/anomaly/models/{filename}
- Upload training data: POST /v1/anomaly/upload-training-data
- Explain: POST /v1/anomaly/explain
"""

import logging

from fastapi import APIRouter, Form, HTTPException, UploadFile

from .service import anomaly_service, get_streaming_loader
from .types import (
    AnomalyExplainRequest,
    AnomalyFitRequest,
    AnomalyLoadRequest,
    AnomalySaveRequest,
    AnomalyScoreRequest,
)

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/anomaly/score")
async def score_anomalies(request: AnomalyScoreRequest):
    """
    Score data points for anomalies.

    Detects anomalies in data using various algorithms:
    - isolation_forest: Fast tree-based method, good general purpose
    - one_class_svm: Support vector machine for outlier detection
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Neural network, best for complex patterns
    - copod: Copula-based outlier detection (fast, parameter-free)
    - hbos: Histogram-based outlier score (fastest)
    - ecod: Empirical cumulative distribution outlier detection

    Note: Model must be fitted first via /v1/anomaly/fit or loaded from disk.

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
    try:
        cache_key = anomaly_service._make_cache_key(
            request.model, request.backend, request.normalization, request.scaler_type
        )

        model = await anomaly_service.load_model(
            model_id=request.model,
            backend=request.backend,
            normalization=request.normalization,
            scaler_type=request.scaler_type,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first or load a pre-trained model.",
            )

        # Prepare data (encode if dict-based)
        prepared_data = anomaly_service.prepare_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=False,  # Use existing encoder
        )

        # Score data
        results = await anomaly_service.score(
            model=model,
            data=prepared_data,
            threshold=request.threshold,
        )

        # Summary statistics
        anomaly_count = sum(1 for r in results if r["is_anomaly"])

        return {
            "object": "list",
            "data": results,
            "total_count": len(results),
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "total_points": len(results),
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / len(results) if results else 0,
                "threshold": request.threshold or model.threshold,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in score_anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/anomaly/upload-training-data")
async def upload_training_data(
    file: UploadFile,
    skip_columns: str = Form(default=""),
):
    """
    Upload a training data file for streaming training.

    Upload CSV, JSON Lines, or Parquet files for memory-efficient training
    on large datasets. Returns a file reference ID that can be used with
    the /v1/anomaly/fit endpoint's training_file parameter.

    Supported formats:
    - CSV: Comma-separated values with header row
    - JSON Lines: One JSON object per line
    - Parquet: Columnar format for large datasets

    Example:
    ```bash
    curl -X POST http://localhost:11540/v1/anomaly/upload-training-data \\
        -F "file=@training_data.csv" \\
        -F "skip_columns=id,timestamp"
    ```
    """
    try:
        loader = get_streaming_loader()

        # Parse skip columns
        skip_cols = [c.strip() for c in skip_columns.split(",") if c.strip()]

        # Process upload
        file_ref = await loader.upload_file(
            file=file,
            skip_columns=skip_cols,
        )

        return {
            "object": "file_reference",
            "id": file_ref.id,
            "filename": file_ref.filename,
            "format": file_ref.format,
            "num_rows": file_ref.num_rows,
            "num_columns": file_ref.num_columns,
            "column_names": file_ref.column_names,
            "skipped_columns": skip_cols,
            "status": "uploaded",
        }

    except Exception as e:
        logger.error(f"Error in upload_training_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/anomaly/fit")
async def fit_anomaly_detector(request: AnomalyFitRequest):
    """
    Fit an anomaly detector on training data.

    Train an anomaly detection model on data assumed to be mostly normal.
    The model learns what "normal" looks like and can then detect deviations.

    Backends:
    - isolation_forest: Fast, works well out of the box (recommended)
    - one_class_svm: Good for small datasets
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Best for complex patterns, requires more data
    - vae: Variational Autoencoder with probabilistic scoring (ELBO-based)
    - copod: Copula-based, fast and parameter-free
    - hbos: Histogram-based, fastest
    - ecod: Empirical cumulative distribution

    Early Stopping (autoencoder/vae only):
    - validation_split: Fraction of data held out for validation (default: 0.1)
    - patience: Epochs without improvement before stopping (default: 10)
    - min_delta: Minimum change in validation loss for improvement (default: 1e-4)

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest",
        "data": [[1.0, 2.0], [1.1, 2.1], [0.9, 1.9], ...],
        "contamination": 0.1
    }
    ```

    After fitting, use /v1/anomaly/score to detect anomalies in new data.
    """
    try:
        cache_key = anomaly_service._make_cache_key(
            request.model, request.backend, request.normalization, request.scaler_type
        )

        # Validate that either data or training_file is provided
        if request.data is None and request.training_file is None:
            raise HTTPException(
                status_code=400,
                detail="Either 'data' or 'training_file' must be provided",
            )

        model = await anomaly_service.load_model(
            model_id=request.model,
            backend=request.backend,
            contamination=request.contamination,
            normalization=request.normalization,
            scaler_type=request.scaler_type,
            validation_split=request.validation_split,
            patience=request.patience,
            min_delta=request.min_delta,
        )

        # Fit from file or in-memory data
        if request.training_file:
            # Streaming training from file reference
            loader = get_streaming_loader()
            file_ref = loader.get_file_ref(request.training_file)
            if file_ref is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Training file not found: {request.training_file}",
                )

            result = await anomaly_service.fit_from_file(
                model=model,
                file_ref=file_ref,
                batch_size=request.batch_size,
                epochs=request.epochs,
            )
        else:
            # Prepare data (encode if dict-based, and fit the encoder)
            prepared_data = anomaly_service.prepare_data(
                data=request.data,
                schema=request.schema,
                cache_key=cache_key,
                fit_mode=True,  # Fit encoder on training data
            )

            # Fit model
            result = await anomaly_service.fit(
                model=model,
                data=prepared_data,
                epochs=request.epochs,
                batch_size=request.batch_size,
            )

        # Include encoder info in response if used
        encoder_info = None
        encoder = anomaly_service.get_encoder(cache_key)
        if encoder:
            encoder_info = {
                "schema": encoder.schema.features if encoder.schema else {},
                "features": list(encoder.schema.features.keys())
                if encoder.schema
                else [],
            }

        # Auto-save model to prevent data loss on restart
        await anomaly_service.auto_save_model(
            model=model,
            model_name=request.model,
            backend=request.backend,
            cache_key=cache_key,
        )

        return {
            "object": "fit_result",
            "model": request.model,
            "backend": request.backend,
            "samples_fitted": result["samples_fitted"],
            "training_time_ms": result["training_time_ms"],
            "model_params": result["model_params"],
            "encoder": encoder_info,
            "status": "fitted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in fit_anomaly_detector: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/anomaly/detect")
async def detect_anomalies(request: AnomalyScoreRequest):
    """
    Detect anomalies in data (returns only anomalous points).

    Same as /v1/anomaly/score but filters to return only points
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
    try:
        cache_key = anomaly_service._make_cache_key(
            request.model, request.backend, request.normalization, request.scaler_type
        )

        model = await anomaly_service.load_model(
            model_id=request.model,
            backend=request.backend,
            normalization=request.normalization,
            scaler_type=request.scaler_type,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first.",
            )

        # Prepare data (encode if dict-based)
        prepared_data = anomaly_service.prepare_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=False,  # Use existing encoder
        )

        # Detect anomalies
        results = await anomaly_service.detect(
            model=model,
            data=prepared_data,
            threshold=request.threshold,
        )

        return {
            "object": "list",
            "data": results,
            "total_count": len(results),
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "anomalies_detected": len(results),
                "threshold": request.threshold or model.threshold,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in detect_anomalies: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/anomaly/save")
async def save_anomaly_model(request: AnomalySaveRequest):
    """
    Save a fitted anomaly model to disk for production use.

    After fitting a model with /v1/anomaly/fit, save it to disk so it
    persists across server restarts.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest"
    }
    ```

    Models are saved to ~/.llamafarm/models/anomaly/ with auto-generated
    filenames based on the model name and backend.
    """
    try:
        cache_key = anomaly_service._make_cache_key(
            request.model, request.backend, request.normalization, request.scaler_type
        )

        from services.model_manager import ModelType, model_manager

        model = model_manager.get(ModelType.ANOMALY, cache_key)

        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' with backend '{request.backend}', "
                f"normalization '{request.normalization}', and scaler_type '{request.scaler_type}' "
                "not found in cache. Fit the model first with /v1/anomaly/fit",
            )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first.",
            )

        result = await anomaly_service.save_model(
            model=model,
            model_name=request.model,
            backend=request.backend,
            cache_key=cache_key,
        )

        return {
            "object": "save_result",
            "model": request.model,
            "backend": request.backend,
            "filename": result["filename"],
            "path": result["path"],
            "encoder_path": result["encoder_path"],
            "status": "saved",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in save_anomaly_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/anomaly/load")
async def load_anomaly_model(request: AnomalyLoadRequest):
    """
    Load a pre-trained anomaly model from disk.

    Load a previously saved model for production inference without
    re-training. The model path is automatically determined from the
    model name and backend - no user control over file paths.

    Example request:
    ```json
    {
        "model": "sensor-detector",
        "backend": "isolation_forest"
    }
    ```

    The model will be loaded from ~/.llamafarm/models/anomaly/ and cached
    for subsequent /v1/anomaly/score and /v1/anomaly/detect calls.
    """
    try:
        model, result = await anomaly_service.load_from_disk(
            model_name=request.model,
            backend=request.backend,
        )

        return {
            "object": "load_result",
            "model": request.model,
            "backend": request.backend,
            "normalization": result["normalization"],
            "filename": result["filename"],
            "is_fitted": result["is_fitted"],
            "threshold": result["threshold"],
            "encoder_loaded": result["encoder_loaded"],
            "encoder_schema": result["encoder_schema"],
            "status": "loaded",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in load_anomaly_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/v1/anomaly/models")
async def list_anomaly_models():
    """
    List all saved anomaly models available for loading.

    Returns models saved in the anomaly models directory.

    Response includes:
    - filename: Name of the saved model file
    - model_name: Extracted model name
    - backend: Detected backend type
    - size_bytes: File size
    - modified: Last modification timestamp
    """
    try:
        models = anomaly_service.list_saved_models()

        return {
            "object": "list",
            "data": models,
            "total_count": len(models),
        }

    except Exception as e:
        logger.error(f"Error in list_anomaly_models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/v1/anomaly/models/{filename}")
async def delete_anomaly_model(filename: str):
    """
    Delete a saved anomaly model from disk.

    Permanently removes the model file. This cannot be undone.

    Path Parameters:
    - filename: The model filename to delete (e.g., "my-model_isolation_forest.joblib")
    """
    try:
        result = await anomaly_service.delete_model(filename)

        return {
            "object": "delete_result",
            "filename": result["filename"],
            "encoder_deleted": result["encoder_deleted"],
            "status": "deleted",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_anomaly_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/v1/anomaly/explain")
async def explain_anomaly(request: AnomalyExplainRequest):
    """
    Explain why data points are flagged as anomalies using SHAP.

    SHAP (SHapley Additive exPlanations) provides feature importance
    showing which features contributed most to the anomaly score.

    Requires a trained anomaly detection model.

    Example request:
    ```json
    {
        "model_id": "my-anomaly-model",
        "data": [[95.0, 20.0, 150.0]],
        "feature_names": ["cpu", "memory", "latency"],
        "background_samples": 100
    }
    ```

    Response:
    ```json
    {
        "object": "anomaly_explanation",
        "explanations": [{
            "features": [
                {"feature": "cpu", "importance": 0.82, "value": 95.0, "direction": "high"},
                {"feature": "latency", "importance": 0.45, "value": 150.0, "direction": "high"},
                {"feature": "memory", "importance": 0.12, "value": 20.0, "direction": "low"}
            ],
            "top_contributors": [...]
        }]
    }
    ```
    """
    try:
        cache_key = anomaly_service._make_cache_key(
            request.model_id,
            request.backend,
            request.normalization,
            request.scaler_type,
        )

        from services.model_manager import ModelType, model_manager

        model = model_manager.get(ModelType.ANOMALY, cache_key)

        if model is None:
            raise HTTPException(
                status_code=404,
                detail=f"Anomaly model '{request.model_id}' not found. Train a model first with /v1/anomaly/fit",
            )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{request.model_id}' is not trained. Train it first with /v1/anomaly/fit",
            )

        explanations = await anomaly_service.explain(
            model=model,
            data=request.data,
            feature_names=request.feature_names,
            background_samples=request.background_samples,
            nsamples=request.nsamples,
        )

        return {
            "object": "anomaly_explanation",
            "model_id": request.model_id,
            "explanations": explanations,
            "total_points": len(explanations),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in explain_anomaly: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
