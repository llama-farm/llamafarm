"""
Anomaly detection endpoints.

Train and use anomaly detection models for detecting outliers in data.
"""

from fastapi import APIRouter, HTTPException

from core.logging import UniversalRuntimeLogger
from models import AnomalyModel
from state import (
    ANOMALY_MODELS_DIR,
    get_device,
    get_encoders_cache,
    get_model_load_lock,
    get_models_cache,
    sanitize_filename,
    validate_path_within_directory,
)
from utils.feature_encoder import FeatureEncoder

from .service import (
    auto_save_anomaly_model,
    get_model_path,
    load_anomaly,
    make_anomaly_cache_key,
    prepare_anomaly_data,
)
from .types import (
    AnomalyFitRequest,
    AnomalyLoadRequest,
    AnomalySaveRequest,
    AnomalyScoreRequest,
)

router = APIRouter()
logger = UniversalRuntimeLogger("universal-runtime.anomaly")


@router.post("/v1/anomaly/score")
async def score_anomalies(request: AnomalyScoreRequest):
    """
    Score data points for anomalies.

    Detects anomalies in data using various algorithms:
    - isolation_forest: Fast tree-based method, good general purpose
    - one_class_svm: Support vector machine for outlier detection
    - local_outlier_factor: Density-based, good for clustering anomalies
    - autoencoder: Neural network, best for complex patterns

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
        cache_key = make_anomaly_cache_key(
            request.model, request.backend, request.normalization
        )

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            normalization=request.normalization,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first or load a pre-trained model.",
            )

        # Prepare data (encode if dict-based)
        prepared_data = prepare_anomaly_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=False,  # Use existing encoder
        )

        # Score data
        results = await model.score(
            data=prepared_data,
            threshold=request.threshold,
        )

        # Format response
        data = [
            {
                "index": r.index,
                "score": r.score,
                "is_anomaly": r.is_anomaly,
                "raw_score": r.raw_score,
            }
            for r in results
        ]

        # Summary statistics
        anomaly_count = sum(1 for r in results if r.is_anomaly)

        return {
            "object": "list",
            "data": data,
            "total_count": len(data),
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "total_points": len(data),
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / len(data) if data else 0,
                "threshold": request.threshold or model.threshold,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in score_anomalies: {e}", exc_info=True)
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
        cache_key = make_anomaly_cache_key(
            request.model, request.backend, request.normalization
        )

        # Prepare data (encode if dict-based, and fit the encoder)
        prepared_data = prepare_anomaly_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=True,  # Fit encoder on training data
        )

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            contamination=request.contamination,
            normalization=request.normalization,
        )

        # Fit model
        result = await model.fit(
            data=prepared_data,
            epochs=request.epochs,
            batch_size=request.batch_size,
        )

        # Include encoder info in response if used
        encoder_info = None
        encoders_cache = get_encoders_cache()
        if cache_key in encoders_cache:
            encoder = encoders_cache[cache_key]
            encoder_info = {
                "schema": encoder.schema.features if encoder.schema else {},
                "features": list(encoder.schema.features.keys())
                if encoder.schema
                else [],
            }

        # Auto-save model to prevent data loss on restart
        # This is mandatory - models must persist across server restarts
        await auto_save_anomaly_model(
            model=model,
            model_name=request.model,
            backend=request.backend,
            cache_key=cache_key,
        )

        return {
            "object": "fit_result",
            "model": request.model,
            "backend": request.backend,
            "samples_fitted": result.samples_fitted,
            "training_time_ms": result.training_time_ms,
            "model_params": result.model_params,
            "encoder": encoder_info,
            "status": "fitted",
        }

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
        cache_key = make_anomaly_cache_key(
            request.model, request.backend, request.normalization
        )

        model = await load_anomaly(
            model_id=request.model,
            backend=request.backend,
            normalization=request.normalization,
        )

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first.",
            )

        # Prepare data (encode if dict-based)
        prepared_data = prepare_anomaly_data(
            data=request.data,
            schema=request.schema,
            cache_key=cache_key,
            fit_mode=False,  # Use existing encoder
        )

        # Detect anomalies
        results = await model.detect(
            data=prepared_data,
            threshold=request.threshold,
        )

        # Format response
        data = [
            {
                "index": r.index,
                "score": r.score,
                "raw_score": r.raw_score,
            }
            for r in results
        ]

        return {
            "object": "list",
            "data": data,
            "total_count": len(data),
            "model": request.model,
            "backend": request.backend,
            "summary": {
                "anomalies_detected": len(data),
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
        cache_key = make_anomaly_cache_key(
            request.model, request.backend, request.normalization
        )

        models_cache = get_models_cache()
        encoders_cache = get_encoders_cache()

        if cache_key not in models_cache:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' with backend '{request.backend}' and "
                f"normalization '{request.normalization}' not found in cache. "
                "Fit the model first with /v1/anomaly/fit",
            )

        model = models_cache[cache_key]

        if not model.is_fitted:
            raise HTTPException(
                status_code=400,
                detail="Model not fitted. Call /v1/anomaly/fit first.",
            )

        # Create models directory if needed
        ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name (no user-controlled paths)
        save_path = get_model_path(request.model, request.backend)
        await model.save(str(save_path))

        # Determine actual saved file
        if request.backend == "autoencoder":
            actual_path = save_path.with_suffix(".pt")
        else:
            actual_path = save_path.with_suffix(".joblib")
            if not actual_path.exists():
                actual_path = save_path.with_suffix(".pkl")

        # Save encoder if one exists for this model
        encoder_path = None
        if cache_key in encoders_cache:
            encoder = encoders_cache[cache_key]
            encoder_save_path = save_path.parent / f"{save_path.name}_encoder.json"
            encoder.save(encoder_save_path)
            encoder_path = str(encoder_save_path)
            logger.info(f"Saved feature encoder to {encoder_save_path}")

        return {
            "object": "save_result",
            "model": request.model,
            "backend": request.backend,
            "filename": actual_path.name,
            "path": str(actual_path),
            "encoder_path": encoder_path,
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
        models_cache = get_models_cache()
        encoders_cache = get_encoders_cache()
        model_load_lock = get_model_load_lock()

        # Generate path from model name (no user-controlled paths)
        base_path = get_model_path(request.model, request.backend)

        # Determine actual file (check for different extensions)
        model_path = None
        for ext in [".joblib", ".pkl", ".pt"]:
            candidate = base_path.with_suffix(ext)
            if candidate.exists():
                model_path = candidate
                break

        if model_path is None:
            available = (
                [f.name for f in ANOMALY_MODELS_DIR.glob("*") if f.is_file()]
                if ANOMALY_MODELS_DIR.exists()
                else []
            )
            raise HTTPException(
                status_code=404,
                detail=f"Model '{request.model}' with backend '{request.backend}' not found. "
                f"Available models: {available}",
            )

        async with model_load_lock:
            logger.info(f"Loading pre-trained anomaly model: {model_path}")
            device = get_device()

            model = AnomalyModel(
                model_id=str(model_path),  # Pass path as model_id for loading
                device=device,
                backend=request.backend,
            )

            await model.load()

            # Use the model's actual normalization (loaded from file) for the cache key
            cache_key = make_anomaly_cache_key(
                request.model, request.backend, model.normalization
            )

            # Remove existing model from cache if present
            if cache_key in models_cache:
                await models_cache[cache_key].unload()
                del models_cache[cache_key]

            models_cache[cache_key] = model

        # Try to load encoder if one exists
        encoder_loaded = False
        encoder_schema = None
        # Derive encoder path from base path (same name pattern)
        encoder_path = base_path.parent / f"{base_path.name}_encoder.json"
        if encoder_path.exists():
            encoder = FeatureEncoder.load(encoder_path)
            encoders_cache[cache_key] = encoder
            encoder_loaded = True
            encoder_schema = encoder.schema
            logger.info(f"Loaded feature encoder from {encoder_path}")

        return {
            "object": "load_result",
            "model": request.model,
            "backend": request.backend,
            "normalization": model.normalization,
            "filename": model_path.name,
            "is_fitted": model.is_fitted,
            "threshold": model.threshold,
            "encoder_loaded": encoder_loaded,
            "encoder_schema": encoder_schema,
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

    Returns models saved in the ANOMALY_MODELS_DIR directory.

    Response includes:
    - filename: Name of the saved model file
    - size_bytes: File size
    - modified: Last modification timestamp
    - backend: Detected backend type (from file extension)
    """
    try:
        ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        models = []
        for path in ANOMALY_MODELS_DIR.glob("*"):
            if path.is_file() and path.suffix in (".pt", ".pkl", ".joblib"):
                stat = path.stat()

                # Detect backend from extension
                backend = "autoencoder" if path.suffix == ".pt" else "sklearn"

                models.append(
                    {
                        "filename": path.name,
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                        "backend": backend,
                    }
                )

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x["modified"], reverse=True)

        return {
            "object": "list",
            "data": models,
            "models_dir": str(ANOMALY_MODELS_DIR),
            "total": len(models),
        }

    except Exception as e:
        logger.error(f"Error in list_anomaly_models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.delete("/v1/anomaly/models/{filename}")
async def delete_anomaly_model(filename: str):
    """
    Delete a saved anomaly model.

    Removes the model file from disk. Does not affect cached models.
    """
    try:
        # Sanitize filename to prevent path traversal attacks
        # Use sanitize_filename to preserve extension dots (.joblib)
        safe_filename = sanitize_filename(filename)
        if not safe_filename:
            raise HTTPException(
                status_code=400,
                detail="Invalid filename",
            )

        # Also reject any path separators or special directory names
        if (
            "/" in filename
            or "\\" in filename
            or ".." in filename
            or safe_filename == "."
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid filename: path separators not allowed",
            )

        model_path = ANOMALY_MODELS_DIR / safe_filename

        # Validate the resolved path is still within the safe directory
        try:
            resolved_path = validate_path_within_directory(
                model_path, ANOMALY_MODELS_DIR
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        if not resolved_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Model file not found: {safe_filename}",
            )

        resolved_path.unlink()

        return {
            "object": "delete_result",
            "filename": safe_filename,
            "deleted": True,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_anomaly_model: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e
