"""CatBoost Classification/Regression router.

Provides endpoints for CatBoost gradient boosting models with:
- Native categorical support
- Incremental learning
- GPU acceleration (when available)
"""

import asyncio
import logging
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api_types.catboost import (
    CatBoostDeleteResponse,
    CatBoostFeatureImportance,
    CatBoostFeatureImportanceResponse,
    CatBoostFitRequest,
    CatBoostFitResponse,
    CatBoostInfoResponse,
    CatBoostLoadRequest,
    CatBoostLoadResponse,
    CatBoostModelInfo,
    CatBoostModelsResponse,
    CatBoostPrediction,
    CatBoostPredictRequest,
    CatBoostPredictResponse,
    CatBoostUpdateRequest,
    CatBoostUpdateResponse,
)
from models.catboost_model import CatBoostModel, get_catboost_info
from services.error_handler import handle_endpoint_errors
from utils.model_cache import ModelCache

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level state (set by server.py)
_catboost_cache: ModelCache[CatBoostModel] | None = None
_model_load_lock: asyncio.Lock | None = None
_catboost_models_dir: Path | None = None


def set_catboost_state(
    cache: ModelCache[CatBoostModel],
    lock: asyncio.Lock,
    models_dir: Path,
) -> None:
    """Set shared state from server.py."""
    global _catboost_cache, _model_load_lock, _catboost_models_dir
    _catboost_cache = cache
    _model_load_lock = lock
    _catboost_models_dir = models_dir


def _get_model_path(model_id: str) -> Path:
    """Get the path for a CatBoost model file."""
    if _catboost_models_dir is None:
        raise RuntimeError("CatBoost models directory not set")
    # Prevent path traversal
    safe_name = Path(model_id).name
    if safe_name != model_id or ".." in model_id:
        raise ValueError(f"Invalid model_id: {model_id}")
    return _catboost_models_dir / f"{safe_name}.joblib"


def _list_saved_models() -> list[CatBoostModelInfo]:
    """List all saved CatBoost models."""
    if _catboost_models_dir is None:
        return []

    models = []
    if _catboost_models_dir.exists():
        for path in _catboost_models_dir.glob("*.joblib"):
            model_id = path.stem
            models.append(
                CatBoostModelInfo(
                    model_id=model_id,
                    model_type="unknown",  # Would need to load to know
                    path=str(path),
                )
            )

    return models


@router.get("/v1/catboost/info")
@handle_endpoint_errors("catboost-info")
async def get_info() -> CatBoostInfoResponse:
    """Get CatBoost availability and capabilities."""
    info = get_catboost_info()
    return CatBoostInfoResponse(**info)


@router.get("/v1/catboost/models")
@handle_endpoint_errors("catboost-models")
async def list_models() -> CatBoostModelsResponse:
    """List all saved CatBoost models."""
    models = _list_saved_models()
    return CatBoostModelsResponse(models=models)


@router.post("/v1/catboost/fit")
@handle_endpoint_errors("catboost-fit")
async def fit_model(request: CatBoostFitRequest) -> CatBoostFitResponse:
    """Train a CatBoost model.

    Trains a new CatBoost classifier or regressor and saves it automatically.
    """
    if _catboost_cache is None:
        raise HTTPException(status_code=500, detail="CatBoost not initialized")

    # Generate model ID if not provided
    model_id = request.model_id or f"catboost-{uuid.uuid4().hex[:8]}"

    try:
        # Create model
        model = CatBoostModel(
            model_id=model_id,
            model_type=request.model_type,
            iterations=request.iterations,
            learning_rate=request.learning_rate,
            depth=request.depth,
            cat_features=request.cat_features,
            random_state=request.random_state,
        )

        await model.load()

        # Prepare validation set if requested
        eval_set = None
        train_data = request.data
        train_labels = request.labels

        if request.validation_fraction is not None:
            import random
            n_samples = len(train_data)
            n_val = int(n_samples * request.validation_fraction)
            indices = list(range(n_samples))
            # Use seeded random for reproducible validation splits
            rng = random.Random(request.random_state if request.random_state is not None else 42)
            rng.shuffle(indices)

            val_indices = indices[:n_val]
            train_indices = indices[n_val:]

            train_data = [train_data[i] for i in train_indices]
            train_labels = [train_labels[i] for i in train_indices]
            val_data = [request.data[i] for i in val_indices]
            val_labels = [request.labels[i] for i in val_indices]
            eval_set = (val_data, val_labels)

        # Fit the model
        fit_result = await model.fit(
            X=train_data,
            y=train_labels,
            feature_names=request.feature_names,
            eval_set=eval_set,
            early_stopping_rounds=request.early_stopping_rounds,
        )

        # Save the model
        save_path = _get_model_path(model_id)
        await model.save(save_path)

        # Cache the model
        cache_key = f"catboost:{model_id}"
        _catboost_cache[cache_key] = model

        return CatBoostFitResponse(
            model_id=model_id,
            model_type=request.model_type,
            samples_fitted=fit_result["samples_fitted"],
            n_features=fit_result["n_features"],
            iterations=fit_result["iterations"],
            best_iteration=fit_result.get("best_iteration"),
            classes=fit_result.get("classes"),
            saved_path=str(save_path),
            fit_time_ms=fit_result["fit_time_ms"],
        )

    except Exception as e:
        logger.error(f"CatBoost fit failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to train CatBoost model") from e


@router.post("/v1/catboost/predict")
@handle_endpoint_errors("catboost-predict")
async def predict(request: CatBoostPredictRequest) -> CatBoostPredictResponse:
    """Make predictions with a CatBoost model."""
    if _catboost_cache is None:
        raise HTTPException(status_code=500, detail="CatBoost not initialized")

    start_time = time.time()
    cache_key = f"catboost:{request.model_id}"

    # Try to get from cache
    model = _catboost_cache.get(cache_key)

    # If not in cache, try to load from disk
    if model is None:
        model_path = _get_model_path(request.model_id)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

        model = CatBoostModel(model_id=request.model_id)
        await model.load_from_path(model_path)
        _catboost_cache[cache_key] = model

    try:
        if request.return_proba:
            probas = await model.predict_proba(request.data)
            predictions = []
            classes = model.classes or list(range(probas.shape[1]))

            for i, proba in enumerate(probas):
                pred_idx = proba.argmax()
                predictions.append(
                    CatBoostPrediction(
                        sample_index=i,
                        prediction=classes[pred_idx],
                        probabilities={str(c): float(p) for c, p in zip(classes, proba, strict=True)},
                    )
                )
        else:
            preds = await model.predict(request.data)
            predictions = [
                CatBoostPrediction(sample_index=i, prediction=pred)
                for i, pred in enumerate(preds.tolist() if hasattr(preds, "tolist") else list(preds))
            ]

        predict_time_ms = (time.time() - start_time) * 1000

        return CatBoostPredictResponse(
            model_id=request.model_id,
            predictions=predictions,
            predict_time_ms=predict_time_ms,
        )

    except Exception as e:
        logger.error(f"CatBoost predict failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@router.post("/v1/catboost/update")
@handle_endpoint_errors("catboost-update")
async def update_model(request: CatBoostUpdateRequest) -> CatBoostUpdateResponse:
    """Incrementally update a CatBoost model with new data."""
    if _catboost_cache is None:
        raise HTTPException(status_code=500, detail="CatBoost not initialized")

    cache_key = f"catboost:{request.model_id}"

    # Try to get from cache
    model = _catboost_cache.get(cache_key)

    # If not in cache, try to load from disk
    if model is None:
        model_path = _get_model_path(request.model_id)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

        model = CatBoostModel(model_id=request.model_id)
        await model.load_from_path(model_path)
        _catboost_cache[cache_key] = model

    try:
        # Perform incremental update
        update_result = await model.update(
            X=request.data,
            y=request.labels,
            sample_weight=request.sample_weight,
        )

        # Save updated model
        save_path = _get_model_path(request.model_id)
        await model.save(save_path)

        return CatBoostUpdateResponse(
            model_id=request.model_id,
            samples_added=update_result["samples_added"],
            trees_before=update_result["trees_before"],
            trees_after=update_result["trees_after"],
            update_time_ms=update_result["update_time_ms"],
        )

    except Exception as e:
        logger.error(f"CatBoost update failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Update failed") from e


@router.post("/v1/catboost/load")
@handle_endpoint_errors("catboost-load")
async def load_model(request: CatBoostLoadRequest) -> CatBoostLoadResponse:
    """Load a CatBoost model from disk."""
    if _catboost_cache is None:
        raise HTTPException(status_code=500, detail="CatBoost not initialized")

    model_path = _get_model_path(request.model_id)
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {request.model_id}")

    try:
        model = CatBoostModel(model_id=request.model_id)
        await model.load_from_path(model_path)

        cache_key = f"catboost:{request.model_id}"
        _catboost_cache[cache_key] = model

        return CatBoostLoadResponse(
            model_id=request.model_id,
            model_type=model.model_type,
            n_features=model.n_features or 0,
            classes=model.classes,
        )

    except Exception as e:
        logger.error(f"CatBoost load failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Load failed") from e


@router.delete("/v1/catboost/{model_id}")
@handle_endpoint_errors("catboost-delete")
async def delete_model(model_id: str) -> CatBoostDeleteResponse:
    """Delete a CatBoost model."""
    if _catboost_cache is None:
        raise HTTPException(status_code=500, detail="CatBoost not initialized")

    # Remove from cache
    cache_key = f"catboost:{model_id}"
    if cache_key in _catboost_cache:
        model = _catboost_cache.pop(cache_key)
        if model:
            await model.unload()

    # Delete from disk
    model_path = _get_model_path(model_id)
    deleted = False
    if model_path.exists():
        model_path.unlink()
        deleted = True
        logger.info(f"Deleted CatBoost model: {model_id}")

    return CatBoostDeleteResponse(model_id=model_id, deleted=deleted)


@router.get("/v1/catboost/{model_id}/importance")
@handle_endpoint_errors("catboost-importance")
async def get_feature_importance(model_id: str) -> CatBoostFeatureImportanceResponse:
    """Get feature importance for a CatBoost model."""
    if _catboost_cache is None:
        raise HTTPException(status_code=500, detail="CatBoost not initialized")

    cache_key = f"catboost:{model_id}"

    # Try to get from cache
    model = _catboost_cache.get(cache_key)

    # If not in cache, try to load from disk
    if model is None:
        model_path = _get_model_path(model_id)
        if not model_path.exists():
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

        model = CatBoostModel(model_id=model_id)
        await model.load_from_path(model_path)
        _catboost_cache[cache_key] = model

    try:
        importance = await model.get_feature_importance()

        return CatBoostFeatureImportanceResponse(
            model_id=model_id,
            importances=[
                CatBoostFeatureImportance(feature=name, importance=imp)
                for name, imp in importance
            ],
            importance_type="FeatureImportance",
        )

    except Exception as e:
        logger.error(f"Feature importance failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get feature importance") from e
