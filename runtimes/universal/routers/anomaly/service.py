"""Anomaly detection service for model loading and business logic.

Handles anomaly model management including:
- Training anomaly detection models
- Scoring and detecting anomalies
- Saving and loading models
- SHAP-based explanations
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from services.model_manager import ModelManager, ModelType, model_manager
from utils.device import get_optimal_device

if TYPE_CHECKING:
    from models import AnomalyModel

logger = logging.getLogger(__name__)

# Data directory for saving models
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
ANOMALY_MODELS_DIR = _LF_DATA_DIR / "models" / "anomaly"

# Global device cache
_current_device: str | None = None

# Global encoder cache (for dict-based data)
_encoders: dict[str, Any] = {}

# Global streaming data loader
_streaming_loader = None


def get_device() -> str:
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


def get_streaming_loader():
    """Get or create the global streaming data loader."""
    global _streaming_loader
    if _streaming_loader is None:
        from utils.streaming_data import StreamingDataLoader

        _streaming_loader = StreamingDataLoader()
    return _streaming_loader


def _sanitize_model_name(name: str) -> str:
    """Sanitize model name to create a safe filename.

    Only allows alphanumeric characters, hyphens, and underscores.
    This prevents path traversal and ensures consistent naming.
    """
    return "".join(c for c in name if c.isalnum() or c in "-_")


def _get_model_path(model_name: str, backend: str) -> Path:
    """Get the path for a model file based on name and backend.

    The path is always within ANOMALY_MODELS_DIR - users cannot control it.
    """
    safe_name = _sanitize_model_name(model_name)
    safe_backend = _sanitize_model_name(backend)
    filename = f"{safe_name}_{safe_backend}"
    return ANOMALY_MODELS_DIR / filename


class AnomalyService:
    """Service for anomaly detection model operations."""

    def __init__(self, manager: ModelManager | None = None):
        """Initialize the anomaly service.

        Args:
            manager: Optional ModelManager instance. Uses global singleton if not provided.
        """
        self._manager = manager or model_manager

    def _make_cache_key(
        self,
        model_id: str,
        backend: str,
        normalization: str | None = None,
        scaler_type: str | None = None,
    ) -> str:
        """Generate a cache key for an anomaly model."""
        return self._manager.make_anomaly_cache_key(
            model_id, backend, normalization, scaler_type
        )

    async def load_model(
        self,
        model_id: str,
        backend: str = "isolation_forest",
        contamination: float = 0.1,
        threshold: float | None = None,
        normalization: str = "standardization",
        scaler_type: str = "robust",
        validation_split: float = 0.1,
        patience: int = 10,
        min_delta: float = 1e-4,
    ) -> "AnomalyModel":
        """Load an anomaly detection model.

        Args:
            model_id: Model identifier or path to pre-trained model
            backend: Anomaly detection backend
            contamination: Expected proportion of anomalies
            threshold: Custom anomaly threshold
            normalization: Score normalization method (standardization, zscore, raw)
            scaler_type: Input data scaler type (robust or standard)
            validation_split: Fraction of data for validation (autoencoder/vae only)
            patience: Epochs without improvement before stopping (autoencoder/vae only)
            min_delta: Minimum change in validation loss for improvement

        Returns:
            Loaded AnomalyModel instance
        """
        from models import AnomalyModel

        cache_key = self._make_cache_key(model_id, backend, normalization, scaler_type)

        async def loader():
            logger.info(f"Loading anomaly model ({backend}): {model_id}")
            device = get_device()

            model = AnomalyModel(
                model_id=model_id,
                device=device,
                backend=backend,
                contamination=contamination,
                threshold=threshold,
                normalization=normalization,
                scaler_type=scaler_type,
                validation_split=validation_split,
                patience=patience,
                min_delta=min_delta,
            )

            await model.load()
            return model

        return await self._manager.get_or_load(ModelType.ANOMALY, cache_key, loader)

    def prepare_data(
        self,
        data: list[list[float]] | list[dict],
        schema: dict[str, str] | None,
        cache_key: str,
        fit_mode: bool = False,
    ) -> list[list[float]]:
        """Prepare data for anomaly detection by encoding if needed.

        Args:
            data: Raw data (numeric arrays or dicts)
            schema: Feature encoding schema (required for dict data during fit)
            cache_key: Cache key for storing/retrieving encoder
            fit_mode: If True, fit the encoder on the data. If False, use existing encoder.

        Returns:
            Encoded numeric data as list of lists
        """
        from fastapi import HTTPException

        from utils.feature_encoder import FeatureEncoder

        # If data is already numeric, return as-is
        if not data:
            return []

        if isinstance(data[0], list):
            # Already numeric arrays
            return data

        # Dict-based data - need to encode
        if fit_mode:
            # Require schema for training
            if schema is None:
                raise HTTPException(
                    status_code=400,
                    detail="Schema is required when fitting with dict-based data. "
                    "Example: schema = {'time_ms': 'numeric', 'user_agent': 'hash'}",
                )
            # Fit encoder on training data
            encoder = FeatureEncoder()
            encoder.fit(data, schema)
            _encoders[cache_key] = encoder
            logger.info(f"Fitted feature encoder for {cache_key} with schema: {schema}")
        else:
            # Use existing encoder (schema already learned during fit)
            if cache_key not in _encoders:
                raise HTTPException(
                    status_code=400,
                    detail=f"No encoder found for model '{cache_key}'. "
                    "Train with /v1/anomaly/fit using dict data first, or pass schema.",
                )
            encoder = _encoders[cache_key]

        # Transform data
        encoded = encoder.transform(data)
        return encoded.tolist()

    def get_encoder(self, cache_key: str):
        """Get encoder for a model if one exists."""
        return _encoders.get(cache_key)

    async def auto_save_model(
        self,
        model: "AnomalyModel",
        model_name: str,
        backend: str,
        cache_key: str,
    ) -> None:
        """Auto-save anomaly model after fit to prevent data loss."""
        # Create models directory if needed
        ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name
        save_path = _get_model_path(model_name, backend)
        await model.save(str(save_path))

        # Determine actual saved file path for logging
        if backend in ("autoencoder", "vae"):
            actual_path = save_path.with_suffix(".pt")
        else:
            actual_path = save_path.with_suffix(".joblib")
            if not actual_path.exists():
                actual_path = save_path.with_suffix(".pkl")

        logger.debug(f"Model saved to {actual_path}")

        # Save encoder if one exists for this model
        if cache_key in _encoders:
            encoder = _encoders[cache_key]
            encoder_save_path = save_path.parent / f"{save_path.name}_encoder.json"
            encoder.save(encoder_save_path)
            logger.debug(f"Feature encoder saved to {encoder_save_path}")

    async def score(
        self,
        model: "AnomalyModel",
        data: list[list[float]],
        threshold: float | None = None,
    ) -> list[dict]:
        """Score data points for anomalies.

        Returns:
            List of score results with index, score, is_anomaly, raw_score
        """
        results = await model.score(data=data, threshold=threshold)

        # Format response
        return [
            {
                "index": r.index,
                "score": r.score,
                "is_anomaly": r.is_anomaly,
                "raw_score": r.raw_score,
            }
            for r in results
        ]

    async def detect(
        self,
        model: "AnomalyModel",
        data: list[list[float]],
        threshold: float | None = None,
    ) -> list[dict]:
        """Detect anomalies in data (returns only anomalous points).

        Returns:
            List of anomalous points with index, score, raw_score
        """
        results = await model.detect(data=data, threshold=threshold)

        return [
            {
                "index": r.index,
                "score": r.score,
                "raw_score": r.raw_score,
            }
            for r in results
        ]

    async def fit(
        self,
        model: "AnomalyModel",
        data: list[list[float]],
        epochs: int = 100,
        batch_size: int = 32,
    ) -> dict:
        """Fit model on training data.

        Returns:
            Fit result dict with samples_fitted, training_time_ms, model_params
        """
        result = await model.fit(
            data=data,
            epochs=epochs,
            batch_size=batch_size,
        )

        return {
            "samples_fitted": result.samples_fitted,
            "training_time_ms": result.training_time_ms,
            "model_params": result.model_params,
        }

    async def fit_from_file(
        self,
        model: "AnomalyModel",
        file_ref: Any,
        batch_size: int = 32,
        epochs: int = 100,
    ) -> dict:
        """Fit model from streaming file reference.

        Returns:
            Fit result dict
        """
        result = await model.fit_from_file(
            file_ref=file_ref,
            batch_size=batch_size,
            epochs=epochs,
        )

        return {
            "samples_fitted": result.samples_fitted,
            "training_time_ms": result.training_time_ms,
            "model_params": result.model_params,
        }

    async def save_model(
        self,
        model: "AnomalyModel",
        model_name: str,
        backend: str,
        cache_key: str,
    ) -> dict:
        """Save a fitted model to disk.

        Returns:
            Save result dict with filename, path, encoder_path
        """
        # Create models directory if needed
        ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Generate path from model name (no user-controlled paths)
        save_path = _get_model_path(model_name, backend)
        await model.save(str(save_path))

        # Determine actual saved file
        if backend in ("autoencoder", "vae"):
            actual_path = save_path.with_suffix(".pt")
        else:
            actual_path = save_path.with_suffix(".joblib")
            if not actual_path.exists():
                actual_path = save_path.with_suffix(".pkl")

        # Save encoder if one exists for this model
        encoder_path = None
        if cache_key in _encoders:
            encoder = _encoders[cache_key]
            encoder_save_path = save_path.parent / f"{save_path.name}_encoder.json"
            encoder.save(encoder_save_path)
            encoder_path = str(encoder_save_path)
            logger.info(f"Saved feature encoder to {encoder_save_path}")

        return {
            "filename": actual_path.name,
            "path": str(actual_path),
            "encoder_path": encoder_path,
        }

    async def load_from_disk(
        self,
        model_name: str,
        backend: str,
    ) -> tuple["AnomalyModel", dict]:
        """Load a pre-trained model from disk.

        Returns:
            Tuple of (model, load_result dict)
        """
        import asyncio

        from fastapi import HTTPException

        from models import AnomalyModel
        from utils.feature_encoder import FeatureEncoder

        # Generate path from model name (no user-controlled paths)
        base_path = _get_model_path(model_name, backend)

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
                detail=f"Model '{model_name}' with backend '{backend}' not found. "
                f"Available models: {available}",
            )

        # Load with double-check locking
        lock = asyncio.Lock()
        async with lock:
            logger.info(f"Loading pre-trained anomaly model: {model_path}")
            device = get_device()

            model = AnomalyModel(
                model_id=str(model_path),  # Pass path as model_id for loading
                device=device,
                backend=backend,
            )

            await model.load()

            # Cache the model using its actual normalization
            cache_key = self._make_cache_key(model_name, backend, model.normalization)

            # Store in manager
            self._manager.set(ModelType.ANOMALY, cache_key, model)

        # Try to load encoder if one exists
        encoder_loaded = False
        encoder_schema = None
        encoder_path = base_path.parent / f"{base_path.name}_encoder.json"
        if encoder_path.exists():
            encoder = FeatureEncoder.load(encoder_path)
            _encoders[cache_key] = encoder
            encoder_loaded = True
            encoder_schema = encoder.schema
            logger.info(f"Loaded feature encoder from {encoder_path}")

        return model, {
            "normalization": model.normalization,
            "filename": model_path.name,
            "is_fitted": model.is_fitted,
            "threshold": model.threshold,
            "encoder_loaded": encoder_loaded,
            "encoder_schema": encoder_schema,
        }

    def list_saved_models(self) -> list[dict]:
        """List all saved anomaly models.

        Returns:
            List of model info dicts
        """
        ANOMALY_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        models = []
        for path in ANOMALY_MODELS_DIR.glob("*"):
            if path.is_file() and path.suffix in (".pt", ".pkl", ".joblib"):
                # Try to extract model name and backend from filename
                # Format: {model_name}_{backend}.{ext}
                name = path.stem
                parts = name.rsplit("_", 1)
                model_name = parts[0] if len(parts) == 2 else name
                backend = parts[1] if len(parts) == 2 else "unknown"

                stat = path.stat()
                models.append(
                    {
                        "filename": path.name,
                        "model_name": model_name,
                        "backend": backend,
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )

        return sorted(models, key=lambda x: x["modified"], reverse=True)

    async def delete_model(self, filename: str) -> dict:
        """Delete a saved model from disk.

        Returns:
            Delete result dict
        """
        from fastapi import HTTPException

        # Sanitize filename to prevent path traversal
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "-_.")
        model_path = ANOMALY_MODELS_DIR / safe_filename

        if not model_path.exists() or not model_path.is_file():
            raise HTTPException(
                status_code=404,
                detail=f"Model file '{filename}' not found",
            )

        # Also delete encoder if exists
        encoder_path = (
            model_path.with_suffix("").parent / f"{model_path.stem}_encoder.json"
        )

        model_path.unlink()
        encoder_deleted = False
        if encoder_path.exists():
            encoder_path.unlink()
            encoder_deleted = True

        return {
            "filename": filename,
            "encoder_deleted": encoder_deleted,
        }

    async def explain(
        self,
        model: "AnomalyModel",
        data: list[list[float]],
        feature_names: list[str] | None = None,
        background_samples: int = 100,
        nsamples: int = 100,
    ) -> list[dict]:
        """Explain anomaly scores using SHAP.

        Returns:
            List of explanation dicts
        """
        import numpy as np
        from fastapi import HTTPException

        from utils.anomaly_explainer import create_explainer_for_sklearn

        # Get the sklearn model (IsolationForest, etc.)
        sklearn_model = model.get_sklearn_model()
        if sklearn_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model does not support SHAP explanations (no sklearn model)",
            )

        # Convert data to numpy array
        data_array = np.array(data, dtype=np.float64)

        # Get background data from the model's training data
        training_data = model.get_training_data()
        if training_data is None or len(training_data) == 0:
            raise HTTPException(
                status_code=400,
                detail="Model has no training data for SHAP background",
            )

        # Create explainer
        explainer = create_explainer_for_sklearn(
            model=sklearn_model,
            background_data=training_data,
            feature_names=feature_names,
            n_background_samples=background_samples,
        )

        # Get explanations
        explanations = explainer.explain(data_array, nsamples=nsamples)

        return explanations


# Global service instance
anomaly_service = AnomalyService()
