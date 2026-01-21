"""Classifier service for text classification using SetFit.

Handles:
- Model loading and caching
- Fit (training)
- Predict (inference)
- Save/load to disk
"""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from services.model_manager import ModelManager, ModelType, model_manager
from utils.device import get_optimal_device

if TYPE_CHECKING:
    from models import ClassifierModel

logger = logging.getLogger(__name__)

# Global device cache
_current_device: str | None = None

# Model storage directory
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
CLASSIFIER_MODELS_DIR = _LF_DATA_DIR / "models" / "classifier"


def get_device() -> str:
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


def _sanitize_model_name(name: str) -> str:
    """Sanitize model name to create a safe filename.

    Only allows alphanumeric characters, hyphens, and underscores.
    This prevents path traversal and ensures consistent naming.
    """
    return "".join(c for c in name if c.isalnum() or c in "-_")


def _get_classifier_path(model_name: str) -> Path:
    """Get the path for a classifier model directory.

    The path is always within CLASSIFIER_MODELS_DIR - users cannot control it.
    """
    safe_name = _sanitize_model_name(model_name)
    return CLASSIFIER_MODELS_DIR / safe_name


def _validate_path_within_directory(path: Path, safe_dir: Path) -> Path:
    """Validate that a path is within the allowed directory.

    This is a security function to prevent path traversal attacks.
    Returns the resolved (absolute) path if valid.

    Raises:
        ValueError: If path is outside the allowed directory
    """
    resolved = path.resolve()
    safe_resolved = safe_dir.resolve()

    try:
        resolved.relative_to(safe_resolved)
    except ValueError:
        raise ValueError(f"Path '{path}' is outside the allowed directory") from None

    return resolved


class ClassifierService:
    """Service for text classifier operations."""

    def __init__(self, manager: ModelManager | None = None):
        """Initialize the Classifier service.

        Args:
            manager: Optional ModelManager instance. Uses global singleton if not provided.
        """
        self._manager = manager or model_manager

    async def load_model(
        self,
        model_id: str,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> "ClassifierModel":
        """Load or get cached classifier model.

        Args:
            model_id: Unique model identifier
            base_model: Base sentence transformer model

        Returns:
            Loaded ClassifierModel instance
        """
        from models import ClassifierModel

        cache_key = f"classifier:{model_id}"

        async def loader():
            logger.info(f"Loading classifier model: {model_id}")
            device = get_device()
            model = ClassifierModel(
                model_id=model_id,
                device=device,
                base_model=base_model,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(ModelType.CLASSIFIER, cache_key, loader)

    async def get_model(self, model_id: str) -> "ClassifierModel | None":
        """Get a cached classifier model without loading.

        Args:
            model_id: Model identifier

        Returns:
            ClassifierModel if cached, None otherwise
        """
        cache_key = f"classifier:{model_id}"
        return self._manager.get(ModelType.CLASSIFIER, cache_key)

    async def fit(
        self,
        model_id: str,
        texts: list[str],
        labels: list[str],
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_iterations: int = 20,
        batch_size: int = 16,
    ) -> dict:
        """Fit a text classifier.

        Args:
            model_id: Model identifier
            texts: Training texts
            labels: Labels for each text
            base_model: Base sentence transformer
            num_iterations: Training iterations
            batch_size: Training batch size

        Returns:
            Fit result with training info
        """
        model = await self.load_model(model_id, base_model)

        result = await model.fit(
            texts=texts,
            labels=labels,
            num_iterations=num_iterations,
            batch_size=batch_size,
        )

        return {
            "model": model_id,
            "base_model": result.base_model,
            "samples_fitted": result.samples_fitted,
            "num_classes": result.num_classes,
            "labels": result.labels,
            "training_time_ms": result.training_time_ms,
        }

    async def predict(self, model_id: str, texts: list[str]) -> dict:
        """Classify texts using a fitted model.

        Args:
            model_id: Model identifier (must be fitted)
            texts: Texts to classify

        Returns:
            Classification results

        Raises:
            KeyError: If model not found
            ValueError: If model not fitted
        """
        model = await self.get_model(model_id)
        if model is None:
            raise KeyError(f"Classifier '{model_id}' not found")

        if not model.is_fitted:
            raise ValueError("Model not fitted")

        results = await model.classify(texts)

        return {
            "model": model_id,
            "results": results,
        }

    async def save(self, model_id: str) -> dict:
        """Save a fitted classifier to disk.

        Args:
            model_id: Model identifier (must be fitted)

        Returns:
            Save result with path

        Raises:
            KeyError: If model not found
            ValueError: If model not fitted
        """
        model = await self.get_model(model_id)
        if model is None:
            raise KeyError(f"Classifier '{model_id}' not found in cache")

        if not model.is_fitted:
            raise ValueError("Model not fitted")

        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = _get_classifier_path(model_id)
        await model.save(str(save_path))

        return {
            "model": model_id,
            "path": str(save_path),
            "is_fitted": model.is_fitted,
            "labels": model.labels,
            "num_classes": len(model.labels),
        }

    async def auto_save(self, model_id: str, model: "ClassifierModel") -> dict:
        """Auto-save a classifier after fitting.

        Args:
            model_id: Model identifier
            model: The fitted model

        Returns:
            Dict with model_path (None if save failed)
        """
        if not model.is_fitted:
            return {"model_path": None}

        try:
            CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = _get_classifier_path(model_id)
            await model.save(str(save_path))
            logger.info(f"Auto-saved classifier model to {save_path}")
            return {"model_path": str(save_path)}
        except Exception as e:
            logger.warning(f"Failed to auto-save classifier model: {e}")
            return {"model_path": None}

    async def load_from_disk(self, model_id: str) -> dict:
        """Load a pre-trained classifier from disk.

        Args:
            model_id: Model identifier to load

        Returns:
            Load result with model info

        Raises:
            FileNotFoundError: If model not found on disk
        """
        from models import ClassifierModel

        model_path = _get_classifier_path(model_id)

        if not model_path.exists():
            available = (
                [f.name for f in CLASSIFIER_MODELS_DIR.glob("*") if f.is_dir()]
                if CLASSIFIER_MODELS_DIR.exists()
                else []
            )
            raise FileNotFoundError(
                f"Classifier '{model_id}' not found. "
                f"Available models: {available or 'none'}"
            )

        cache_key = f"classifier:{model_id}"

        # Unload existing if present
        existing = self._manager.get(ModelType.CLASSIFIER, cache_key)
        if existing:
            await existing.unload()
            self._manager.remove(ModelType.CLASSIFIER, cache_key)

        logger.info(f"Loading pre-trained classifier: {model_path}")
        device = get_device()

        model = ClassifierModel(
            model_id=str(model_path),
            device=device,
            load_from=str(model_path),
        )

        await model.load()
        self._manager.set(ModelType.CLASSIFIER, cache_key, model)

        return {
            "model": model_id,
            "path": str(model_path),
            "is_fitted": model.is_fitted,
            "labels": model.labels,
            "num_classes": len(model.labels),
        }

    def list_models(self) -> list[dict]:
        """List all saved classifier models.

        Returns:
            List of model info dicts
        """
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        models = []
        for path in CLASSIFIER_MODELS_DIR.glob("*"):
            if path.is_dir():
                labels = []
                labels_file = path / "labels.txt"
                if labels_file.exists():
                    labels = labels_file.read_text().strip().split("\n")

                models.append(
                    {
                        "name": path.name,
                        "path": str(path),
                        "labels": labels,
                        "num_classes": len(labels),
                    }
                )

        return models

    def delete_model(self, model_name: str) -> dict:
        """Delete a saved classifier model.

        Args:
            model_name: Name of the model to delete

        Returns:
            Deletion result

        Raises:
            FileNotFoundError: If model not found
            ValueError: If path validation fails
        """
        import shutil

        model_path = _get_classifier_path(model_name)

        resolved_path = _validate_path_within_directory(
            model_path, CLASSIFIER_MODELS_DIR
        )

        if not resolved_path.exists():
            raise FileNotFoundError(f"Classifier model '{model_name}' not found")

        shutil.rmtree(resolved_path)

        return {
            "model": model_name,
            "path": str(resolved_path),
        }


# Global service instance
classifier_service = ClassifierService()
