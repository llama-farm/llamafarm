"""Vision service for model loading and business logic.

Handles CLIP, few-shot classifiers, object detection, open-vocab detection,
and background removal models.
"""

import logging
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from services.model_manager import ModelManager, ModelType, model_manager
from utils.device import get_optimal_device

if TYPE_CHECKING:
    from models import (
        BackgroundRemovalModel,
        CLIPVisionModel,
        FewShotImageClassifier,
        ObjectDetectionModel,
        OpenVocabDetectionModel,
    )

logger = logging.getLogger(__name__)

# Data directory for saving models
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
FEW_SHOT_MODELS_DIR = _LF_DATA_DIR / "models" / "few_shot"

# Global device cache
_current_device: str | None = None


def get_device() -> str:
    """Get the optimal device for the current platform."""
    global _current_device
    if _current_device is None:
        _current_device = get_optimal_device()
        logger.info(f"Using device: {_current_device}")
    return _current_device


def _sanitize_model_name(name: str) -> str:
    """Sanitize a model name for use in filesystem paths.

    Prevents path traversal attacks by removing dangerous characters.
    """
    # Remove any path components
    name = os.path.basename(name)
    # Only allow alphanumeric, dash, underscore, dot
    name = re.sub(r"[^a-zA-Z0-9_.-]", "_", name)
    # Prevent hidden files
    name = name.lstrip(".")
    return name or "unnamed"


class VisionService:
    """Service for vision model operations."""

    def __init__(self, manager: ModelManager | None = None):
        """Initialize the vision service.

        Args:
            manager: Optional ModelManager instance. Uses global singleton if not provided.
        """
        self._manager = manager or model_manager

    # =========================================================================
    # CLIP Zero-Shot Classification
    # =========================================================================

    async def load_vision_model(
        self, model_name: str = "openai/clip-vit-base-patch32"
    ) -> "CLIPVisionModel":
        """Load a CLIP vision model.

        Args:
            model_name: HuggingFace model name

        Returns:
            Loaded CLIPVisionModel instance
        """
        from models import CLIPVisionModel

        cache_key = self._manager.make_vision_cache_key(model_name)

        async def loader():
            logger.info(f"Loading CLIP vision model: {model_name}")
            device = get_device()
            model = CLIPVisionModel(
                model_id=cache_key,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(ModelType.VISION, cache_key, loader)

    async def classify_zero_shot(
        self, image: str, labels: list[str], model_name: str
    ) -> dict:
        """Classify an image using zero-shot CLIP.

        Args:
            image: Base64-encoded image or file path
            labels: Labels to classify against
            model_name: CLIP model to use

        Returns:
            Classification result with label, score, and all_scores
        """
        model = await self.load_vision_model(model_name)
        return await model.classify_zero_shot(image, labels)

    async def classify_zero_shot_batch(
        self, images: list[str], labels: list[str], model_name: str
    ) -> list[dict]:
        """Classify multiple images using zero-shot CLIP.

        Args:
            images: List of base64-encoded images or file paths
            labels: Labels to classify against
            model_name: CLIP model to use

        Returns:
            List of classification results
        """
        model = await self.load_vision_model(model_name)
        return await model.classify_zero_shot_batch(images, labels)

    # =========================================================================
    # Few-Shot Classification
    # =========================================================================

    def _get_few_shot_path(self, classifier_id: str) -> Path:
        """Get the path for a few-shot classifier file."""
        safe_name = _sanitize_model_name(classifier_id)
        return FEW_SHOT_MODELS_DIR / f"{safe_name}.fsc"

    async def load_few_shot_classifier(
        self,
        classifier_id: str,
        model_name: str = "openai/clip-vit-base-patch32",
        create_if_missing: bool = True,
    ) -> "FewShotImageClassifier | None":
        """Load or create a few-shot classifier.

        Args:
            classifier_id: Unique identifier for this classifier
            model_name: CLIP model to use for embeddings
            create_if_missing: If True, create a new classifier if not found

        Returns:
            FewShotImageClassifier instance or None if not found
        """
        from models import FewShotImageClassifier

        cache_key = self._manager.make_few_shot_cache_key(classifier_id, model_name)

        # Check if already cached
        if self._manager.contains(ModelType.FEW_SHOT, cache_key):
            return self._manager.get(ModelType.FEW_SHOT, cache_key)

        if not create_if_missing:
            return None

        async def loader():
            logger.info(
                f"Creating few-shot classifier: {classifier_id} (CLIP: {model_name})"
            )
            device = get_device()
            model = FewShotImageClassifier(
                model_id=classifier_id,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(ModelType.FEW_SHOT, cache_key, loader)

    async def auto_save_few_shot_classifier(
        self,
        classifier: "FewShotImageClassifier",
        classifier_id: str,
    ) -> dict[str, str | None]:
        """Auto-save few-shot classifier after training.

        Returns:
            Dict with saved file path
        """
        try:
            FEW_SHOT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
            save_path = self._get_few_shot_path(classifier_id)
            state_bytes = classifier.save_state()

            # Write atomically using a temp file
            temp_path = save_path.with_suffix(".tmp")
            temp_path.write_bytes(state_bytes)
            temp_path.rename(save_path)

            logger.info(f"Auto-saved few-shot classifier to {save_path}")
            return {"model_path": str(save_path)}

        except Exception as e:
            logger.warning(f"Auto-save failed (model still in memory): {e}")
            return {"model_path": None}

    async def train_few_shot(
        self,
        classifier_id: str,
        images: list[str],
        labels: list[str],
        model_name: str,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> dict:
        """Train a few-shot classifier.

        Returns:
            Training result with success status and metrics
        """
        classifier = await self.load_few_shot_classifier(classifier_id, model_name)
        result = await classifier.fit(
            images=images,
            labels=labels,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        # Auto-save after training
        saved = await self.auto_save_few_shot_classifier(classifier, classifier_id)
        return {**result, "saved_path": saved.get("model_path")}

    async def refine_few_shot(
        self,
        classifier_id: str,
        images: list[str],
        labels: list[str],
        model_name: str,
        epochs: int = 50,
        learning_rate: float = 0.0005,
    ) -> dict:
        """Refine a few-shot classifier with additional data."""
        classifier = await self.load_few_shot_classifier(
            classifier_id, model_name, create_if_missing=False
        )
        if classifier is None:
            raise ValueError(f"Classifier '{classifier_id}' not found")
        if not classifier.is_trained:
            raise ValueError(f"Classifier '{classifier_id}' is not trained")

        result = await classifier.refine(
            images=images,
            labels=labels,
            epochs=epochs,
            learning_rate=learning_rate,
        )
        # Auto-save after refinement
        saved = await self.auto_save_few_shot_classifier(classifier, classifier_id)
        return {**result, "saved_path": saved.get("model_path")}

    async def predict_few_shot(
        self, classifier_id: str, image: str, model_name: str
    ) -> dict:
        """Classify an image using a trained few-shot classifier."""
        classifier = await self.load_few_shot_classifier(
            classifier_id, model_name, create_if_missing=False
        )
        if classifier is None:
            raise ValueError(f"Classifier '{classifier_id}' not found")
        if not classifier.is_trained:
            raise ValueError(f"Classifier '{classifier_id}' is not trained")
        return await classifier.predict(image)

    async def predict_few_shot_batch(
        self, classifier_id: str, images: list[str], model_name: str
    ) -> list[dict]:
        """Classify multiple images using a trained few-shot classifier."""
        classifier = await self.load_few_shot_classifier(
            classifier_id, model_name, create_if_missing=False
        )
        if classifier is None:
            raise ValueError(f"Classifier '{classifier_id}' not found")
        if not classifier.is_trained:
            raise ValueError(f"Classifier '{classifier_id}' is not trained")
        return await classifier.predict_batch(images)

    def list_few_shot_classifiers(self) -> list[dict]:
        """List all saved few-shot classifiers."""
        import json

        FEW_SHOT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        classifiers = []

        for path in FEW_SHOT_MODELS_DIR.glob("*.fsc"):
            info = {
                "classifier_id": path.stem,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
            # Try to extract metadata
            try:
                state_bytes = path.read_bytes()
                metadata_len = int.from_bytes(state_bytes[:4], byteorder="big")
                metadata_bytes = state_bytes[4 : 4 + metadata_len]
                metadata = json.loads(metadata_bytes.decode("utf-8"))
                info["classes"] = metadata.get("classes", [])
                info["model"] = metadata.get("model", "unknown")
            except Exception:
                info["classes"] = []
                info["model"] = "unknown"
            classifiers.append(info)

        return classifiers

    async def load_saved_few_shot_classifier(
        self, classifier_id: str, model_name: str
    ) -> "FewShotImageClassifier":
        """Load a previously saved few-shot classifier."""
        from models import FewShotImageClassifier

        model_path = self._get_few_shot_path(classifier_id)
        if not model_path.exists():
            available = [f.stem for f in FEW_SHOT_MODELS_DIR.glob("*.fsc")]
            raise FileNotFoundError(
                f"Classifier '{classifier_id}' not found. Available: {available}"
            )

        cache_key = self._manager.make_few_shot_cache_key(classifier_id, model_name)

        # Remove existing if present
        existing = self._manager.remove(ModelType.FEW_SHOT, cache_key)
        if existing:
            await existing.unload()

        async with self._manager.load_lock:
            logger.info(f"Loading saved few-shot classifier: {model_path}")
            device = get_device()

            classifier = FewShotImageClassifier(
                model_id=classifier_id,
                device=device,
                hf_model_name=model_name,
            )
            await classifier.load()

            # Load saved state
            state_bytes = model_path.read_bytes()
            classifier.load_state(state_bytes)

            self._manager.set(ModelType.FEW_SHOT, cache_key, classifier)

        return classifier

    async def unload_few_shot_classifier(
        self, classifier_id: str, model_name: str
    ) -> bool:
        """Unload a few-shot classifier from memory."""
        cache_key = self._manager.make_few_shot_cache_key(classifier_id, model_name)
        classifier = self._manager.remove(ModelType.FEW_SHOT, cache_key)
        if classifier:
            await classifier.unload()
            return True
        return False

    def get_few_shot_classifier_info(
        self, classifier_id: str, model_name: str
    ) -> dict | None:
        """Get info about a loaded few-shot classifier."""
        cache_key = self._manager.make_few_shot_cache_key(classifier_id, model_name)
        classifier = self._manager.get(ModelType.FEW_SHOT, cache_key)
        if classifier:
            return classifier.get_model_info()
        return None

    # =========================================================================
    # Object Detection (YOLOS)
    # =========================================================================

    async def load_object_detection_model(
        self, model_name: str = "hustvl/yolos-tiny"
    ) -> "ObjectDetectionModel":
        """Load an object detection model."""
        from models import ObjectDetectionModel

        cache_key = self._manager.make_object_detection_cache_key(model_name)

        async def loader():
            logger.info(f"Loading object detection model: {model_name}")
            device = get_device()
            model = ObjectDetectionModel(
                model_id=cache_key,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(
            ModelType.OBJECT_DETECTION, cache_key, loader
        )

    async def detect_objects(
        self,
        image: str,
        threshold: float,
        labels: list[str] | None,
        model_name: str,
    ) -> dict:
        """Detect objects in an image."""
        model = await self.load_object_detection_model(model_name)
        return await model.detect(image, threshold=threshold, labels=labels)

    async def detect_objects_batch(
        self,
        images: list[str],
        threshold: float,
        labels: list[str] | None,
        model_name: str,
    ) -> list[dict]:
        """Detect objects in multiple images."""
        model = await self.load_object_detection_model(model_name)
        return await model.detect_batch(images, threshold=threshold, labels=labels)

    # =========================================================================
    # Open-Vocabulary Detection (OWL-ViT)
    # =========================================================================

    async def load_open_vocab_model(
        self, model_name: str = "google/owlvit-base-patch32"
    ) -> "OpenVocabDetectionModel":
        """Load an open-vocabulary detection model."""
        from models import OpenVocabDetectionModel

        cache_key = self._manager.make_open_vocab_cache_key(model_name)

        async def loader():
            logger.info(f"Loading OWL-ViT model: {model_name}")
            device = get_device()
            model = OpenVocabDetectionModel(
                model_id=cache_key,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(ModelType.OPEN_VOCAB, cache_key, loader)

    async def detect_by_text(
        self,
        image: str,
        queries: list[str],
        threshold: float,
        top_k: int | None,
        model_name: str,
    ) -> dict:
        """Detect objects using text queries."""
        model = await self.load_open_vocab_model(model_name)
        return await model.detect_by_text(
            image=image, queries=queries, threshold=threshold, top_k=top_k
        )

    async def detect_by_text_batch(
        self,
        images: list[str],
        queries: list[str],
        threshold: float,
        top_k: int | None,
        model_name: str,
    ) -> list[dict]:
        """Detect objects in multiple images using text queries."""
        model = await self.load_open_vocab_model(model_name)
        return await model.detect_batch_by_text(
            images=images, queries=queries, threshold=threshold, top_k=top_k
        )

    async def detect_by_image(
        self,
        image: str,
        query_images: list[str],
        threshold: float,
        top_k: int | None,
        model_name: str,
    ) -> dict:
        """Detect objects using reference images."""
        model = await self.load_open_vocab_model(model_name)
        return await model.detect_by_image(
            image=image,
            query_images=query_images,
            threshold=threshold,
            top_k=top_k,
        )

    # =========================================================================
    # Background Removal (RMBG)
    # =========================================================================

    async def load_background_removal_model(
        self, model_name: str = "briaai/RMBG-1.4"
    ) -> "BackgroundRemovalModel":
        """Load a background removal model."""
        from models import BackgroundRemovalModel

        cache_key = self._manager.make_background_removal_cache_key(model_name)

        async def loader():
            logger.info(f"Loading background removal model: {model_name}")
            device = get_device()
            model = BackgroundRemovalModel(
                model_id=cache_key,
                device=device,
                hf_model_name=model_name,
            )
            await model.load()
            return model

        return await self._manager.get_or_load(
            ModelType.BACKGROUND_REMOVAL, cache_key, loader
        )

    async def remove_background(
        self, image: str, return_mask: bool, model_name: str
    ) -> dict:
        """Remove background from an image."""
        model = await self.load_background_removal_model(model_name)
        return await model.remove_background(image, return_mask=return_mask)

    async def remove_background_batch(
        self, images: list[str], return_mask: bool, model_name: str
    ) -> list[dict]:
        """Remove background from multiple images."""
        model = await self.load_background_removal_model(model_name)
        return await model.remove_background_batch(images, return_mask=return_mask)


# Global singleton instance
vision_service = VisionService()
