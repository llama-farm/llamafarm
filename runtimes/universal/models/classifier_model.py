"""
Text classifier model using SetFit for few-shot learning.

SetFit (Sentence Transformer Fine-tuning) enables training text classifiers
with just 8-16 examples per class. It works by:
1. Fine-tuning a sentence-transformer with contrastive learning
2. Training a small classification head on top

Designed for:
- Intent classification
- Sentiment analysis
- Topic categorization
- Spam detection
- Any text classification task with limited labeled data

Security Notes:
- Model loading is restricted to a designated safe directory (CLASSIFIER_MODELS_DIR)
- Path traversal attacks are prevented by validating paths are within the safe directory
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import BaseModel

logger = logging.getLogger(__name__)

# Safe directory for classifier models - uses standard LlamaFarm data directory
# ~/.llamafarm/models/classifier/ (or LF_DATA_DIR/models/classifier/)
# Only files within this directory can be loaded - prevents path traversal attacks
_LF_DATA_DIR = Path(os.environ.get("LF_DATA_DIR", Path.home() / ".llamafarm"))
CLASSIFIER_MODELS_DIR = (_LF_DATA_DIR / "models" / "classifier").resolve()


def _validate_model_path(model_path: Path) -> Path:
    """Validate that model path is within the safe directory.

    Security: This function prevents path traversal attacks by ensuring
    the model path resolves to a location within CLASSIFIER_MODELS_DIR.

    Raises:
        ValueError: If path is outside the safe directory or uses path traversal
    """
    # Check for path traversal patterns
    path_str = str(model_path)
    if ".." in path_str:
        raise ValueError(
            f"Security error: Model path '{model_path}' contains '..' "
            "Path traversal is not allowed."
        )

    # Resolve to absolute path
    resolved_path = model_path.resolve()

    # Verify the resolved path is within the safe directory
    try:
        resolved_path.relative_to(CLASSIFIER_MODELS_DIR)
    except ValueError:
        raise ValueError(
            f"Security error: Model path '{model_path}' is outside the allowed "
            f"directory '{CLASSIFIER_MODELS_DIR}'. Path traversal is not allowed."
        ) from None

    return resolved_path


@dataclass
class ClassifierFitResult:
    """Result from fitting a classifier."""

    samples_fitted: int
    num_classes: int
    labels: list[str]
    training_time_ms: float
    base_model: str


@dataclass
class ClassificationResult:
    """Result from classifying a single text."""

    text: str
    label: str
    score: float
    all_scores: dict[str, float]


class ClassifierModel(BaseModel):
    """SetFit-based text classifier for few-shot learning.

    Supports training classifiers with just 8-16 examples per class.
    Uses sentence-transformers for embeddings and trains a small
    classification head on top.

    Usage patterns:
    1. Fit: Train on labeled examples, then classify new texts
    2. Save/Load: Persist trained models for production use
    """

    def __init__(
        self,
        model_id: str,
        device: str,
        base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize classifier model.

        Args:
            model_id: Model identifier (for caching) or path to pre-trained model
            device: Target device (cuda/mps/cpu)
            base_model: Base sentence-transformer model for embeddings
        """
        super().__init__(model_id, device)
        self.base_model = base_model
        self.model_type = "classifier_setfit"
        self.supports_streaming = False

        # SetFit model
        self._classifier = None
        self._is_fitted = False
        self._labels: list[str] = []

    @property
    def is_fitted(self) -> bool:
        """Check if model is fitted."""
        return self._is_fitted

    @property
    def labels(self) -> list[str]:
        """Get the class labels."""
        return self._labels

    async def load(self) -> None:
        """Load or initialize the classifier model."""
        logger.info(f"Loading classifier model: {self.base_model}")

        # Check if model_id is a path to a pre-trained model
        model_path = Path(self.model_id)
        if model_path.exists() and model_path.is_dir():
            # Validate path is within safe directory before loading
            try:
                validated_path = _validate_model_path(model_path)
                await self._load_pretrained(validated_path)
            except ValueError as e:
                logger.error(f"Security validation failed: {e}")
                raise
        else:
            await self._initialize_model()

        logger.info(f"Classifier model initialized: {self.base_model}")

    async def _load_pretrained(self, model_path: Path) -> None:
        """Load a pre-trained SetFit model from disk.

        Security Note: This method should only be called with validated paths
        from _validate_model_path() to prevent path traversal attacks.
        """
        logger.info(f"Loading pretrained classifier from validated path: {model_path}")

        try:
            from setfit import SetFitModel
        except ImportError as e:
            raise ImportError(
                "SetFit not installed. Install with: uv pip install setfit"
            ) from e

        self._classifier = SetFitModel.from_pretrained(str(model_path))

        # Move to device
        if hasattr(self._classifier, "model_body"):
            self._classifier.model_body = self._classifier.model_body.to(self.device)

        # Load labels - try multiple sources in order of reliability
        # 1. First try labels.txt file (most reliable, saved by our save() method)
        labels_path = model_path / "labels.txt"
        if labels_path.exists():
            with open(labels_path) as f:
                self._labels = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded labels from labels.txt: {self._labels}")
        # 2. Fall back to SetFit model attributes
        elif hasattr(self._classifier, "labels") and self._classifier.labels:
            self._labels = list(self._classifier.labels)
        elif hasattr(self._classifier, "model_head") and hasattr(
            self._classifier.model_head, "classes_"
        ):
            self._labels = list(self._classifier.model_head.classes_)
        else:
            logger.warning(
                "Could not load labels from model. Classification will return indices."
            )

        self._is_fitted = True
        logger.info(f"Loaded classifier with labels: {self._labels}")

    async def _initialize_model(self) -> None:
        """Initialize a fresh SetFit model for training."""
        try:
            from setfit import SetFitModel
        except ImportError as e:
            raise ImportError(
                "SetFit not installed. Install with: uv pip install setfit"
            ) from e

        logger.info(f"Initializing SetFit model with base: {self.base_model}")

        self._classifier = SetFitModel.from_pretrained(self.base_model)

        # Move to device
        if hasattr(self._classifier, "model_body"):
            self._classifier.model_body = self._classifier.model_body.to(self.device)

    async def fit(
        self,
        texts: list[str],
        labels: list[str],
        num_iterations: int = 20,
        batch_size: int = 16,
    ) -> ClassifierFitResult:
        """Fit the classifier on labeled training data.

        Args:
            texts: List of training texts
            labels: List of corresponding labels
            num_iterations: Number of training iterations (default: 20)
            batch_size: Training batch size

        Returns:
            ClassifierFitResult with training statistics
        """
        try:
            from datasets import Dataset
            from setfit import SetFitModel, SetFitTrainer
        except ImportError as e:
            raise ImportError(
                "SetFit not installed. Install with: uv pip install setfit"
            ) from e

        start_time = time.time()

        # Store unique labels
        self._labels = sorted(set(labels))
        logger.info(
            f"Training classifier with {len(self._labels)} classes: {self._labels}"
        )

        # Create dataset
        train_dataset = Dataset.from_dict({"text": texts, "label": labels})

        # Initialize model if not already done
        if self._classifier is None:
            self._classifier = SetFitModel.from_pretrained(self.base_model)
            if hasattr(self._classifier, "model_body"):
                self._classifier.model_body = self._classifier.model_body.to(
                    self.device
                )

        # Create trainer with parameters directly (SetFit 1.1+ API)
        trainer = SetFitTrainer(
            model=self._classifier,
            train_dataset=train_dataset,
            num_iterations=num_iterations,
            batch_size=batch_size,
        )

        trainer.train()
        self._is_fitted = True

        training_time = (time.time() - start_time) * 1000

        return ClassifierFitResult(
            samples_fitted=len(texts),
            num_classes=len(self._labels),
            labels=self._labels,
            training_time_ms=training_time,
            base_model=self.base_model,
        )

    async def classify(self, texts: list[str]) -> list[ClassificationResult]:
        """Classify input texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of ClassificationResult objects
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Model not fitted. Call fit() first or load a pre-trained model."
            )

        # Get predictions
        predictions = self._classifier.predict(texts)
        probabilities = self._classifier.predict_proba(texts)

        results = []
        for text, pred, probs in zip(texts, predictions, probabilities, strict=True):
            # Convert prediction to string label
            if isinstance(pred, int):
                label = self._labels[pred] if pred < len(self._labels) else str(pred)
            else:
                label = str(pred)

            # Build all_scores dict
            all_scores = {}
            if hasattr(probs, "__iter__"):
                for j, prob in enumerate(probs):
                    class_label = self._labels[j] if j < len(self._labels) else str(j)
                    all_scores[class_label] = float(prob)

            # Get the score for the predicted class
            score = all_scores.get(label, 1.0)

            results.append(
                ClassificationResult(
                    text=text,
                    label=label,
                    score=score,
                    all_scores=all_scores,
                )
            )

        return results

    async def save(self, path: str) -> None:
        """Save the fitted model to disk.

        Security: Path is validated to be within CLASSIFIER_MODELS_DIR to prevent
        path traversal attacks.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Nothing to save.")

        model_path = Path(path)

        # Validate path is within the safe directory
        # First ensure CLASSIFIER_MODELS_DIR exists
        CLASSIFIER_MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # If path is relative or doesn't start with safe dir, make it relative to safe dir
        if not model_path.is_absolute():
            model_path = CLASSIFIER_MODELS_DIR / model_path
        validated_path = _validate_model_path(model_path)

        validated_path.mkdir(parents=True, exist_ok=True)

        # Save the SetFit model
        self._classifier.save_pretrained(str(validated_path))

        # Save labels separately for loading (SetFit doesn't always preserve labels)
        labels_path = validated_path / "labels.txt"
        with open(labels_path, "w") as f:
            f.write("\n".join(self._labels))

        logger.info(f"Classifier model saved to {validated_path}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        self._classifier = None
        self._is_fitted = False
        self._labels = []
        await super().unload()

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the model."""
        info = super().get_model_info()
        info.update(
            {
                "base_model": self.base_model,
                "is_fitted": self._is_fitted,
                "num_classes": len(self._labels),
                "labels": self._labels,
            }
        )
        return info
