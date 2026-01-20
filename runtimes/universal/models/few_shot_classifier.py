"""Few-Shot Image Classifier using CLIP embeddings.

Train a classifier on just 5-50 images per class using frozen CLIP features
with a trainable linear probe.

Usage:
    classifier = FewShotImageClassifier(model_id="my-classifier")
    await classifier.load()

    # Train with a few examples
    await classifier.fit(
        images=["cat1.jpg", "cat2.jpg", "dog1.jpg", "dog2.jpg"],
        labels=["cat", "cat", "dog", "dog"]
    )

    # Classify new images
    result = await classifier.predict("unknown.jpg")
    # {"label": "cat", "score": 0.92, "all_scores": {"cat": 0.92, "dog": 0.08}}
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from models.base import BaseModel
from utils.image_utils import load_image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class LinearProbe(nn.Module):
    """Simple linear classifier head."""

    def __init__(self, input_dim: int, num_classes: int, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class FewShotImageClassifier(BaseModel):
    """Few-shot image classifier using CLIP embeddings with linear probe.

    This classifier extracts CLIP image embeddings (frozen) and trains
    a simple linear classifier on top. Works well with 5-50 images per class.

    Attributes:
        model_id: Unique identifier for this classifier
        device: Device to run on (cpu, cuda, mps)
        hf_model_name: CLIP model to use for embeddings
        classes: List of class labels (set after training)
        is_trained: Whether the classifier has been trained
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_CLIP_MODEL,
        token: str | None = None,
    ):
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "few_shot_classifier"
        self.supports_streaming = False
        self._is_loaded = False
        self._is_trained = False
        self.classes: list[str] = []
        self.classifier: LinearProbe | None = None
        self._embed_dim: int = 512  # Will be updated on load

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded and self.model is not None

    @property
    def is_trained(self) -> bool:
        return self._is_trained and self.classifier is not None

    async def load(self) -> None:
        """Load the CLIP model for feature extraction."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading CLIP model for few-shot classifier: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"Few-shot classifier loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading."""
        from transformers import CLIPModel, CLIPProcessor

        self.processor = CLIPProcessor.from_pretrained(
            self.hf_model_name,
            token=self.token,
        )
        self.model = CLIPModel.from_pretrained(
            self.hf_model_name,
            token=self.token,
            torch_dtype=self.get_dtype(),
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Freeze CLIP parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Get embedding dimension
        self._embed_dim = self.model.config.projection_dim

    async def unload(self) -> None:
        """Unload the model and free resources."""
        await super().unload()
        self._is_loaded = False
        self._is_trained = False
        self.classifier = None
        self.classes = []

    def _extract_features(self, images: list) -> torch.Tensor:
        """Extract CLIP embeddings for a list of images."""

        # Load all images
        pil_images = [load_image(img) for img in images]

        # Process in batches
        batch_size = 32
        all_features = []

        for i in range(0, len(pil_images), batch_size):
            batch = pil_images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: self.to_device(v) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                # L2 normalize
                features = outputs / outputs.norm(dim=-1, keepdim=True)
                all_features.append(features)

        return torch.cat(all_features, dim=0)

    async def fit(
        self,
        images: list[str | bytes],
        labels: list[str],
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Train the classifier on labeled images.

        Args:
            images: List of images (file paths, base64, or bytes)
            labels: List of class labels (same length as images)
            epochs: Number of training epochs (default: 100)
            learning_rate: Learning rate (default: 0.001)
            batch_size: Batch size for training (default: 32)

        Returns:
            Training statistics including final accuracy
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Input validation
        if not images:
            raise ValueError("Images list cannot be empty")

        if not labels:
            raise ValueError("Labels list cannot be empty")

        if len(images) != len(labels):
            raise ValueError(f"Number of images ({len(images)}) must match labels ({len(labels)})")

        if len(images) < 2:
            raise ValueError("Need at least 2 images to train")

        # Validate all labels are strings
        for i, label in enumerate(labels):
            if not isinstance(label, str):
                raise ValueError(f"Label at index {i} must be a string, got {type(label).__name__}")
            if not label.strip():
                raise ValueError(f"Label at index {i} cannot be empty or whitespace")

        # Check for minimum class count
        unique_classes = set(labels)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Need at least 2 distinct classes to train. "
                f"Found {len(unique_classes)}: {sorted(unique_classes)}"
            )

        # Warn about class imbalance (but don't block)
        from collections import Counter
        class_counts = Counter(labels)
        min_count = min(class_counts.values())
        max_count = max(class_counts.values())
        if max_count > min_count * 10:
            logger.warning(
                f"Significant class imbalance detected: "
                f"min={min_count}, max={max_count}. "
                f"Consider balancing your training data."
            )

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._fit_sync,
            images,
            labels,
            epochs,
            learning_rate,
            batch_size,
        )

    def _fit_sync(
        self,
        images: list,
        labels: list[str],
        epochs: int,
        learning_rate: float,
        batch_size: int,
    ) -> dict[str, Any]:
        """Synchronous training."""
        import time
        start_time = time.time()

        # Get unique classes
        self.classes = sorted(set(labels))
        num_classes = len(self.classes)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        logger.info(f"Training few-shot classifier with {len(images)} images, {num_classes} classes")

        # Extract features
        logger.info("Extracting CLIP features...")
        features = self._extract_features(images)

        # Convert features to float32 for training stability (float16 causes NaN on MPS)
        features = features.float()

        # Convert labels to indices
        label_indices = torch.tensor([class_to_idx[lbl] for lbl in labels], device=self.device)

        # Create classifier as float32 for training stability
        self.classifier = LinearProbe(self._embed_dim, num_classes, dtype=torch.float32).to(self.device)

        # Training
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.classifier.train()
        losses = []

        for _epoch in range(epochs):
            # Simple full-batch training for small datasets
            logits = self.classifier(features)
            loss = criterion(logits, label_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Evaluate final accuracy
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(features)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == label_indices).float().mean().item()

        self._is_trained = True
        training_time = time.time() - start_time

        # Class distribution
        class_counts = {c: labels.count(c) for c in self.classes}

        logger.info(f"Training complete: accuracy={accuracy:.2%}, time={training_time:.2f}s")

        return {
            "success": True,
            "num_samples": len(images),
            "num_classes": num_classes,
            "classes": self.classes,
            "class_counts": class_counts,
            "final_loss": losses[-1],
            "final_accuracy": accuracy,
            "training_time_ms": training_time * 1000,
            "epochs": epochs,
        }

    async def predict(
        self,
        image: str | bytes,
    ) -> dict[str, Any]:
        """Classify a single image.

        Args:
            image: Image to classify (file path, base64, or bytes)

        Returns:
            Dictionary with label, score, and all_scores
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._predict_sync, image)

    def _predict_sync(self, image: str | bytes) -> dict[str, Any]:
        """Synchronous prediction."""
        # Extract features and convert to float32 to match classifier dtype
        features = self._extract_features([image]).float()

        # Classify
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1).squeeze()

        probs_np = probs.cpu().numpy()
        best_idx = int(probs_np.argmax())

        all_scores = {c: float(probs_np[i]) for i, c in enumerate(self.classes)}

        return {
            "label": self.classes[best_idx],
            "score": float(probs_np[best_idx]),
            "all_scores": all_scores,
        }

    async def predict_batch(
        self,
        images: list[str | bytes],
    ) -> list[dict[str, Any]]:
        """Classify multiple images.

        Args:
            images: List of images to classify

        Returns:
            List of prediction results
        """
        if not self.is_trained:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._predict_batch_sync, images)

    def _predict_batch_sync(self, images: list) -> list[dict[str, Any]]:
        """Synchronous batch prediction."""
        # Extract features for all images and convert to float32 to match classifier dtype
        features = self._extract_features(images).float()

        # Classify all at once
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(features)
            probs = torch.softmax(logits, dim=1)

        probs_np = probs.cpu().numpy()

        results = []
        for i in range(len(images)):
            p = probs_np[i]
            best_idx = int(p.argmax())
            results.append({
                "label": self.classes[best_idx],
                "score": float(p[best_idx]),
                "all_scores": {c: float(p[j]) for j, c in enumerate(self.classes)},
            })

        return results

    async def refine(
        self,
        images: list[str | bytes],
        labels: list[str],
        epochs: int = 50,
        learning_rate: float = 0.0005,
    ) -> dict[str, Any]:
        """Refine the classifier with additional examples.

        Use this to add more training data or correct misclassifications.
        Can add new classes or more examples of existing classes.

        Args:
            images: Additional images
            labels: Labels for the new images
            epochs: Training epochs for refinement
            learning_rate: Learning rate (lower than initial training)

        Returns:
            Updated training statistics
        """
        if not self.is_trained:
            # If not trained yet, just do regular training
            return await self.fit(images, labels, epochs=epochs * 2, learning_rate=learning_rate * 2)

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._refine_sync,
            images,
            labels,
            epochs,
            learning_rate,
        )

    def _refine_sync(
        self,
        images: list,
        labels: list[str],
        epochs: int,
        learning_rate: float,
    ) -> dict[str, Any]:
        """Synchronous refinement."""
        import time
        start_time = time.time()

        # Check for new classes
        new_classes = [lbl for lbl in set(labels) if lbl not in self.classes]
        if new_classes:
            # Need to expand classifier head
            old_classes = self.classes
            self.classes = sorted(set(self.classes + new_classes))
            num_classes = len(self.classes)

            logger.info(f"Adding {len(new_classes)} new classes: {new_classes}")

            # Create new classifier with more outputs (float32 for training stability)
            old_classifier = self.classifier
            self.classifier = LinearProbe(self._embed_dim, num_classes, dtype=torch.float32).to(self.device)

            # Copy old weights for existing classes
            with torch.no_grad():
                for i, c in enumerate(old_classes):
                    new_idx = self.classes.index(c)
                    self.classifier.linear.weight[new_idx] = old_classifier.linear.weight[i].float()
                    self.classifier.linear.bias[new_idx] = old_classifier.linear.bias[i].float()

        class_to_idx = {c: i for i, c in enumerate(self.classes)}

        # Extract features for new images
        logger.info(f"Refining with {len(images)} additional images...")
        features = self._extract_features(images)
        # Convert features to float32 for training stability
        features = features.float()
        label_indices = torch.tensor([class_to_idx[lbl] for lbl in labels], device=self.device)

        # Fine-tune
        optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.classifier.train()
        losses = []

        for _epoch in range(epochs):
            logits = self.classifier(features)
            loss = criterion(logits, label_indices)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        # Evaluate
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(features)
            predictions = logits.argmax(dim=1)
            accuracy = (predictions == label_indices).float().mean().item()

        training_time = time.time() - start_time

        return {
            "success": True,
            "refined_samples": len(images),
            "num_classes": len(self.classes),
            "classes": self.classes,
            "new_classes_added": new_classes,
            "final_loss": losses[-1],
            "accuracy_on_new_data": accuracy,
            "training_time_ms": training_time * 1000,
        }

    def save_state(self) -> bytes:
        """Save the trained classifier state.

        Uses a safe serialization format with JSON for metadata and
        PyTorch's safetensors-compatible format for weights.
        """
        if not self.is_trained:
            raise RuntimeError("No trained classifier to save")

        import io
        import json

        # Separate metadata (JSON-safe) from model weights
        metadata = {
            "version": 1,
            "classes": self.classes,
            "embed_dim": self._embed_dim,
            "hf_model_name": self.hf_model_name,
        }

        # Save model weights using PyTorch's safe format (protocol 2 for weights_only compatibility)
        weights_buffer = io.BytesIO()
        torch.save(self.classifier.state_dict(), weights_buffer, pickle_protocol=2)
        weights_bytes = weights_buffer.getvalue()

        # Pack format: 4-byte metadata length + JSON metadata + weights
        metadata_bytes = json.dumps(metadata).encode("utf-8")
        metadata_len = len(metadata_bytes).to_bytes(4, byteorder="big")

        return metadata_len + metadata_bytes + weights_bytes

    def load_state(self, state_bytes: bytes) -> None:
        """Load a previously saved classifier state.

        Safely deserializes metadata from JSON and weights from PyTorch format.
        """
        import io
        import json

        # Unpack format: 4-byte metadata length + JSON metadata + weights
        metadata_len = int.from_bytes(state_bytes[:4], byteorder="big")
        metadata_bytes = state_bytes[4:4 + metadata_len]
        weights_bytes = state_bytes[4 + metadata_len:]

        # Parse JSON metadata (safe from code execution)
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        # Validate metadata
        if metadata.get("version", 0) < 1:
            raise ValueError("Invalid or unsupported state format version")

        self.classes = metadata["classes"]
        self._embed_dim = metadata["embed_dim"]
        self.hf_model_name = metadata["hf_model_name"]

        # Load model weights using PyTorch (weights_only=True for safety)
        weights_buffer = io.BytesIO(weights_bytes)
        state_dict = torch.load(weights_buffer, map_location=self.device, weights_only=True)

        # Recreate classifier as float32 for consistency
        self.classifier = LinearProbe(self._embed_dim, len(self.classes), dtype=torch.float32).to(self.device)
        self.classifier.load_state_dict(state_dict)
        self.classifier.eval()

        self._is_trained = True

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the classifier."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "is_trained": self.is_trained,
            "classes": self.classes if self.is_trained else [],
            "num_classes": len(self.classes) if self.is_trained else 0,
        })
        return info
