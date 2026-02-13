"""
CLIP-based image classification and embedding model.

Supports:
- Zero-shot image classification
- Few-shot learning with classifier head
- Image and text embeddings (same vector space)
- Cross-modal similarity search
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Literal

import numpy as np

from .vision_base import (
    ClassificationModel,
    ClassificationResult,
    EmbeddingModel,
    EmbeddingResult,
)

if TYPE_CHECKING:
    import torch
    from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)


# Supported CLIP variants
CLIP_VARIANTS = {
    "clip-vit-base": "openai/clip-vit-base-patch32",
    "clip-vit-large": "openai/clip-vit-large-patch14",
    "clip-vit-base-16": "openai/clip-vit-base-patch16",
    # SigLIP (better for classification)
    "siglip-base": "google/siglip-base-patch16-224",
    "siglip-large": "google/siglip-large-patch16-256",
}

# Default prompt template for zero-shot classification
DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"


class CLIPClassifier(ClassificationModel):
    """CLIP-based image classifier.

    Supports zero-shot classification using text prompts and few-shot
    learning by training a classifier head on image embeddings.

    Example:
        ```python
        model = CLIPClassifier("clip-vit-base")
        await model.load()

        # Zero-shot classification
        result = await model.classify(
            image_bytes,
            classes=["cat", "dog", "bird"]
        )
        print(f"Predicted: {result.class_name} ({result.confidence:.2f})")
        ```
    """

    def __init__(
        self,
        model_id: str = "clip-vit-base",
        device: str = "auto",
        token: str | None = None,
        prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    ):
        """Initialize CLIP classifier.

        Args:
            model_id: Model variant or HuggingFace model ID
            device: Device to run on (auto, cpu, cuda, mps)
            token: HuggingFace token for gated models
            prompt_template: Template for zero-shot prompts (use {} for class name)
        """
        super().__init__(model_id, device, token)
        self.prompt_template = prompt_template
        self.clip_model: AutoModel | None = None
        self.processor: AutoProcessor | None = None
        self._class_embeddings: torch.Tensor | None = None
        self._embedding_dim: int = 0

    async def load(self) -> None:
        """Load the CLIP model and processor."""
        if self._loaded:
            logger.debug(f"CLIP model {self.model_id} already loaded")
            return

        from transformers import AutoModel, AutoProcessor

        # Resolve device
        self.device = self._resolve_device(self.device)
        logger.info(f"Loading CLIP model {self.model_id} on {self.device}")

        start_time = time.perf_counter()

        # Resolve model ID
        hf_model_id = CLIP_VARIANTS.get(self.model_id, self.model_id)

        def _load_model():
            # Load model and processor (AutoModel/AutoProcessor handle both CLIP and SigLIP)
            model = AutoModel.from_pretrained(
                hf_model_id,
                token=self.token,
            )
            processor = AutoProcessor.from_pretrained(
                hf_model_id,
                token=self.token,
            )
            # Move to device
            model = model.to(self.device)
            model.eval()
            return model, processor

        self.clip_model, self.processor = await asyncio.to_thread(_load_model)

        # Get embedding dimension
        self._embedding_dim = self.clip_model.config.projection_dim

        self._loaded = True
        load_time = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"CLIP model {self.model_id} loaded in {load_time:.1f}ms "
            f"(embedding dim: {self._embedding_dim})"
        )

    async def unload(self) -> None:
        """Unload the CLIP model."""
        self.clip_model = None
        self.processor = None
        self._class_embeddings = None
        self._loaded = False
        await super().unload()

    async def set_classes(self, class_names: list[str]) -> None:
        """Pre-compute text embeddings for class names.

        This improves classification speed when using the same classes
        repeatedly.

        Args:
            class_names: List of class names
        """
        if not self._loaded:
            await self.load()

        import torch

        self.class_names = class_names

        # Create prompts from class names
        prompts = [self.prompt_template.format(name) for name in class_names]

        def _encode_classes():
            # Encode text
            text_inputs = self.processor(
                text=prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**text_inputs)
                # Normalize embeddings
                return text_features / text_features.norm(dim=-1, keepdim=True)

        self._class_embeddings = await asyncio.to_thread(_encode_classes)

        logger.debug(f"Pre-computed embeddings for {len(class_names)} classes")

    async def classify(
        self,
        image: bytes | np.ndarray,
        classes: list[str] | None = None,
        top_k: int = 5,
    ) -> ClassificationResult:
        """Classify an image.

        Args:
            image: Input image as bytes or numpy array
            classes: Class names for zero-shot classification
            top_k: Number of top predictions to include in all_scores

        Returns:
            ClassificationResult with predicted class and scores
        """
        if not self._loaded:
            await self.load()

        import torch

        start_time = time.perf_counter()

        # Use provided classes or pre-computed ones
        if classes:
            await self.set_classes(classes)
        elif not self.class_names or self._class_embeddings is None:
            raise ValueError("No classes provided. Call set_classes() or pass classes parameter.")

        # Process image
        pil_image = self._image_to_pil(image)

        def _run_inference():
            image_inputs = self.processor(
                images=pil_image,
                return_tensors="pt",
            ).to(self.device)

            # Get image embedding
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**image_inputs)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Use trained classifier head if available, otherwise zero-shot
                if hasattr(self, "_classifier_head") and self._classifier_head is not None:
                    logits = self._classifier_head(image_features).squeeze()
                    probs = logits.softmax(dim=-1)
                else:
                    # Zero-shot: compute similarity with class embeddings
                    similarity = (image_features @ self._class_embeddings.T).squeeze()
                    probs = similarity.softmax(dim=-1)

            return probs.cpu().numpy()

        probs_np = await asyncio.to_thread(_run_inference)

        inference_time = (time.perf_counter() - start_time) * 1000

        # Get top predictions
        top_indices = np.argsort(probs_np)[::-1][:top_k]

        best_idx = int(top_indices[0])
        best_prob = float(probs_np[best_idx])

        # Build all_scores dict
        all_scores = {
            self.class_names[i]: float(probs_np[i])
            for i in top_indices
        }

        return ClassificationResult(
            confidence=best_prob,
            inference_time_ms=inference_time,
            model_name=self.model_id,
            class_name=self.class_names[best_idx],
            class_id=best_idx,
            all_scores=all_scores,
        )

    async def train_few_shot(
        self,
        images: list[bytes | np.ndarray],
        labels: list[str],
        epochs: int = 5,
        learning_rate: float = 1e-3,
    ) -> dict:
        """Train a classifier head with few-shot examples.

        Uses contrastive learning (SetFit-style) to train a small
        classifier head on top of frozen CLIP embeddings.

        Args:
            images: List of training images
            labels: List of labels (one per image)
            epochs: Number of training epochs
            learning_rate: Learning rate for classifier head

        Returns:
            Training metrics dictionary
        """
        if not self._loaded:
            await self.load()

        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        logger.info(f"Starting few-shot training with {len(images)} examples")

        # Get unique classes
        unique_labels = list(set(labels))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.class_names = unique_labels

        # Convert images to PIL upfront (lightweight)
        pil_images = [self._image_to_pil(img) for img in images]

        def _train():
            # Extract embeddings for all images
            embeddings = []
            for pil_image in pil_images:
                inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    features = self.clip_model.get_image_features(**inputs)
                    features = features / features.norm(dim=-1, keepdim=True)
                    embeddings.append(features)

            embeddings = torch.cat(embeddings, dim=0)
            targets = torch.tensor([label_to_idx[l] for l in labels], device=self.device)

            # Simple classifier head
            classifier = nn.Linear(self._embedding_dim, len(unique_labels)).to(self.device)
            optimizer = AdamW(classifier.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()

            # Training loop
            classifier.train()
            losses = []

            for epoch in range(epochs):
                optimizer.zero_grad()
                logits = classifier(embeddings)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

                if (epoch + 1) % max(1, epochs // 5) == 0:
                    logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

            # Compute final accuracy
            classifier.eval()
            with torch.no_grad():
                logits = classifier(embeddings)
                preds = logits.argmax(dim=-1)
                accuracy = (preds == targets).float().mean().item()

            return classifier, losses, accuracy

        classifier, losses, accuracy = await asyncio.to_thread(_train)

        # Store classifier for inference
        self._classifier_head = classifier

        return {
            "epochs": epochs,
            "final_loss": losses[-1],
            "accuracy": accuracy,
            "num_classes": len(unique_labels),
            "num_examples": len(images),
        }

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "variant": self.model_id,
            "embedding_dim": self._embedding_dim,
            "num_classes": len(self.class_names) if self.class_names else 0,
            "has_classifier_head": hasattr(self, "_classifier_head"),
        })
        return info


class CLIPEmbedder(EmbeddingModel):
    """CLIP-based image and text embedder.

    Generates embeddings in a shared vector space for both images
    and text, enabling cross-modal similarity search.

    Example:
        ```python
        model = CLIPEmbedder("clip-vit-base")
        await model.load()

        # Embed images
        img_result = await model.embed_images([image_bytes])

        # Embed text
        txt_result = await model.embed_texts(["a photo of a cat"])

        # Compute similarity
        similarity = np.dot(img_result.embeddings[0], txt_result.embeddings[0])
        ```
    """

    def __init__(
        self,
        model_id: str = "clip-vit-base",
        device: str = "auto",
        token: str | None = None,
    ):
        """Initialize CLIP embedder.

        Args:
            model_id: Model variant or HuggingFace model ID
            device: Device to run on (auto, cpu, cuda, mps)
            token: HuggingFace token for gated models
        """
        super().__init__(model_id, device, token)
        self.clip_model: AutoModel | None = None
        self.processor: AutoProcessor | None = None

    async def load(self) -> None:
        """Load the CLIP model and processor."""
        if self._loaded:
            return

        from transformers import AutoModel, AutoProcessor

        self.device = self._resolve_device(self.device)
        logger.info(f"Loading CLIP embedder {self.model_id} on {self.device}")

        start_time = time.perf_counter()

        hf_model_id = CLIP_VARIANTS.get(self.model_id, self.model_id)

        def _load_model():
            # AutoModel/AutoProcessor handle both CLIP and SigLIP architectures
            model = AutoModel.from_pretrained(hf_model_id, token=self.token)
            processor = AutoProcessor.from_pretrained(hf_model_id, token=self.token)
            model = model.to(self.device)
            model.eval()
            return model, processor

        self.clip_model, self.processor = await asyncio.to_thread(_load_model)

        self.embedding_dim = self.clip_model.config.projection_dim

        self._loaded = True
        load_time = (time.perf_counter() - start_time) * 1000
        logger.info(f"CLIP embedder loaded in {load_time:.1f}ms")

    async def unload(self) -> None:
        """Unload the CLIP model."""
        self.clip_model = None
        self.processor = None
        self._loaded = False
        await super().unload()

    async def embed_images(
        self,
        images: list[bytes | np.ndarray],
    ) -> EmbeddingResult:
        """Generate embeddings for images.

        Args:
            images: List of images as bytes or numpy arrays

        Returns:
            EmbeddingResult with embedding vectors
        """
        if not self._loaded:
            await self.load()

        import torch

        start_time = time.perf_counter()

        # Process all images
        pil_images = [self._image_to_pil(img) for img in images]

        def _embed():
            inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)

            # Get embeddings
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                # Normalize
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().tolist()

        embeddings = await asyncio.to_thread(_embed)
        inference_time = (time.perf_counter() - start_time) * 1000

        return EmbeddingResult(
            confidence=1.0,
            inference_time_ms=inference_time,
            model_name=self.model_id,
            embeddings=embeddings,
            dimensions=self.embedding_dim,
        )

    async def embed_texts(
        self,
        texts: list[str],
    ) -> EmbeddingResult:
        """Generate embeddings for texts.

        Args:
            texts: List of text strings

        Returns:
            EmbeddingResult with embedding vectors
        """
        if not self._loaded:
            await self.load()

        import torch

        start_time = time.perf_counter()

        def _embed():
            # Process texts
            inputs = self.processor(
                text=texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                features = self.clip_model.get_text_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            return features.cpu().numpy().tolist()

        embeddings = await asyncio.to_thread(_embed)
        inference_time = (time.perf_counter() - start_time) * 1000

        return EmbeddingResult(
            confidence=1.0,
            inference_time_ms=inference_time,
            model_name=self.model_id,
            embeddings=embeddings,
            dimensions=self.embedding_dim,
        )

    async def export(
        self,
        format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"],
        output_path: str,
        **kwargs,
    ) -> str:
        """Export CLIP model to ONNX using HuggingFace optimum.

        Only ONNX format is supported for CLIP models.

        Args:
            format: Must be "onnx"
            output_path: Directory to save the exported model
            **kwargs: Ignored for CLIP export

        Returns:
            Path to exported model.onnx
        """
        if format != "onnx":
            raise ValueError(f"CLIP export only supports ONNX, got {format}")

        from pathlib import Path

        try:
            from optimum.onnxruntime import ORTModelForFeatureExtraction
        except ImportError:
            raise ImportError(
                "ONNX export requires the 'optimum' package with ONNX Runtime support. "
                "Install it with: pip install optimum[onnxruntime]"
            ) from None
        from transformers import AutoTokenizer

        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        hf_model_id = CLIP_VARIANTS.get(self.model_id, self.model_id)

        logger.info(f"Exporting CLIP model {hf_model_id} to ONNX")

        ort_model = ORTModelForFeatureExtraction.from_pretrained(
            hf_model_id, export=True
        )
        ort_model.save_pretrained(output_dir)

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        tokenizer.save_pretrained(output_dir)

        export_file = output_dir / "model.onnx"
        logger.info(f"CLIP model exported to: {export_file}")

        return str(export_file)

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "variant": self.model_id,
            "embedding_dim": self.embedding_dim,
        })
        return info
