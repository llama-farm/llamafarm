"""Vision model for zero-shot image classification using CLIP.

Provides zero-shot image classification using OpenAI's CLIP model.
No training required - just provide labels and classify images.

Supported image inputs:
- File path (local or URL)
- Base64-encoded image data
- PIL Image objects

Usage:
    model = CLIPVisionModel(model_id="clip")
    await model.load()
    result = await model.classify_zero_shot("image.png", ["cat", "dog", "bird"])
    # result = {"label": "cat", "score": 0.87, "all_scores": {...}}
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import torch

from models.base import BaseModel
from utils.image_utils import load_image

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Default CLIP model - good balance of speed and accuracy
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"


class CLIPVisionModel(BaseModel):
    """Zero-shot image classification model using CLIP.

    CLIP (Contrastive Language-Image Pre-training) enables classification
    of images into arbitrary categories without training. Simply provide
    a list of text labels and the model will predict probabilities for each.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace model name (default: openai/clip-vit-base-patch32)
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_CLIP_MODEL,
        token: str | None = None,
    ):
        """Initialize CLIP vision model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace model to load
            token: Optional HuggingFace token for private models
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "vision_clip"
        self.supports_streaming = False
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.model is not None and self.processor is not None

    async def load(self) -> None:
        """Load the CLIP model and processor.

        Loads the model from HuggingFace Hub. First load may take time
        to download model weights (~350MB for vit-base-patch32).
        """
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading CLIP model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"CLIP model loaded successfully: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
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

    async def unload(self) -> None:
        """Unload the model and free resources."""
        await super().unload()
        self._is_loaded = False

    async def classify_zero_shot(
        self,
        image: str | bytes | Any,
        labels: list[str],
    ) -> dict[str, Any]:
        """Classify an image using zero-shot learning.

        Args:
            image: Image input - can be:
                - File path (str or Path)
                - Base64-encoded string
                - Raw bytes
                - PIL Image object
            labels: List of text labels to classify against

        Returns:
            Dictionary with:
                - label: Best matching label (lowercase)
                - score: Confidence score (0-1)
                - all_scores: Dict mapping each label to its score
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._classify_zero_shot_sync,
            image,
            labels,
        )

    def _classify_zero_shot_sync(
        self,
        image: str | bytes | Any,
        labels: list[str],
    ) -> dict[str, Any]:
        """Synchronous zero-shot classification."""

        # Load and process image
        pil_image = load_image(image)

        # Normalize labels to lowercase for consistent output
        normalized_labels = [label.lower() for label in labels]
        text_prompts = [f"a photo of a {label}" for label in normalized_labels]

        # Process inputs
        inputs = self.processor(
            text=text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
        )

        # Move inputs to device
        inputs = {k: self.to_device(v) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get probabilities using softmax
        logits_per_image = outputs.logits_per_image
        probs = torch.softmax(logits_per_image, dim=1)
        probs = probs.squeeze().cpu().numpy()

        # Handle single label case
        probs = [float(probs)] if len(labels) == 1 else probs.tolist()

        # Build result
        all_scores = {
            label: float(prob)
            for label, prob in zip(normalized_labels, probs, strict=True)
        }
        best_idx = max(range(len(probs)), key=lambda i: probs[i])

        return {
            "label": normalized_labels[best_idx],
            "score": float(probs[best_idx]),
            "all_scores": all_scores,
        }

    async def classify_zero_shot_batch(
        self,
        images: list[str | bytes | Any],
        labels: list[str],
    ) -> list[dict[str, Any]]:
        """Classify multiple images using zero-shot learning.

        Args:
            images: List of image inputs (file paths, base64, bytes, or PIL Images)
            labels: List of text labels to classify against

        Returns:
            List of result dictionaries (same format as classify_zero_shot)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._classify_zero_shot_batch_sync,
            images,
            labels,
        )

    def _classify_zero_shot_batch_sync(
        self,
        images: list[str | bytes | Any],
        labels: list[str],
    ) -> list[dict[str, Any]]:
        """Synchronous batch zero-shot classification."""

        # Load all images
        pil_images = [load_image(img) for img in images]

        # Normalize labels to lowercase for consistent output
        normalized_labels = [label.lower() for label in labels]
        text_prompts = [f"a photo of a {label}" for label in normalized_labels]

        results = []
        for pil_image in pil_images:
            # Process inputs for this image
            inputs = self.processor(
                text=text_prompts,
                images=pil_image,
                return_tensors="pt",
                padding=True,
            )

            # Move inputs to device
            inputs = {k: self.to_device(v) for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Get probabilities using softmax
            logits_per_image = outputs.logits_per_image
            probs = torch.softmax(logits_per_image, dim=1)
            probs = probs.squeeze().cpu().numpy()

            # Handle single label case
            probs = [float(probs)] if len(labels) == 1 else probs.tolist()

            # Build result
            all_scores = {
                label: float(prob)
                for label, prob in zip(normalized_labels, probs, strict=True)
            }
            best_idx = max(range(len(probs)), key=lambda i: probs[i])

            results.append({
                "label": normalized_labels[best_idx],
                "score": float(probs[best_idx]),
                "all_scores": all_scores,
            })

        return results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
        })
        return info
