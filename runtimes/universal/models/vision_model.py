"""
Vision model wrapper for image classification and understanding.
"""

from transformers import (
    AutoModelForImageClassification,
    AutoProcessor,
    AutoImageProcessor,
    CLIPModel,
    CLIPProcessor,
)
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import io
import base64
import logging

from .base import BaseModel

logger = logging.getLogger(__name__)


class VisionModel(BaseModel):
    """Wrapper for HuggingFace vision models (ViT, CLIP, etc.)."""

    def __init__(self, model_id: str, device: str, task: str = "classification"):
        """
        Initialize vision model.

        Args:
            model_id: HuggingFace model ID
            device: Target device (cuda/mps/cpu)
            task: Model task - "classification", "embedding", or "clip"
        """
        super().__init__(model_id, device)
        self.task = task
        self.model_type = f"vision_{task}"
        self.supports_streaming = False

    async def load(self):
        """Load the vision model."""
        logger.info(f"Loading vision model ({self.task}): {self.model_id}")

        dtype = self.get_dtype()

        # Load model and processor based on task
        if self.task == "clip" or "clip" in self.model_id.lower():
            self.model = CLIPModel.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
            )
            self.processor = CLIPProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self.task = "clip"
        else:
            # For classification or general image models
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_id,
                dtype=dtype,
                trust_remote_code=True,
            )
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )
            except Exception:
                # Fallback to image processor
                self.processor = AutoImageProcessor.from_pretrained(
                    self.model_id, trust_remote_code=True
                )

        self.model = self.model.to(self.device)
        self.model.eval()

        logger.info(f"Vision model loaded on {self.device}")

    def _decode_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Decode image from various input formats."""
        if isinstance(image_input, Image.Image):
            return image_input

        if isinstance(image_input, str):
            # Base64 string
            if image_input.startswith("data:image"):
                image_input = image_input.split(",", 1)[1]
            image_bytes = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_bytes))

        if isinstance(image_input, bytes):
            return Image.open(io.BytesIO(image_input))

        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    async def classify(
        self, images: List[Union[str, bytes, Image.Image]], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Classify images.

        Args:
            images: List of images (base64, bytes, or PIL Image)
            top_k: Number of top predictions to return

        Returns:
            List of classification results
        """
        if self.task == "clip":
            raise ValueError("Use clip_classify() for CLIP models")

        # Decode images
        pil_images = [self._decode_image(img) for img in images]

        # Process images
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Classify
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Format results
        results = []
        for pred in predictions:
            scores = pred.cpu().tolist()

            # Get top-k predictions
            top_indices = (
                torch.topk(pred, k=min(top_k, len(scores))).indices.cpu().tolist()
            )

            top_predictions = [
                {
                    "label": self.model.config.id2label.get(idx, str(idx)),
                    "score": scores[idx],
                }
                for idx in top_indices
            ]

            results.append({"predictions": top_predictions})

        return results

    async def embed_image(
        self, images: List[Union[str, bytes, Image.Image]], normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for images (requires CLIP or similar model).

        Args:
            images: List of images
            normalize: Whether to normalize embeddings

        Returns:
            List of embedding vectors
        """
        if self.task != "clip":
            raise ValueError("Image embeddings require CLIP model")

        # Decode images
        pil_images = [self._decode_image(img) for img in images]

        # Process images
        inputs = self.processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Normalize
        if normalize:
            image_features = F.normalize(image_features, p=2, dim=1)

        return image_features.cpu().tolist()

    async def embed_text(
        self, texts: List[str], normalize: bool = True
    ) -> List[List[float]]:
        """
        Generate embeddings for text (CLIP only).

        Args:
            texts: List of texts
            normalize: Whether to normalize embeddings

        Returns:
            List of embedding vectors
        """
        if self.task != "clip":
            raise ValueError("Text embeddings require CLIP model")

        # Process texts
        inputs = self.processor(text=texts, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate embeddings
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Normalize
        if normalize:
            text_features = F.normalize(text_features, p=2, dim=1)

        return text_features.cpu().tolist()

    async def clip_classify(
        self,
        images: List[Union[str, bytes, Image.Image]],
        candidate_labels: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Zero-shot classification using CLIP.

        Args:
            images: List of images
            candidate_labels: List of possible labels

        Returns:
            Classification results with scores for each label
        """
        if self.task != "clip":
            raise ValueError("Zero-shot classification requires CLIP model")

        # Decode images
        pil_images = [self._decode_image(img) for img in images]

        # Process inputs
        inputs = self.processor(
            text=candidate_labels, images=pil_images, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get similarities
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        # Format results
        results = []
        for prob in probs:
            scores = prob.cpu().tolist()
            label_scores = [
                {"label": label, "score": score}
                for label, score in zip(candidate_labels, scores)
            ]
            # Sort by score descending
            label_scores.sort(key=lambda x: x["score"], reverse=True)
            results.append({"predictions": label_scores})

        return results

    async def generate(self, *args, **kwargs):
        """Not applicable for vision models."""
        raise NotImplementedError("Vision models do not support text generation")
