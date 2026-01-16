"""Open-Vocabulary Object Detection using OWL-ViT.

OWL-ViT (Vision Transformer for Open-World Localization) enables:
- Text-conditioned detection: Find objects based on text descriptions
- Image-conditioned detection: Find similar objects using reference images
- Zero-shot detection: Detect arbitrary objects without retraining

Usage:
    model = OpenVocabDetectionModel(model_id="owl-vit")
    await model.load()

    # Text-conditioned: "Find all cats and dogs"
    result = await model.detect_by_text(image, queries=["a cat", "a dog"])

    # Image-conditioned: "Find objects similar to this reference"
    result = await model.detect_by_image(image, query_images=[ref_cat_image])
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import torch

from models.base import BaseModel
from utils.image_utils import load_image

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Default OWL-ViT model - base variant with good speed/accuracy
DEFAULT_OWL_MODEL = "google/owlvit-base-patch32"


class OpenVocabDetectionModel(BaseModel):
    """Open-vocabulary object detection using OWL-ViT.

    OWL-ViT enables detecting arbitrary objects based on text descriptions
    or reference images, without retraining.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace model name
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_OWL_MODEL,
        token: str | None = None,
    ):
        """Initialize open-vocabulary detection model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace model to load
            token: Optional HuggingFace token
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "open_vocab_detection"
        self.supports_streaming = False
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.model is not None and self.processor is not None

    async def load(self) -> None:
        """Load the OWL-ViT model and processor."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading OWL-ViT model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"OWL-ViT model loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.processor = OwlViTProcessor.from_pretrained(
            self.hf_model_name,
            token=self.token,
        )
        self.model = OwlViTForObjectDetection.from_pretrained(
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

    async def detect_by_text(
        self,
        image: str | bytes | Any,
        queries: list[str],
        threshold: float = 0.1,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Detect objects based on text descriptions.

        This is the main use case for open-vocabulary detection. Provide
        text descriptions of what you're looking for, and the model will
        find and locate matching objects.

        Args:
            image: Image input - can be file path, base64 string, bytes, or PIL Image
            queries: List of text queries describing objects to find
                     (e.g., ["a photo of a cat", "a dog playing", "a red car"])
            threshold: Minimum confidence threshold (0-1). Lower values
                       find more objects but with less certainty.
            top_k: Optional limit on number of detections to return

        Returns:
            Dictionary with:
                - objects: List of detected objects with query, score, box
                - count: Number of objects detected
                - queries: The queries used
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not queries:
            raise ValueError("At least one text query is required")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._detect_by_text_sync,
            image,
            queries,
            threshold,
            top_k,
        )

    def _detect_by_text_sync(
        self,
        image: str | bytes | Any,
        queries: list[str],
        threshold: float,
        top_k: int | None,
    ) -> dict[str, Any]:
        """Synchronous text-conditioned detection."""
        pil_image = load_image(image)
        width, height = pil_image.size

        # Process inputs
        inputs = self.processor(text=queries, images=pil_image, return_tensors="pt")
        inputs = {k: self.to_device(v) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process
        target_sizes = torch.tensor([[height, width]])
        results = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]

        # Build output
        objects = []
        for score, label_id, box in zip(
            results["scores"].cpu().numpy(),
            results["labels"].cpu().numpy(),
            results["boxes"].cpu().numpy(),
            strict=True,
        ):
            objects.append({
                "query": queries[label_id],
                "label": queries[label_id],  # For compatibility
                "score": float(score),
                "box": {
                    "x1": float(box[0]),
                    "y1": float(box[1]),
                    "x2": float(box[2]),
                    "y2": float(box[3]),
                },
            })

        # Sort by score descending
        objects.sort(key=lambda x: x["score"], reverse=True)

        # Apply top_k limit
        if top_k is not None and top_k > 0:
            objects = objects[:top_k]

        return {
            "objects": objects,
            "count": len(objects),
            "queries": queries,
            "image_size": {"width": width, "height": height},
        }

    async def detect_by_image(
        self,
        image: str | bytes | Any,
        query_images: list[str | bytes | Any],
        threshold: float = 0.9,
        top_k: int | None = None,
    ) -> dict[str, Any]:
        """Detect objects similar to reference images.

        Use this for few-shot detection when you have example images
        of what you want to find.

        Args:
            image: Target image to search in
            query_images: Reference images showing examples of objects to find
            threshold: Minimum similarity threshold (0-1). Higher values
                       require closer matches.
            top_k: Optional limit on number of detections to return

        Returns:
            Dictionary with:
                - objects: List of detected objects with query_idx, score, box
                - count: Number of objects detected
                - num_queries: Number of reference images used
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not query_images:
            raise ValueError("At least one query image is required")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._detect_by_image_sync,
            image,
            query_images,
            threshold,
            top_k,
        )

    def _detect_by_image_sync(
        self,
        image: str | bytes | Any,
        query_images: list[str | bytes | Any],
        threshold: float,
        top_k: int | None,
    ) -> dict[str, Any]:
        """Synchronous image-conditioned detection."""
        pil_image = load_image(image)
        width, height = pil_image.size

        # Load query images
        pil_query_images = [load_image(q) for q in query_images]

        # Process inputs for image-guided detection
        inputs = self.processor(
            images=pil_image,
            query_images=pil_query_images,
            return_tensors="pt",
        )
        inputs = {k: self.to_device(v) for k, v in inputs.items()}

        # Run inference with image-guided mode
        with torch.no_grad():
            outputs = self.model.image_guided_detection(**inputs)

        # Post-process
        target_sizes = torch.tensor([[height, width]])
        results = self.processor.post_process_image_guided_detection(
            outputs,
            threshold=threshold,
            nms_threshold=0.3,
            target_sizes=target_sizes,
        )[0]

        # Build output - handle case where no detections are found
        objects = []
        scores = results.get("scores")
        labels = results.get("labels")
        boxes = results.get("boxes")

        if scores is not None and labels is not None and boxes is not None and len(scores) > 0:
            for score, label_id, box in zip(
                scores.cpu().numpy(),
                labels.cpu().numpy(),
                boxes.cpu().numpy(),
                strict=True,
            ):
                objects.append({
                    "query_index": int(label_id),
                    "score": float(score),
                    "box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    },
                })

        # Sort by score descending
        objects.sort(key=lambda x: x["score"], reverse=True)

        # Apply top_k limit
        if top_k is not None and top_k > 0:
            objects = objects[:top_k]

        return {
            "objects": objects,
            "count": len(objects),
            "num_queries": len(query_images),
            "image_size": {"width": width, "height": height},
        }

    async def detect_batch_by_text(
        self,
        images: list[str | bytes | Any],
        queries: list[str],
        threshold: float = 0.1,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Detect objects in multiple images using text queries.

        Processes all images in a single batch for better performance than
        calling detect_by_text in a loop.

        Args:
            images: List of image inputs
            queries: Text queries to search for (applied to all images)
            threshold: Minimum confidence threshold
            top_k: Optional limit per image

        Returns:
            List of detection results (one per image)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not images:
            return []

        if not queries:
            raise ValueError("At least one text query is required")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._detect_batch_by_text_sync,
            images,
            queries,
            threshold,
            top_k,
        )

    def _detect_batch_by_text_sync(
        self,
        images: list[str | bytes | Any],
        queries: list[str],
        threshold: float,
        top_k: int | None,
    ) -> list[dict[str, Any]]:
        """Synchronous batch text-conditioned detection."""
        # Load all images
        pil_images = [load_image(img) for img in images]
        target_sizes = torch.tensor([[img.size[1], img.size[0]] for img in pil_images])

        # Process all images in a single batch
        inputs = self.processor(
            text=queries, images=pil_images, return_tensors="pt", padding=True
        )
        inputs = {k: self.to_device(v) for k, v in inputs.items()}

        # Run inference on batch
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results for entire batch
        batch_results = self.processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )

        # Format results for each image
        final_results = []
        for results, pil_image in zip(batch_results, pil_images, strict=True):
            width, height = pil_image.size
            objects = []

            for score, label_id, box in zip(
                results["scores"].cpu().numpy(),
                results["labels"].cpu().numpy(),
                results["boxes"].cpu().numpy(),
                strict=True,
            ):
                objects.append({
                    "query": queries[label_id],
                    "label": queries[label_id],
                    "score": float(score),
                    "box": {
                        "x1": float(box[0]),
                        "y1": float(box[1]),
                        "x2": float(box[2]),
                        "y2": float(box[3]),
                    },
                })

            # Sort by score descending
            objects.sort(key=lambda x: x["score"], reverse=True)

            # Apply top_k limit
            if top_k is not None and top_k > 0:
                objects = objects[:top_k]

            final_results.append({
                "objects": objects,
                "count": len(objects),
                "queries": queries,
                "image_size": {"width": width, "height": height},
            })

        return final_results

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "capabilities": [
                "text_conditioned_detection",
                "image_conditioned_detection",
                "zero_shot_detection",
            ],
        })
        return info
