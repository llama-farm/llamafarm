"""Object Detection Model using YOLOS.

Provides object detection with bounding boxes using YOLOS
(You Only Look at One Sequence), a Vision Transformer-based detector.

Usage:
    model = ObjectDetectionModel(model_id="yolos")
    await model.load()
    result = await model.detect("image.jpg")
    # Returns: {"objects": [{"label": "cat", "score": 0.95, "box": [x1, y1, x2, y2]}, ...]}
"""

import asyncio
import base64
import logging
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from models.base import BaseModel

if TYPE_CHECKING:
    from PIL import Image

logger = logging.getLogger(__name__)

# Default YOLOS model - good balance of speed and accuracy
DEFAULT_YOLOS_MODEL = "hustvl/yolos-tiny"

# COCO class labels (80 classes)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


class ObjectDetectionModel(BaseModel):
    """Object detection model using YOLOS.

    YOLOS is a Vision Transformer-based object detector that uses
    a sequence-to-sequence approach for detection.

    Attributes:
        model_id: Unique identifier for this model instance
        device: Device to run inference on (cpu, cuda, mps)
        hf_model_name: HuggingFace model name
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        hf_model_name: str = DEFAULT_YOLOS_MODEL,
        token: str | None = None,
    ):
        """Initialize object detection model.

        Args:
            model_id: Unique identifier for this model instance
            device: Device to run on (cpu, cuda, mps)
            hf_model_name: HuggingFace model to load
            token: Optional HuggingFace token
        """
        super().__init__(model_id=model_id, device=device, token=token)
        self.hf_model_name = hf_model_name
        self.model_type = "object_detection"
        self.supports_streaming = False
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded and ready for inference."""
        return self._is_loaded and self.model is not None and self.processor is not None

    async def load(self) -> None:
        """Load the YOLOS model and processor."""
        if self.is_loaded:
            logger.info(f"Model {self.model_id} already loaded")
            return

        logger.info(f"Loading YOLOS object detection model: {self.hf_model_name}")

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync)

        self._is_loaded = True
        logger.info(f"YOLOS model loaded: {self.model_id}")

    def _load_sync(self) -> None:
        """Synchronous model loading (run in executor)."""
        from transformers import YolosForObjectDetection, YolosImageProcessor

        self.processor = YolosImageProcessor.from_pretrained(
            self.hf_model_name,
            token=self.token,
        )
        # YOLOS has MPS compatibility issues with float16, force float32
        self.model = YolosForObjectDetection.from_pretrained(
            self.hf_model_name,
            token=self.token,
            torch_dtype=self.get_dtype(force_float32=(self.device == "mps")),
        )
        self.model = self.model.to(self.device)
        self.model.eval()

    async def unload(self) -> None:
        """Unload the model and free resources."""
        await super().unload()
        self._is_loaded = False

    async def detect(
        self,
        image: str | bytes | Any,
        threshold: float = 0.5,
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Detect objects in an image.

        Args:
            image: Image input - can be file path, base64 string, bytes, or PIL Image
            threshold: Minimum confidence threshold (0-1)
            labels: Optional list of labels to filter results (e.g., ["cat", "dog"])

        Returns:
            Dictionary with:
                - objects: List of detected objects with label, score, box
                - count: Number of objects detected
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._detect_sync,
            image,
            threshold,
            labels,
        )

    def _detect_sync(
        self,
        image: str | bytes | Any,
        threshold: float,
        labels: list[str] | None,
    ) -> dict[str, Any]:
        """Synchronous object detection."""
        # Load image
        pil_image = self._load_image(image)
        width, height = pil_image.size

        # Process image
        inputs = self.processor(images=pil_image, return_tensors="pt")
        # YOLOS on MPS requires float32 - use explicit dtype to avoid to_device defaulting to float16
        model_dtype = self.get_dtype(force_float32=(self.device == "mps"))
        inputs = {k: self.to_device(v, dtype=model_dtype) for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
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
            label = self.model.config.id2label[label_id]

            # Filter by labels if specified
            if labels and label not in labels:
                continue

            objects.append({
                "label": label,
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

        return {
            "objects": objects,
            "count": len(objects),
            "image_size": {"width": width, "height": height},
        }

    async def detect_batch(
        self,
        images: list[str | bytes | Any],
        threshold: float = 0.5,
        labels: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect objects in multiple images.

        Args:
            images: List of image inputs
            threshold: Minimum confidence threshold
            labels: Optional labels to filter

        Returns:
            List of detection results (one per image)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = []
        for image in images:
            result = await self.detect(image, threshold=threshold, labels=labels)
            results.append(result)
        return results

    def _load_image(self, image: str | bytes | Any) -> "Image.Image":
        """Load an image from various input formats."""
        from PIL import Image

        # Already a PIL Image
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        # Raw bytes
        if isinstance(image, bytes):
            return Image.open(BytesIO(image)).convert("RGB")

        # String input - could be file path or base64
        if isinstance(image, (str, Path)):
            # Long strings are almost certainly base64
            if isinstance(image, str) and len(image) > 4096:
                try:
                    img_bytes = base64.b64decode(image)
                    return Image.open(BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    raise ValueError(f"Invalid base64 image data: {str(e)[:100]}") from e

            # Could be file path
            path = Path(image)
            try:
                if path.exists():
                    return Image.open(path).convert("RGB")
            except OSError:
                pass

            # Try as base64
            try:
                img_bytes = base64.b64decode(image)
                return Image.open(BytesIO(img_bytes)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image path or base64: {str(e)[:100]}") from e

        raise ValueError(f"Unsupported image type: {type(image)}")

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "hf_model_name": self.hf_model_name,
            "is_loaded": self.is_loaded,
            "supported_classes": COCO_CLASSES,
        })
        return info
