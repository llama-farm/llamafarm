"""
Base classes for vision models (detection, classification, segmentation).

This module provides abstract base classes and result dataclasses for:
- Object detection (YOLO)
- Image classification (CLIP)
- Image segmentation (SAM/MobileSAM)
- Streaming vision detection
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from .base import BaseModel

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Result Dataclasses
# =============================================================================


@dataclass
class VisionResult:
    """Base result for all vision operations."""

    confidence: float
    inference_time_ms: float
    model_name: str


@dataclass
class DetectionBox:
    """Single detection bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    class_id: int
    confidence: float
    mask: list[list[float]] | None = None  # Polygon points [[x,y], ...]


@dataclass
class DetectionResult(VisionResult):
    """Object detection result."""

    boxes: list[DetectionBox] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0


@dataclass
class ClassificationResult(VisionResult):
    """Image classification result."""

    class_name: str = ""
    class_id: int = 0
    all_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class SegmentationMask:
    """Single segmentation mask."""

    mask: np.ndarray  # Binary mask (H, W)
    box: DetectionBox | None  # Bounding box if available
    confidence: float
    area: int


@dataclass
class SegmentationResult(VisionResult):
    """Image segmentation result."""

    masks: list[SegmentationMask] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0


@dataclass
class EmbeddingResult(VisionResult):
    """Image/text embedding result."""

    embeddings: list[list[float]] = field(default_factory=list)
    dimensions: int = 0


# =============================================================================
# Base Model Classes
# =============================================================================


class VisionModel(BaseModel):
    """Base class for all vision models.

    Extends BaseModel with vision-specific abstract methods for:
    - Detection
    - Classification
    - Segmentation
    - Training
    - Export
    """

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        token: str | None = None,
    ):
        super().__init__(model_id, device, token)
        self.model_type = "vision"
        self.task: Literal["detection", "classification", "segmentation", "embedding"] = "detection"
        self._loaded = False

    async def load(self) -> None:
        """Load the vision model.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load()")

    @abstractmethod
    async def infer(
        self,
        image: bytes | np.ndarray,
        **kwargs,
    ) -> VisionResult:
        """Run inference on a single image.

        Args:
            image: Image as bytes (JPEG/PNG) or numpy array (H, W, C)
            **kwargs: Model-specific parameters

        Returns:
            VisionResult subclass appropriate for the model task
        """
        pass

    async def train(
        self,
        dataset_path: str,
        epochs: int = 10,
        batch_size: int = 16,
        **kwargs,
    ) -> dict:
        """Train or fine-tune the model.

        Args:
            dataset_path: Path to training dataset (YOLO format, COCO, etc.)
            epochs: Number of training epochs
            batch_size: Training batch size
            **kwargs: Additional training parameters

        Returns:
            Training metrics dictionary
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support training")

    async def export(
        self,
        format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"],
        output_path: str,
        **kwargs,
    ) -> str:
        """Export model to deployment format.

        Args:
            format: Export format
            output_path: Output file path
            **kwargs: Format-specific options (quantization, etc.)

        Returns:
            Path to exported model
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support export to {format}")

    def _resolve_device(self, device: str) -> str:
        """Resolve 'auto' device to actual device."""
        if device != "auto":
            return device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        except ImportError:
            return "cpu"

    def _image_to_numpy(self, image: bytes | np.ndarray) -> np.ndarray:
        """Convert image bytes to numpy array.

        Args:
            image: Image as bytes (JPEG/PNG) or numpy array

        Returns:
            Numpy array in RGB format (H, W, C)
        """
        if isinstance(image, np.ndarray):
            return image

        from PIL import Image
        import io

        img = Image.open(io.BytesIO(image))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    def _image_to_pil(self, image: bytes | np.ndarray):
        """Convert image to PIL Image.

        Args:
            image: Image as bytes or numpy array

        Returns:
            PIL Image in RGB mode
        """
        from PIL import Image
        import io

        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        else:
            img = Image.open(io.BytesIO(image))
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the loaded model."""
        info = super().get_model_info()
        info.update({
            "task": self.task,
            "loaded": self._loaded,
        })
        return info


class DetectionModel(VisionModel):
    """Base class for object detection models (YOLO, etc.)."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        confidence_threshold: float = 0.5,
        token: str | None = None,
    ):
        super().__init__(model_id, device, token)
        self.task = "detection"
        self.confidence_threshold = confidence_threshold
        self.class_names: list[str] = []

    @abstractmethod
    async def detect(
        self,
        image: bytes | np.ndarray,
        confidence_threshold: float | None = None,
        classes: list[str] | None = None,
    ) -> DetectionResult:
        """Detect objects in image.

        Args:
            image: Input image
            confidence_threshold: Override default threshold
            classes: Filter to specific classes

        Returns:
            DetectionResult with bounding boxes
        """
        pass

    async def infer(self, image: bytes | np.ndarray, **kwargs) -> VisionResult:
        """Run detection inference."""
        return await self.detect(
            image,
            confidence_threshold=kwargs.get("confidence_threshold"),
            classes=kwargs.get("classes"),
        )


class ClassificationModel(VisionModel):
    """Base class for image classification models (CLIP, etc.)."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        token: str | None = None,
    ):
        super().__init__(model_id, device, token)
        self.task = "classification"
        self.class_names: list[str] = []

    @abstractmethod
    async def classify(
        self,
        image: bytes | np.ndarray,
        classes: list[str] | None = None,
        top_k: int = 5,
    ) -> ClassificationResult:
        """Classify image.

        Args:
            image: Input image
            classes: Classes for zero-shot classification
            top_k: Number of top predictions to return

        Returns:
            ClassificationResult with class scores
        """
        pass

    async def set_classes(self, class_names: list[str]) -> None:
        """Set classification classes (for zero-shot)."""
        self.class_names = class_names

    async def infer(self, image: bytes | np.ndarray, **kwargs) -> VisionResult:
        """Run classification inference."""
        return await self.classify(
            image,
            classes=kwargs.get("classes"),
            top_k=kwargs.get("top_k", 5),
        )


class SegmentationModel(VisionModel):
    """Base class for image segmentation models (SAM, MobileSAM, etc.)."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        token: str | None = None,
    ):
        super().__init__(model_id, device, token)
        self.task = "segmentation"

    @abstractmethod
    async def segment(
        self,
        image: bytes | np.ndarray,
        points: list[tuple[float, float, int]] | None = None,
        boxes: list[tuple[float, float, float, float]] | None = None,
        multimask_output: bool = False,
    ) -> SegmentationResult:
        """Segment image.

        Args:
            image: Input image
            points: Point prompts as (x, y, label) where label is 0 or 1
            boxes: Box prompts as (x1, y1, x2, y2)
            multimask_output: Return multiple mask candidates

        Returns:
            SegmentationResult with masks
        """
        pass

    async def infer(self, image: bytes | np.ndarray, **kwargs) -> VisionResult:
        """Run segmentation inference."""
        return await self.segment(
            image,
            points=kwargs.get("points"),
            boxes=kwargs.get("boxes"),
            multimask_output=kwargs.get("multimask_output", False),
        )


class EmbeddingModel(VisionModel):
    """Base class for image/text embedding models (CLIP, etc.)."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        token: str | None = None,
    ):
        super().__init__(model_id, device, token)
        self.task = "embedding"
        self.embedding_dim: int = 0

    @abstractmethod
    async def embed_images(
        self,
        images: list[bytes | np.ndarray],
    ) -> EmbeddingResult:
        """Generate embeddings for images.

        Args:
            images: List of images

        Returns:
            EmbeddingResult with embedding vectors
        """
        pass

    @abstractmethod
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
        pass

    async def infer(self, image: bytes | np.ndarray, **kwargs) -> VisionResult:
        """Run embedding inference on single image."""
        return await self.embed_images([image])
