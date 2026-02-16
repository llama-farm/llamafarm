"""Base classes for vision models (detection, classification).

Simplified MVP — no segmentation, no embedding model base.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .base import BaseModel

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
class EmbeddingResult(VisionResult):
    """Image/text embedding result."""
    embeddings: list[list[float]] = field(default_factory=list)
    dimensions: int = 0


# =============================================================================
# Base Model Classes
# =============================================================================


class VisionModel(BaseModel):
    """Base class for all vision models."""

    def __init__(self, model_id: str, device: str = "auto", token: str | None = None):
        super().__init__(model_id, device, token)
        self.model_type = "vision"
        self._loaded = False

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass  # torch not installed — fall back to CPU
        return "cpu"

    def _image_to_numpy(self, image: bytes | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        import io

        from PIL import Image
        img = Image.open(io.BytesIO(image))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)

    def _image_to_pil(self, image: bytes | np.ndarray):
        import io

        from PIL import Image
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        img = Image.open(io.BytesIO(image))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img

    def get_model_info(self) -> dict[str, Any]:
        info = super().get_model_info()
        info["loaded"] = self._loaded
        return info


class DetectionModel(VisionModel):
    """Base class for object detection models."""

    def __init__(self, model_id: str, device: str = "auto",
                 confidence_threshold: float = 0.5, token: str | None = None):
        super().__init__(model_id, device, token)
        self.confidence_threshold = confidence_threshold
        self.class_names: list[str] = []

    @abstractmethod
    async def detect(self, image: bytes | np.ndarray,
                     confidence_threshold: float | None = None,
                     classes: list[str] | None = None) -> DetectionResult:
        pass

    async def train(self, dataset_path: str, epochs: int = 10,
                    batch_size: int = 16, **kwargs) -> dict:
        raise NotImplementedError(f"{self.__class__.__name__} does not support training")

    async def export(self, format: Literal["onnx", "coreml", "tensorrt", "tflite", "openvino"],
                     output_path: str, **kwargs) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} does not support export to {format}")

    async def load(self) -> None:
        raise NotImplementedError

    async def infer(self, image: bytes | np.ndarray, **kwargs) -> VisionResult:
        return await self.detect(image, **kwargs)


class ClassificationModel(VisionModel):
    """Base class for image classification models."""

    def __init__(self, model_id: str, device: str = "auto", token: str | None = None):
        super().__init__(model_id, device, token)
        self.class_names: list[str] = []

    @abstractmethod
    async def classify(self, image: bytes | np.ndarray,
                       classes: list[str] | None = None,
                       top_k: int = 5) -> ClassificationResult:
        pass

    async def load(self) -> None:
        raise NotImplementedError

    async def infer(self, image: bytes | np.ndarray, **kwargs) -> VisionResult:
        return await self.classify(image, classes=kwargs.get("classes"), top_k=kwargs.get("top_k", 5))
