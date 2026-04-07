"""
Model wrappers for Edge Runtime.

Only includes model types needed for edge inference:
- Language models (GGUF and transformers)
- Vision models (YOLO detection, CLIP classification)
"""

from .base import BaseModel
from .clip_model import CLIPModel
from .gguf_language_model import GGUFLanguageModel
from .language_model import LanguageModel
from .vision_base import (
    ClassificationModel,
    ClassificationResult,
    DetectionBox,
    DetectionModel,
    DetectionResult,
    EmbeddingResult,
    VisionModel,
    VisionResult,
)
from .yolo_model import YOLOModel

try:
    from .hailo_model import HailoYOLOModel
except ImportError:
    HailoYOLOModel = None  # type: ignore[assignment,misc]

__all__ = [
    "BaseModel",
    "LanguageModel",
    "GGUFLanguageModel",
    "YOLOModel",
    "HailoYOLOModel",
    "CLIPModel",
    "VisionModel",
    "DetectionModel",
    "ClassificationModel",
    "VisionResult",
    "DetectionBox",
    "DetectionResult",
    "ClassificationResult",
    "EmbeddingResult",
]
