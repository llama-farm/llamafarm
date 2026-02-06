"""
Model wrappers for Universal Runtime.

Supports HuggingFace Transformers, Diffusers, GGUF models, OCR, document understanding,
anomaly detection, text classification, speech-to-text, and vision (detection, classification,
segmentation).
"""

from .anomaly_model import AnomalyModel
from .base import BaseModel
from .classifier_model import ClassifierModel
from .document_model import DocumentModel
from .emotion_model import EmotionModel, EmotionResult
from .encoder_model import EncoderModel
from .gguf_encoder_model import GGUFEncoderModel
from .gguf_language_model import GGUFLanguageModel
from .language_model import LanguageModel
from .ocr_model import OCRModel
from .speech_model import SpeechModel
from .tts_model import (
    ChatterboxConfig,
    TTSModel,
    VoiceProfile,
    check_builtin_voices,
    download_builtin_voices,
)
from .vision_base import (
    VisionModel,
    VisionResult,
    DetectionModel,
    DetectionResult,
    DetectionBox,
    ClassificationModel,
    ClassificationResult,
    SegmentationModel,
    SegmentationResult,
    SegmentationMask,
    EmbeddingModel,
    EmbeddingResult,
)
from .yolo_model import YOLOModel
from .clip_model import CLIPClassifier, CLIPEmbedder
from .streaming_vision import (
    StreamingVisionDetector,
    StreamingConfig,
    StreamSession,
    FrameResult,
    get_streaming_detector,
    set_streaming_model_loader,
)

__all__ = [
    # Base
    "BaseModel",
    # Language
    "LanguageModel",
    "GGUFLanguageModel",
    # Encoder
    "EncoderModel",
    "GGUFEncoderModel",
    # OCR & Document
    "OCRModel",
    "DocumentModel",
    # Anomaly
    "AnomalyModel",
    # Classifier
    "ClassifierModel",
    # Emotion
    "EmotionModel",
    "EmotionResult",
    # Speech
    "SpeechModel",
    # TTS
    "TTSModel",
    "VoiceProfile",
    "ChatterboxConfig",
    "check_builtin_voices",
    "download_builtin_voices",
    # Vision base (new)
    "VisionModel",
    "VisionResult",
    "DetectionModel",
    "DetectionResult",
    "DetectionBox",
    "ClassificationModel",
    "ClassificationResult",
    "SegmentationModel",
    "SegmentationResult",
    "SegmentationMask",
    "EmbeddingModel",
    "EmbeddingResult",
    # Vision implementations (new)
    "YOLOModel",
    "CLIPClassifier",
    "CLIPEmbedder",
    # Streaming vision
    "StreamingVisionDetector",
    "StreamingConfig",
    "StreamSession",
    "FrameResult",
    "get_streaming_detector",
    "set_streaming_model_loader",
]
