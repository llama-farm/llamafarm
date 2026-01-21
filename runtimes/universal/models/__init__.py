"""
Model wrappers for Universal Runtime.

Supports HuggingFace Transformers, Diffusers, GGUF models, OCR, document understanding,
anomaly detection, text classification, and speech-to-text.
"""

from .anomaly_model import AnomalyModel
from .base import BaseModel
from .classifier_model import ClassifierModel
from .document_model import DocumentModel
from .encoder_model import EncoderModel
from .gguf_encoder_model import GGUFEncoderModel
from .gguf_language_model import GGUFLanguageModel
from .language_model import LanguageModel
from .ocr_model import OCRModel
from .speech_model import SpeechModel

__all__ = [
    "BaseModel",
    "LanguageModel",
    "GGUFLanguageModel",
    "EncoderModel",
    "GGUFEncoderModel",
    "OCRModel",
    "DocumentModel",
    "AnomalyModel",
    "ClassifierModel",
    "SpeechModel",
]
