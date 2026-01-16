"""
Model wrappers for Universal Runtime.

Supports HuggingFace Transformers, Diffusers, GGUF models, OCR, document understanding,
anomaly detection, text classification, and speech-to-text.
"""

from .anomaly_model import AnomalyModel
from .background_removal_model import BackgroundRemovalModel
from .base import BaseModel
from .classifier_model import ClassifierModel
from .document_model import DocumentModel
from .encoder_model import EncoderModel
from .few_shot_classifier import FewShotImageClassifier
from .gguf_encoder_model import GGUFEncoderModel
from .gguf_language_model import GGUFLanguageModel
from .language_detection_model import LanguageDetectionModel
from .language_model import LanguageModel
from .object_detection_model import ObjectDetectionModel
from .ocr_model import OCRModel
from .open_vocab_detection_model import OpenVocabDetectionModel
from .pii_model import PIIModel
from .speech_model import SpeechModel
from .table_qa_model import TableQAModel
from .timeseries_model import TimeSeriesModel
from .vision_model import CLIPVisionModel

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
    "CLIPVisionModel",
    "FewShotImageClassifier",
    "LanguageDetectionModel",
    "PIIModel",
    "ObjectDetectionModel",
    "OpenVocabDetectionModel",
    "BackgroundRemovalModel",
    "TimeSeriesModel",
    "TableQAModel",
    "SpeechModel",
]
