"""
Model wrappers for Universal Runtime.

Supports HuggingFace Transformers, Diffusers, GGUF models, OCR, document understanding,
anomaly detection, and text classification.

PyTorch-based models are optional - they're only available if torch is installed.
GGUF models work without PyTorch via llama-cpp.
"""

import logging

logger = logging.getLogger(__name__)

# GGUF models don't require PyTorch - always available
from .gguf_encoder_model import GGUFEncoderModel
from .gguf_language_model import GGUFLanguageModel

# PyTorch-based models are optional
_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    logger.info(
        "PyTorch not installed - encoder/transformer models will not be available"
    )


# Conditional imports for PyTorch-based models
if _TORCH_AVAILABLE:
    from .anomaly_model import AnomalyModel
    from .base import BaseModel
    from .classifier_model import ClassifierModel
    from .document_model import DocumentModel
    from .encoder_model import EncoderModel
    from .language_model import LanguageModel
    from .ocr_model import OCRModel
else:
    # Stub classes that raise helpful errors when PyTorch isn't available
    class _TorchRequiredStub:
        """Stub class for models that require PyTorch."""

        _model_name = "Model"

        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{self._model_name} requires PyTorch. Install with: pip install torch"
            )

    class BaseModel(_TorchRequiredStub):
        _model_name = "BaseModel"

    class LanguageModel(_TorchRequiredStub):
        _model_name = "LanguageModel"

    class EncoderModel(_TorchRequiredStub):
        _model_name = "EncoderModel"

    class OCRModel(_TorchRequiredStub):
        _model_name = "OCRModel"

    class DocumentModel(_TorchRequiredStub):
        _model_name = "DocumentModel"

    class AnomalyModel(_TorchRequiredStub):
        _model_name = "AnomalyModel"

    class ClassifierModel(_TorchRequiredStub):
        _model_name = "ClassifierModel"


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
]
