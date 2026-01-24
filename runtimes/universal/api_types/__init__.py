"""Universal Runtime type definitions.

This module exports all request/response types for the Universal Runtime API.

Note: This module is named 'api_types' rather than 'types' to avoid conflicting
with Python's built-in 'types' module.
"""

from .anomaly import (
    AnomalyDeleteResponse,
    AnomalyFitRequest,
    AnomalyFitResponse,
    AnomalyLoadRequest,
    AnomalyLoadResponse,
    AnomalyModelInfo,
    AnomalyModelsResponse,
    AnomalySaveRequest,
    AnomalySaveResponse,
    AnomalyScoreRequest,
    AnomalyScoreResponse,
    AnomalyScoreResult,
)
from .audio import (
    TranscriptionRequest,
    TranscriptionResponse,
    TranscriptionSegment,
    TranscriptionWord,
    TranslationRequest,
    TranslationResponse,
)
from .classifier import (
    ClassifierDeleteResponse,
    ClassifierFitRequest,
    ClassifierFitResponse,
    ClassifierLoadRequest,
    ClassifierLoadResponse,
    ClassifierModelInfo,
    ClassifierModelsResponse,
    ClassifierPrediction,
    ClassifierPredictRequest,
    ClassifierPredictResponse,
    ClassifierSaveRequest,
    ClassifierSaveResponse,
)
from .common import ErrorDetail, ListResponse, UsageInfo
from .nlp import (
    ClassifyRequest,
    ClassifyResponse,
    ClassifyResult,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EntityResult,
    NERRequest,
    NERResponse,
    NERResult,
    RerankRequest,
    RerankResponse,
    RerankResult,
)
from .vision import (
    DocumentExtractRequest,
    DocumentField,
    DocumentResponse,
    DocumentResult,
    OCRBox,
    OCRRequest,
    OCRResponse,
    OCRResult,
)

__all__ = [
    # Common
    "UsageInfo",
    "ListResponse",
    "ErrorDetail",
    # NLP
    "EmbeddingRequest",
    "EmbeddingData",
    "EmbeddingResponse",
    "RerankRequest",
    "RerankResult",
    "RerankResponse",
    "ClassifyRequest",
    "ClassifyResult",
    "ClassifyResponse",
    "NERRequest",
    "EntityResult",
    "NERResult",
    "NERResponse",
    # Anomaly
    "AnomalyScoreRequest",
    "AnomalyFitRequest",
    "AnomalySaveRequest",
    "AnomalyLoadRequest",
    "AnomalyScoreResult",
    "AnomalyScoreResponse",
    "AnomalyFitResponse",
    "AnomalySaveResponse",
    "AnomalyLoadResponse",
    "AnomalyModelInfo",
    "AnomalyModelsResponse",
    "AnomalyDeleteResponse",
    # Classifier
    "ClassifierFitRequest",
    "ClassifierPredictRequest",
    "ClassifierSaveRequest",
    "ClassifierLoadRequest",
    "ClassifierPrediction",
    "ClassifierPredictResponse",
    "ClassifierFitResponse",
    "ClassifierSaveResponse",
    "ClassifierLoadResponse",
    "ClassifierModelInfo",
    "ClassifierModelsResponse",
    "ClassifierDeleteResponse",
    # Vision
    "OCRRequest",
    "OCRBox",
    "OCRResult",
    "OCRResponse",
    "DocumentExtractRequest",
    "DocumentField",
    "DocumentResult",
    "DocumentResponse",
    # Audio
    "TranscriptionRequest",
    "TranscriptionSegment",
    "TranscriptionWord",
    "TranscriptionResponse",
    "TranslationRequest",
    "TranslationResponse",
]
