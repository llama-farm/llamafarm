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
    # Common (new)
    BoundingBox,
    # Detection (new)
    Detection,
    DetectRequest,
    DetectResponse,
    # OCR (existing)
    DocumentExtractRequest,
    DocumentField,
    DocumentResponse,
    DocumentResult,
    # Classification (new)
    ImageClassifyRequest,
    ImageClassifyResponse,
    # Embedding (new)
    ImageEmbedRequest,
    ImageEmbedResponse,
    Mask,
    ModelExportRequest,
    ModelExportResponse,
    ModelImportRequest,
    ModelImportResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    ModelUnloadRequest,
    ModelUnloadResponse,
    OCRBox,
    OCRRequest,
    OCRResponse,
    OCRResult,
    Point,
    ReviewBatchRequest,
    ReviewBatchResponse,
    ReviewDecision,
    ReviewDecisionResponse,
    # Review queue (new)
    ReviewItem,
    ReviewListRequest,
    ReviewListResponse,
    # Segmentation (new)
    SegmentRequest,
    SegmentResponse,
    StreamFrameRequest,
    StreamFrameResponse,
    # Streaming (new)
    StreamingConfig,
    StreamStartRequest,
    StreamStartResponse,
    StreamStopRequest,
    StreamStopResponse,
    # Training (new)
    TrainingConfig,
    TrainRequest,
    TrainResponse,
    TrainStatusRequest,
    TrainStatusResponse,
    VisionBackendInfo,
    VisionBackendsResponse,
    # Model management (new)
    VisionModelInfo,
    VisionModelsListResponse,
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
    # Vision - OCR (existing)
    "OCRRequest",
    "OCRBox",
    "OCRResult",
    "OCRResponse",
    "DocumentExtractRequest",
    "DocumentField",
    "DocumentResult",
    "DocumentResponse",
    # Vision - Common (new)
    "BoundingBox",
    "Point",
    "Mask",
    # Vision - Detection (new)
    "Detection",
    "DetectRequest",
    "DetectResponse",
    # Vision - Classification (new)
    "ImageClassifyRequest",
    "ImageClassifyResponse",
    # Vision - Segmentation (new)
    "SegmentRequest",
    "SegmentResponse",
    # Vision - Embedding (new)
    "ImageEmbedRequest",
    "ImageEmbedResponse",
    # Vision - Streaming (new)
    "StreamingConfig",
    "StreamStartRequest",
    "StreamStartResponse",
    "StreamFrameRequest",
    "StreamFrameResponse",
    "StreamStopRequest",
    "StreamStopResponse",
    # Vision - Training (new)
    "TrainingConfig",
    "TrainRequest",
    "TrainResponse",
    "TrainStatusRequest",
    "TrainStatusResponse",
    # Vision - Model Management (new)
    "VisionModelInfo",
    "ModelLoadRequest",
    "ModelLoadResponse",
    "ModelUnloadRequest",
    "ModelUnloadResponse",
    "ModelExportRequest",
    "ModelExportResponse",
    "ModelImportRequest",
    "ModelImportResponse",
    "VisionModelsListResponse",
    "VisionBackendInfo",
    "VisionBackendsResponse",
    # Vision - Review Queue (new)
    "ReviewItem",
    "ReviewDecision",
    "ReviewDecisionResponse",
    "ReviewListRequest",
    "ReviewListResponse",
    "ReviewBatchRequest",
    "ReviewBatchResponse",
    # Audio
    "TranscriptionRequest",
    "TranscriptionSegment",
    "TranscriptionWord",
    "TranscriptionResponse",
    "TranslationRequest",
    "TranslationResponse",
]
