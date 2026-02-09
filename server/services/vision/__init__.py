"""Vision services for LlamaFarm server.

These services handle communication with the Universal Runtime
for all vision-related operations.

NOTE: Vision methods are kept separate from universal_runtime_service.py
      to maintain modularity and keep the codebase manageable.
"""

from .classification_service import VisionClassificationService
from .detection_service import VisionDetectionService
from .embedding_service import VisionEmbeddingService
from .pipeline_service import (
    VisionFederationService,
    VisionModelService,
    VisionSegmentationService,
    VisionStreamingService,
    VisionTrainingService,
)
from .review_service import VisionReviewService

__all__ = [
    "VisionDetectionService",
    "VisionClassificationService",
    "VisionEmbeddingService",
    "VisionReviewService",
    "VisionStreamingService",
    "VisionTrainingService",
    "VisionModelService",
    "VisionSegmentationService",
    "VisionFederationService",
]
