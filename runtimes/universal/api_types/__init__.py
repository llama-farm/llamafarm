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

# Conditional import for timeseries (requires darts package)
try:
    from .timeseries import (
        TimeseriesDeleteResponse,
        TimeseriesFitRequest,
        TimeseriesFitResponse,
        TimeseriesForecastRequest,
        TimeseriesForecastResponse,
        TimeseriesLoadRequest,
        TimeseriesLoadResponse,
        TimeseriesModelInfo,
        TimeseriesModelsResponse,
        TimeseriesPrediction,
        TimeseriesSaveRequest,
        TimeseriesSaveResponse,
    )

    _HAS_TIMESERIES_TYPES = True
except ImportError:
    _HAS_TIMESERIES_TYPES = False

# Conditional import for ADTK (requires adtk package)
try:
    from .adtk import (
        ADTKDeleteResponse,
        ADTKDetector,
        ADTKDetectorsResponse,
        ADTKDetectRequest,
        ADTKDetectResponse,
        ADTKFitRequest,
        ADTKFitResponse,
        ADTKLoadRequest,
        ADTKLoadResponse,
        ADTKModelInfo,
        ADTKModelsResponse,
    )

    _HAS_ADTK_TYPES = True
except ImportError:
    _HAS_ADTK_TYPES = False

# Conditional import for Drift Detection (requires alibi_detect package)
try:
    from .drift import (
        DriftDeleteResponse,
        DriftDetector,
        DriftDetectorsResponse,
        DriftDetectRequest,
        DriftDetectResponse,
        DriftFitRequest,
        DriftFitResponse,
        DriftLoadRequest,
        DriftLoadResponse,
        DriftModelInfo,
        DriftModelsResponse,
        DriftResetResponse,
        DriftStatusResponse,
    )

    _HAS_DRIFT_TYPES = True
except ImportError:
    _HAS_DRIFT_TYPES = False

# Conditional import for CatBoost (requires catboost package)
try:
    from .catboost import (
        CatBoostDeleteResponse,
        CatBoostFeatureImportance,
        CatBoostFeatureImportanceResponse,
        CatBoostFitRequest,
        CatBoostFitResponse,
        CatBoostInfoResponse,
        CatBoostLoadRequest,
        CatBoostLoadResponse,
        CatBoostModelInfo,
        CatBoostModelsResponse,
        CatBoostPrediction,
        CatBoostPredictRequest,
        CatBoostPredictResponse,
        CatBoostUpdateRequest,
        CatBoostUpdateResponse,
    )

    _HAS_CATBOOST_TYPES = True
except ImportError:
    _HAS_CATBOOST_TYPES = False

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

# Add timeseries types to __all__ if available
if _HAS_TIMESERIES_TYPES:
    __all__.extend(
        [
            "TimeseriesFitRequest",
            "TimeseriesForecastRequest",
            "TimeseriesSaveRequest",
            "TimeseriesLoadRequest",
            "TimeseriesPrediction",
            "TimeseriesForecastResponse",
            "TimeseriesFitResponse",
            "TimeseriesSaveResponse",
            "TimeseriesLoadResponse",
            "TimeseriesModelInfo",
            "TimeseriesModelsResponse",
            "TimeseriesDeleteResponse",
        ]
    )

# Add ADTK types to __all__ if available
if _HAS_ADTK_TYPES:
    __all__.extend(
        [
            "ADTKFitRequest",
            "ADTKDetectRequest",
            "ADTKLoadRequest",
            "ADTKDetector",
            "ADTKDetectorsResponse",
            "ADTKDetectResponse",
            "ADTKFitResponse",
            "ADTKLoadResponse",
            "ADTKModelInfo",
            "ADTKModelsResponse",
            "ADTKDeleteResponse",
        ]
    )

# Add Drift Detection types to __all__ if available
if _HAS_DRIFT_TYPES:
    __all__.extend(
        [
            "DriftFitRequest",
            "DriftDetectRequest",
            "DriftLoadRequest",
            "DriftDetector",
            "DriftDetectorsResponse",
            "DriftDetectResponse",
            "DriftFitResponse",
            "DriftLoadResponse",
            "DriftModelInfo",
            "DriftModelsResponse",
            "DriftDeleteResponse",
            "DriftResetResponse",
            "DriftStatusResponse",
        ]
    )

# Add CatBoost types to __all__ if available
if _HAS_CATBOOST_TYPES:
    __all__.extend(
        [
            "CatBoostFitRequest",
            "CatBoostPredictRequest",
            "CatBoostUpdateRequest",
            "CatBoostLoadRequest",
            "CatBoostPrediction",
            "CatBoostPredictResponse",
            "CatBoostFitResponse",
            "CatBoostUpdateResponse",
            "CatBoostLoadResponse",
            "CatBoostModelInfo",
            "CatBoostModelsResponse",
            "CatBoostDeleteResponse",
            "CatBoostInfoResponse",
            "CatBoostFeatureImportance",
            "CatBoostFeatureImportanceResponse",
        ]
    )
