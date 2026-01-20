"""
Pydantic models for ML endpoints.

These models mirror the Universal Runtime's request/response schemas
to provide a consistent API experience.

Note: OCR and Document extraction types have moved to vision/types.py
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

# =============================================================================
# SetFit Classifier Types
# =============================================================================


class ClassifierFitRequest(BaseModel):
    """Request to fit a text classifier."""

    model: str  # Model identifier (for caching/saving)
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    training_data: list[dict[str, str]]  # List of {"text": "...", "label": "..."}
    num_iterations: int = 20
    batch_size: int = 16
    overwrite: bool = (
        True  # If True, overwrite existing model; if False, version with timestamp
    )
    description: str | None = None  # Optional model description


class ClassifierPredictRequest(BaseModel):
    """Request to classify texts."""

    model: str  # Model identifier (must be fitted or loaded)
    texts: list[str]


class ClassifierSaveRequest(BaseModel):
    """Request to save a fitted classifier."""

    model: str  # Model identifier (must be fitted)
    description: str | None = None  # Optional model description


class ClassifierLoadRequest(BaseModel):
    """Request to load a pre-trained classifier."""

    model: str  # Model identifier to load


# =============================================================================
# Anomaly Detection Types
# =============================================================================


class AnomalyFitRequest(BaseModel):
    """Anomaly model fitting request.

    Supports two data formats:
    1. Numeric arrays: data = [[1.0, 2.0], [3.0, 4.0]]
    2. Dict-based with schema: data = [{"time_ms": 100, "user_agent": "curl"}]
       with schema = {"time_ms": "numeric", "user_agent": "hash"}

    Schema encoding types:
    - numeric: Pass through as-is (int/float)
    - hash: MD5 hash to integer (good for high-cardinality like user_agent)
    - label: Category → integer mapping (learned from training data)
    - onehot: One-hot encoding (for low-cardinality categoricals)
    - binary: Boolean-like values (yes/no, true/false → 0/1)
    - frequency: Encode as occurrence frequency from training data

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (varies by backend)
    """

    model: str = "default"  # Model identifier (for caching)
    backend: str = "isolation_forest"  # Backend to use
    data: list[list[float]] | list[dict[str, Any]] | None = None  # Training data
    training_file: str | None = None  # File reference ID from upload-training-data
    schema: dict[str, str] | None = None  # Feature encoding schema
    contamination: float = Field(
        default=0.1,
        gt=0,
        le=0.5,
        description="Expected proportion of anomalies (0-0.5]",
    )
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    scaler_type: Literal["robust", "standard"] = Field(
        default="robust",
        description="Input data scaler: robust (default) or standard",
    )
    epochs: int = Field(default=100, ge=1, description="Training epochs (autoencoder only)")
    batch_size: int = Field(default=32, ge=1, description="Batch size (autoencoder only)")
    # VAE / Early Stopping parameters
    validation_split: float = Field(
        default=0.1,
        ge=0,
        le=0.5,
        description="Fraction of data for validation (autoencoder/vae)",
    )
    patience: int = Field(
        default=10,
        ge=1,
        description="Epochs without improvement before stopping",
    )
    min_delta: float = Field(
        default=1e-4,
        ge=0,
        description="Minimum change in validation loss for improvement",
    )
    overwrite: bool = (
        True  # If True, overwrite existing model; if False, version with timestamp
    )
    description: str | None = None  # Optional model description


class AnomalyScoreRequest(BaseModel):
    """Anomaly scoring request.

    Normalization methods:
    - standardization (default): Sigmoid 0-1 range, threshold ~0.5
    - zscore: Standard deviations from mean, threshold ~2.0-3.0
    - raw: Backend-native scores (varies by backend)
    """

    model: str = "default"  # Model identifier
    backend: str = "isolation_forest"  # Backend
    data: list[list[float]] | list[dict[str, Any]]  # Data points
    schema: dict[str, str] | None = None  # Feature encoding schema
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    scaler_type: Literal["robust", "standard"] = Field(
        default="robust",
        description="Input data scaler: robust (default) or standard",
    )
    threshold: float | None = None  # Override default threshold


class AnomalySaveRequest(BaseModel):
    """Request to save a fitted anomaly model."""

    model: str  # Model identifier (must be fitted)
    backend: str = "isolation_forest"
    normalization: Literal["standardization", "zscore", "raw"] = "standardization"
    description: str | None = None  # Optional model description


class AnomalyLoadRequest(BaseModel):
    """Request to load a pre-trained anomaly model."""

    model: str  # Model identifier to load/cache as
    backend: str = "isolation_forest"


# =============================================================================
# Embeddings Types
# =============================================================================


class EmbeddingsRequest(BaseModel):
    """Request to generate embeddings."""

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"


# =============================================================================
# NLP Types
# =============================================================================


class LanguageDetectRequest(BaseModel):
    """Request to detect language of text."""

    text: str
    model: str = "papluca/xlm-roberta-base-language-detection"


class LanguageDetectBatchRequest(BaseModel):
    """Request to detect languages of multiple texts."""

    texts: list[str]
    model: str = "papluca/xlm-roberta-base-language-detection"


class KeywordExtractRequest(BaseModel):
    """Request to extract keywords from text."""

    text: str
    top_k: int = Field(default=10, ge=1, le=100)
    diversity: float = Field(default=0.5, ge=0, le=1)
    ngram_range: list[int] = [1, 3]


class KeywordExtractBatchRequest(BaseModel):
    """Request to extract keywords from multiple texts."""

    texts: list[str]
    top_k: int = Field(default=10, ge=1, le=100)
    diversity: float = Field(default=0.5, ge=0, le=1)
    ngram_range: list[int] = [1, 3]


class PIIDetectRequest(BaseModel):
    """Request to detect PII in text."""

    text: str
    entity_types: list[str] | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    use_regex: bool = True


class PIIRedactRequest(BaseModel):
    """Request to redact PII from text."""

    text: str
    entity_types: list[str] | None = None
    replacement: str = "[REDACTED]"
    replacement_map: dict[str, str] | None = None
    threshold: float = Field(default=0.5, ge=0, le=1)
    use_regex: bool = True


# =============================================================================
# Time Series Types
# =============================================================================


class TimeSeriesForecastRequest(BaseModel):
    """Request to forecast time series."""

    data: list[float]
    horizon: int = Field(default=10, ge=1, le=100)
    model: str = "amazon/chronos-t5-small"


class ChangePointRequest(BaseModel):
    """Request to detect change points in time series."""

    data: list[float]
    algorithm: Literal["binseg", "pelt", "window", "dynp"] = "binseg"
    n_changepoints: int | None = None
    penalty: float | None = None


# =============================================================================
# Vision Types
# =============================================================================


class ObjectDetectRequest(BaseModel):
    """Request to detect objects in an image."""

    image: str  # Base64-encoded image
    threshold: float = Field(default=0.5, ge=0, le=1)
    labels: list[str] | None = None
    model: str = "hustvl/yolos-tiny"


class ObjectDetectBatchRequest(BaseModel):
    """Request to detect objects in multiple images."""

    images: list[str]
    threshold: float = Field(default=0.5, ge=0, le=1)
    labels: list[str] | None = None
    model: str = "hustvl/yolos-tiny"


class BackgroundRemoveRequest(BaseModel):
    """Request to remove background from an image."""

    image: str  # Base64-encoded image
    model: str = "briaai/RMBG-1.4"


# =============================================================================
# Analysis Types
# =============================================================================


class TableQARequest(BaseModel):
    """Request to answer questions about tabular data."""

    table: list[dict[str, Any]]
    query: str
    model: str = "google/tapas-base-finetuned-wtq"


class DatasetAuditRequest(BaseModel):
    """Request to audit dataset for label quality issues."""

    labels: list[int]
    pred_probs: list[list[float]]
    label_names: list[str] | None = None


class DriftDetectRequest(BaseModel):
    """Request to detect drift between data distributions."""

    reference_data: list[float]
    current_data: list[float]
    algorithm: str = "adwin"


class AnomalyExplainRequest(BaseModel):
    """Request to explain anomaly predictions with SHAP."""

    model_id: str
    data: list[list[float]]
    feature_names: list[str] | None = None
    backend: str = "isolation_forest"
    background_samples: int = Field(default=100, ge=10, le=1000)
    nsamples: int = Field(default=100, ge=10, le=1000)


# =============================================================================
# Audio Types
# =============================================================================


class AudioTranscriptionRequest(BaseModel):
    """Request for audio transcription (speech-to-text)."""

    file: str  # Base64-encoded audio file or file ID
    model: str = "openai/whisper-base"
    language: str | None = None  # Optional language code (e.g., "en", "fr")
    prompt: str | None = None  # Optional text to guide the model
    response_format: str = "json"  # json, text, srt, vtt, verbose_json
    temperature: float = Field(default=0.0, ge=0, le=1)


class AudioTranslationRequest(BaseModel):
    """Request for audio translation (speech-to-English)."""

    file: str  # Base64-encoded audio file or file ID
    model: str = "openai/whisper-base"
    prompt: str | None = None  # Optional text to guide the model
    response_format: str = "json"  # json, text, srt, vtt, verbose_json
    temperature: float = Field(default=0.0, ge=0, le=1)


# =============================================================================
# Batch Endpoint Types
# =============================================================================


class TimeSeriesForecastBatchRequest(BaseModel):
    """Request for batch time series forecasting."""

    data: list[list[float]]  # Multiple time series to forecast
    horizon: int = Field(default=12, ge=1, le=100)
    model: str = "amazon/chronos-t5-tiny"


class ChangePointBatchRequest(BaseModel):
    """Request for batch change point detection."""

    data: list[list[float]]  # Multiple time series
    algorithm: str = "binseg"
    n_changepoints: int | None = None
    penalty: float | None = None


class TableQABatchRequest(BaseModel):
    """Request for batch table QA."""

    tables: list[list[dict[str, Any]]]  # Multiple tables
    queries: list[str]  # Query for each table
    model: str = "google/tapas-base-finetuned-wtq"
