"""Universal Runtime shared services.

This module exports all shared services for the Universal Runtime.
"""

from .cache_key_builder import (
    CacheKeyBuilder,
    make_anomaly_cache_key,
    make_classifier_cache_key,
    make_document_cache_key,
    make_encoder_cache_key,
    make_language_cache_key,
    make_ocr_cache_key,
    make_speech_cache_key,
)
from .error_handler import (
    BackendNotInstalledError,
    ModelNotFittedError,
    ModelNotFoundError,
    UniversalRuntimeError,
    ValidationError,
    format_error_response,
    handle_endpoint_errors,
)
from .model_loader import (
    ModelLoader,
    ModelLoaderRegistry,
    get_model_registry,
    reset_model_registry,
)
from .path_validator import (
    ANOMALY_MODELS_DIR,
    CLASSIFIER_MODELS_DIR,
    MODELS_BASE_DIR,
    PathValidationError,
    ensure_model_directories,
    get_model_path,
    is_valid_model_name,
    sanitize_filename,
    sanitize_model_name,
    validate_model_path,
    validate_path_within_directory,
)
from .training_executor import (
    TrainingContext,
    get_training_executor,
    run_in_executor,
    run_training_task,
    shutdown_training_executor,
)

__all__ = [
    # Path validation
    "MODELS_BASE_DIR",
    "ANOMALY_MODELS_DIR",
    "CLASSIFIER_MODELS_DIR",
    "PathValidationError",
    "sanitize_model_name",
    "sanitize_filename",
    "validate_path_within_directory",
    "validate_model_path",
    "get_model_path",
    "ensure_model_directories",
    "is_valid_model_name",
    # Cache key builder
    "CacheKeyBuilder",
    "make_language_cache_key",
    "make_encoder_cache_key",
    "make_document_cache_key",
    "make_ocr_cache_key",
    "make_anomaly_cache_key",
    "make_classifier_cache_key",
    "make_speech_cache_key",
    # Error handling
    "UniversalRuntimeError",
    "ModelNotFoundError",
    "ModelNotFittedError",
    "ValidationError",
    "BackendNotInstalledError",
    "handle_endpoint_errors",
    "format_error_response",
    # Model loader
    "ModelLoader",
    "ModelLoaderRegistry",
    "get_model_registry",
    "reset_model_registry",
    # Training executor
    "get_training_executor",
    "shutdown_training_executor",
    "run_in_executor",
    "run_training_task",
    "TrainingContext",
]
