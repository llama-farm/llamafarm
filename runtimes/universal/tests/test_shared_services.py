"""Tests for shared services: api_types, path_validator, cache_key_builder, error_handler."""

import tempfile
from pathlib import Path

import pytest


class TestTypesImports:
    """Test that all type modules import without error."""

    def test_common_types_import(self):
        """Test common types import."""
        from api_types.common import ErrorDetail, ListResponse, UsageInfo

        assert UsageInfo is not None
        assert ListResponse is not None
        assert ErrorDetail is not None

    def test_nlp_types_import(self):
        """Test NLP types import."""
        from api_types.nlp import (
            ClassifyRequest,
            ClassifyResponse,
            EmbeddingRequest,
            EmbeddingResponse,
            NERRequest,
            NERResponse,
            RerankRequest,
            RerankResponse,
        )

        assert EmbeddingRequest is not None
        assert EmbeddingResponse is not None
        assert RerankRequest is not None
        assert RerankResponse is not None
        assert ClassifyRequest is not None
        assert ClassifyResponse is not None
        assert NERRequest is not None
        assert NERResponse is not None

    def test_anomaly_types_import(self):
        """Test anomaly types import."""
        from api_types.anomaly import (
            AnomalyFitRequest,
            AnomalyFitResponse,
            AnomalyLoadRequest,
            AnomalySaveRequest,
            AnomalyScoreRequest,
            AnomalyScoreResponse,
        )

        assert AnomalyScoreRequest is not None
        assert AnomalyFitRequest is not None
        assert AnomalySaveRequest is not None
        assert AnomalyLoadRequest is not None
        assert AnomalyScoreResponse is not None
        assert AnomalyFitResponse is not None

    def test_classifier_types_import(self):
        """Test classifier types import."""
        from api_types.classifier import (
            ClassifierFitRequest,
            ClassifierFitResponse,
            ClassifierLoadRequest,
            ClassifierPredictRequest,
            ClassifierPredictResponse,
            ClassifierSaveRequest,
        )

        assert ClassifierFitRequest is not None
        assert ClassifierPredictRequest is not None
        assert ClassifierSaveRequest is not None
        assert ClassifierLoadRequest is not None
        assert ClassifierFitResponse is not None
        assert ClassifierPredictResponse is not None

    def test_vision_types_import(self):
        """Test vision types import."""
        from api_types.vision import (
            DocumentExtractRequest,
            DocumentResponse,
            OCRRequest,
            OCRResponse,
        )

        assert OCRRequest is not None
        assert OCRResponse is not None
        assert DocumentExtractRequest is not None
        assert DocumentResponse is not None

    def test_audio_types_import(self):
        """Test audio types import."""
        from api_types.audio import (
            TranscriptionRequest,
            TranscriptionResponse,
            TranslationRequest,
            TranslationResponse,
        )

        assert TranscriptionRequest is not None
        assert TranscriptionResponse is not None
        assert TranslationRequest is not None
        assert TranslationResponse is not None

    def test_all_types_from_init(self):
        """Test that __init__ exports all types."""
        from api_types import (
            AnomalyFitRequest,
            AnomalyScoreRequest,
            ClassifierFitRequest,
            ClassifyRequest,
            DocumentExtractRequest,
            EmbeddingRequest,
            NERRequest,
            OCRRequest,
            RerankRequest,
            TranscriptionResponse,
        )

        assert EmbeddingRequest is not None
        assert RerankRequest is not None
        assert ClassifyRequest is not None
        assert NERRequest is not None
        assert AnomalyScoreRequest is not None
        assert AnomalyFitRequest is not None
        assert ClassifierFitRequest is not None
        assert OCRRequest is not None
        assert DocumentExtractRequest is not None
        assert TranscriptionResponse is not None


class TestPathValidator:
    """Test path validation functions."""

    def test_sanitize_model_name_alphanumeric(self):
        """Test sanitize_model_name allows alphanumeric."""
        from services.path_validator import sanitize_model_name

        assert sanitize_model_name("model123") == "model123"
        assert sanitize_model_name("my-model") == "my-model"
        assert sanitize_model_name("my_model") == "my_model"

    def test_sanitize_model_name_removes_special(self):
        """Test sanitize_model_name removes special characters."""
        from services.path_validator import sanitize_model_name

        assert sanitize_model_name("model/name") == "modelname"
        assert sanitize_model_name("model..name") == "modelname"
        assert sanitize_model_name("model;name") == "modelname"

    def test_sanitize_filename_preserves_extension(self):
        """Test sanitize_filename preserves extension dots."""
        from services.path_validator import sanitize_filename

        assert sanitize_filename("model.joblib") == "model.joblib"
        assert sanitize_filename("model.tar.gz") == "model.tar.gz"
        assert sanitize_filename("model..name.txt") == "model..name.txt"

    def test_validate_path_within_directory_valid(self):
        """Test validate_path_within_directory allows valid paths."""
        from services.path_validator import validate_path_within_directory

        with tempfile.TemporaryDirectory() as tmpdir:
            safe_dir = Path(tmpdir)
            valid_path = safe_dir / "subdir" / "model.txt"
            result = validate_path_within_directory(valid_path, safe_dir)
            assert result.is_relative_to(safe_dir.resolve())

    def test_validate_path_within_directory_rejects_traversal(self):
        """Test validate_path_within_directory rejects path traversal."""
        from services.path_validator import (
            PathValidationError,
            validate_path_within_directory,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            safe_dir = Path(tmpdir)
            # Try to escape with ..
            malicious_path = safe_dir / ".." / "etc" / "passwd"
            with pytest.raises(PathValidationError) as exc:
                validate_path_within_directory(malicious_path, safe_dir)
            assert "outside allowed directory" in str(exc.value)

    def test_validate_model_path_rejects_dotdot(self):
        """Test validate_model_path rejects .. in path."""
        from services.path_validator import PathValidationError, validate_model_path

        malicious = Path("../../../etc/passwd")
        with pytest.raises(PathValidationError) as exc:
            validate_model_path(malicious, "anomaly")
        assert "contains '..'" in str(exc.value)

    def test_is_valid_model_name(self):
        """Test is_valid_model_name validation."""
        from services.path_validator import is_valid_model_name

        assert is_valid_model_name("my-model") is True
        assert is_valid_model_name("model_123") is True
        assert is_valid_model_name("model/path") is False
        assert is_valid_model_name("../escape") is False
        assert is_valid_model_name("model\\windows") is False

    def test_get_model_path_anomaly(self):
        """Test get_model_path for anomaly models."""
        from services.path_validator import ANOMALY_MODELS_DIR, get_model_path

        path = get_model_path("test-model", "isolation_forest", "anomaly")
        assert path.parent == ANOMALY_MODELS_DIR
        assert "test-model" in str(path)
        assert "isolation_forest" in str(path)

    def test_get_model_path_classifier(self):
        """Test get_model_path for classifier models."""
        from services.path_validator import CLASSIFIER_MODELS_DIR, get_model_path

        path = get_model_path("test-model", "setfit", "classifier")
        assert path.parent == CLASSIFIER_MODELS_DIR
        assert "test-model" in str(path)


class TestCacheKeyBuilder:
    """Test cache key builder functions."""

    def test_cache_key_deterministic(self):
        """Test cache keys are deterministic."""
        from services.cache_key_builder import CacheKeyBuilder

        key1 = CacheKeyBuilder.anomaly_model("test", "isolation_forest", "zscore")
        key2 = CacheKeyBuilder.anomaly_model("test", "isolation_forest", "zscore")
        assert key1 == key2

    def test_cache_key_unique_different_params(self):
        """Test cache keys are unique for different params."""
        from services.cache_key_builder import CacheKeyBuilder

        key1 = CacheKeyBuilder.anomaly_model("test", "isolation_forest", "zscore")
        key2 = CacheKeyBuilder.anomaly_model(
            "test", "isolation_forest", "standardization"
        )
        assert key1 != key2

    def test_encoder_cache_key_includes_task(self):
        """Test encoder cache key includes task."""
        from services.cache_key_builder import CacheKeyBuilder

        key_embed = CacheKeyBuilder.encoder_model("model", task="embedding")
        key_rerank = CacheKeyBuilder.encoder_model("model", task="reranking")
        assert "embedding" in key_embed
        assert "reranking" in key_rerank
        assert key_embed != key_rerank

    def test_ocr_cache_key_sorts_languages(self):
        """Test OCR cache key sorts languages consistently."""
        from services.cache_key_builder import CacheKeyBuilder

        key1 = CacheKeyBuilder.ocr_model("surya", ["fr", "en", "de"])
        key2 = CacheKeyBuilder.ocr_model("surya", ["de", "en", "fr"])
        assert key1 == key2  # Should be same after sorting

    def test_backward_compatible_functions(self):
        """Test backward compatible make_*_cache_key functions."""
        from services.cache_key_builder import (
            make_anomaly_cache_key,
            make_classifier_cache_key,
            make_ocr_cache_key,
        )

        assert "anomaly" in make_anomaly_cache_key("test", "isolation_forest")
        assert "classifier" in make_classifier_cache_key("test")
        assert "ocr" in make_ocr_cache_key("surya", ["en"])


class TestErrorHandler:
    """Test error handler decorator and utilities."""

    def test_custom_errors_have_status_code(self):
        """Test custom errors have appropriate status codes."""
        from services.error_handler import (
            BackendNotInstalledError,
            ModelNotFittedError,
            ModelNotFoundError,
            ValidationError,
        )

        assert ModelNotFoundError("model", "classifier").status_code == 404
        assert ModelNotFittedError("model").status_code == 400
        assert ValidationError("bad input").status_code == 400
        assert BackendNotInstalledError("cuda").status_code == 400

    def test_format_error_response(self):
        """Test format_error_response structure."""
        from services.error_handler import format_error_response

        response = format_error_response(
            "Something went wrong", code="ERR_001", details={"field": "value"}
        )
        assert response["error"]["message"] == "Something went wrong"
        assert response["error"]["code"] == "ERR_001"
        assert response["error"]["details"]["field"] == "value"

    @pytest.mark.asyncio
    async def test_handle_endpoint_errors_decorator(self):
        """Test handle_endpoint_errors decorator catches exceptions."""
        from fastapi import HTTPException

        from services.error_handler import (
            UniversalRuntimeError,
            handle_endpoint_errors,
        )

        @handle_endpoint_errors("test_endpoint")
        async def test_func():
            raise UniversalRuntimeError("Test error", status_code=418)

        with pytest.raises(HTTPException) as exc:
            await test_func()
        assert exc.value.status_code == 418
        assert "Test error" in exc.value.detail

    @pytest.mark.asyncio
    async def test_handle_endpoint_errors_passes_http_exception(self):
        """Test that HTTPException passes through unchanged."""
        from fastapi import HTTPException

        from services.error_handler import handle_endpoint_errors

        @handle_endpoint_errors("test_endpoint")
        async def test_func():
            raise HTTPException(status_code=403, detail="Forbidden")

        with pytest.raises(HTTPException) as exc:
            await test_func()
        assert exc.value.status_code == 403
        assert exc.value.detail == "Forbidden"


class TestServicesInit:
    """Test services __init__ exports."""

    def test_all_path_validator_exports(self):
        """Test path_validator exports from services."""
        from services import (
            ANOMALY_MODELS_DIR,
            CLASSIFIER_MODELS_DIR,
            PathValidationError,
            get_model_path,
            is_valid_model_name,
            sanitize_filename,
            sanitize_model_name,
            validate_model_path,
            validate_path_within_directory,
        )

        assert ANOMALY_MODELS_DIR is not None
        assert CLASSIFIER_MODELS_DIR is not None
        assert PathValidationError is not None
        assert sanitize_model_name is not None
        assert sanitize_filename is not None
        assert validate_path_within_directory is not None
        assert validate_model_path is not None
        assert get_model_path is not None
        assert is_valid_model_name is not None

    def test_all_cache_key_exports(self):
        """Test cache_key_builder exports from services."""
        from services import (
            CacheKeyBuilder,
            make_anomaly_cache_key,
            make_classifier_cache_key,
            make_encoder_cache_key,
            make_ocr_cache_key,
        )

        assert CacheKeyBuilder is not None
        assert make_anomaly_cache_key is not None
        assert make_classifier_cache_key is not None
        assert make_encoder_cache_key is not None
        assert make_ocr_cache_key is not None

    def test_all_error_handler_exports(self):
        """Test error_handler exports from services."""
        from services import (
            BackendNotInstalledError,
            ModelNotFittedError,
            ModelNotFoundError,
            UniversalRuntimeError,
            ValidationError,
            format_error_response,
            handle_endpoint_errors,
        )

        assert UniversalRuntimeError is not None
        assert ModelNotFoundError is not None
        assert ModelNotFittedError is not None
        assert ValidationError is not None
        assert BackendNotInstalledError is not None
        assert handle_endpoint_errors is not None
        assert format_error_response is not None
