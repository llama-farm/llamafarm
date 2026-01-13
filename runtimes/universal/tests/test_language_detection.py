"""Tests for Language Detection Model.

Phase 6 of Universal Runtime ML Enhancements.
"""

import pytest

from models.language_detection_model import LanguageDetectionModel, LANGUAGE_NAMES


class TestLanguageDetectionModelLoading:
    """Test language detection model loading."""

    @pytest.mark.asyncio
    async def test_model_loads_successfully(self):
        """Test that language detection model loads."""
        model = LanguageDetectionModel(model_id="test-lang", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()

    @pytest.mark.asyncio
    async def test_model_unload(self):
        """Test that model can be unloaded."""
        model = LanguageDetectionModel(model_id="test-lang", device="cpu")
        await model.load()
        assert model.is_loaded
        await model.unload()
        assert not model.is_loaded


class TestLanguageDetection:
    """Test language detection functionality."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = LanguageDetectionModel(model_id="test-lang", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_detect_english(self, model):
        """Test detecting English text."""
        result = await model.detect("Hello, how are you today?")
        assert result["language"] == "en"
        assert result["language_name"] == "English"
        assert result["confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_detect_spanish(self, model):
        """Test detecting Spanish text."""
        result = await model.detect("Hola, ¿cómo estás hoy?")
        assert result["language"] == "es"
        assert result["language_name"] == "Spanish"
        assert result["confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_detect_french(self, model):
        """Test detecting French text."""
        result = await model.detect("Bonjour, comment allez-vous aujourd'hui?")
        assert result["language"] == "fr"
        assert result["language_name"] == "French"
        assert result["confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_detect_german(self, model):
        """Test detecting German text."""
        result = await model.detect("Guten Tag, wie geht es Ihnen heute?")
        assert result["language"] == "de"
        assert result["language_name"] == "German"
        assert result["confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_detect_chinese(self, model):
        """Test detecting Chinese text."""
        result = await model.detect("你好，今天你好吗？")
        assert result["language"] == "zh"
        assert result["language_name"] == "Chinese"
        assert result["confidence"] > 0.9

    @pytest.mark.asyncio
    async def test_detect_japanese(self, model):
        """Test detecting Japanese text."""
        result = await model.detect("こんにちは、今日はお元気ですか？")
        assert result["language"] == "ja"
        assert result["language_name"] == "Japanese"
        assert result["confidence"] > 0.9


class TestConfidenceScores:
    """Test confidence score handling."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = LanguageDetectionModel(model_id="test-lang", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_returns_all_scores(self, model):
        """Test that all_scores contains multiple languages."""
        result = await model.detect("Hello world", top_k=5)
        assert "all_scores" in result
        assert len(result["all_scores"]) == 5

    @pytest.mark.asyncio
    async def test_scores_sum_close_to_one(self, model):
        """Test that top-k scores are reasonable (don't need to sum to 1 if top_k < total)."""
        result = await model.detect("Hello world", top_k=20)
        total = sum(result["all_scores"].values())
        # With all languages, should be close to 1
        assert 0.95 <= total <= 1.05


class TestBatchProcessing:
    """Test batch language detection."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = LanguageDetectionModel(model_id="test-lang", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_batch_detection(self, model):
        """Test batch language detection."""
        texts = [
            "Hello world",
            "Bonjour le monde",
            "Hallo Welt",
        ]
        results = await model.detect_batch(texts)

        assert len(results) == 3
        assert results[0]["language"] == "en"
        assert results[1]["language"] == "fr"
        assert results[2]["language"] == "de"

    @pytest.mark.asyncio
    async def test_batch_with_mixed_languages(self, model):
        """Test batch with many different languages."""
        texts = [
            "Hello world",  # English
            "Hola mundo",  # Spanish
            "Bonjour le monde",  # French
            "Ciao mondo",  # Italian
            "Olá mundo",  # Portuguese
        ]
        results = await model.detect_batch(texts)

        assert len(results) == 5
        # Just verify we get results for all
        for result in results:
            assert "language" in result
            assert "confidence" in result
            assert result["confidence"] > 0.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    async def model(self):
        """Create and load a model for testing."""
        model = LanguageDetectionModel(model_id="test-lang", device="cpu")
        await model.load()
        yield model
        await model.unload()

    @pytest.mark.asyncio
    async def test_short_text(self, model):
        """Test with very short text."""
        result = await model.detect("Hi")
        assert "language" in result
        # Short text may have lower confidence but should still work

    @pytest.mark.asyncio
    async def test_long_text_truncation(self, model):
        """Test that long text is truncated properly."""
        long_text = "Hello world. " * 1000  # Very long text
        result = await model.detect(long_text)
        assert result["language"] == "en"
        # Truncated text may have slightly lower confidence
        assert result["confidence"] > 0.7

    @pytest.mark.asyncio
    async def test_mixed_language_text(self, model):
        """Test text with mixed languages."""
        # Text with English and Spanish mixed
        result = await model.detect("Hello hola world mundo")
        # Should detect something, even if confidence is lower
        assert "language" in result


class TestModelInfo:
    """Test model information retrieval."""

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test getting model info."""
        model = LanguageDetectionModel(model_id="test-lang", device="cpu")
        await model.load()

        info = model.get_model_info()
        assert "supported_languages" in info
        assert "en" in info["supported_languages"]
        assert "hf_model_name" in info
        assert info["is_loaded"] is True

        await model.unload()


class TestLanguageNames:
    """Test language name mapping."""

    def test_all_language_codes_have_names(self):
        """Test that all supported language codes have names."""
        expected_languages = [
            "ar", "bg", "de", "el", "en", "es", "fr", "hi", "it",
            "ja", "nl", "pl", "pt", "ru", "sw", "th", "tr", "ur", "vi", "zh"
        ]
        for lang in expected_languages:
            assert lang in LANGUAGE_NAMES
            assert isinstance(LANGUAGE_NAMES[lang], str)
            assert len(LANGUAGE_NAMES[lang]) > 0
