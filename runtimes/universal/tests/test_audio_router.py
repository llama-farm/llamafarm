"""Tests for Audio router endpoints (speech-to-text transcription and translation)."""

import io
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def mock_speech_model():
    """Create a mock speech model."""
    model = MagicMock()

    # Mock transcription result
    class MockSegment:
        def __init__(self, idx, text):
            self.id = idx
            self.start = idx * 2.0
            self.end = (idx + 1) * 2.0
            self.text = text
            self.words = None
            self.avg_logprob = -0.5
            self.no_speech_prob = 0.1

    class MockTranscriptionResult:
        def __init__(self, text, language="en"):
            self.text = text
            self.language = language
            self.duration = 10.5
            self.segments = [
                MockSegment(0, "Hello world."),
                MockSegment(1, "How are you?"),
            ]

    async def mock_transcribe(
        audio_path,
        language=None,
        word_timestamps=False,
        initial_prompt=None,
        temperature=None,
        task="transcribe",
    ):
        return MockTranscriptionResult("Hello world. How are you?", language or "en")

    model.transcribe = mock_transcribe

    async def mock_transcribe_stream(
        audio_path, language=None, word_timestamps=False, initial_prompt=None
    ):
        for seg in MockTranscriptionResult("Hello world. How are you?").segments:
            yield seg

    model.transcribe_stream = mock_transcribe_stream

    return model


@pytest.fixture
def test_app(mock_speech_model):
    """Create a test FastAPI app with the audio router."""
    from routers.audio import router, set_speech_loader

    app = FastAPI()
    app.include_router(router)

    # Set up mock model loader
    async def mock_load_speech(model_id="distil-large-v3", compute_type=None):
        return mock_speech_model

    set_speech_loader(mock_load_speech)

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestTranscriptionEndpoint:
    """Test /v1/audio/transcriptions endpoint."""

    def test_transcription_with_file(self, client):
        """Test POST /v1/audio/transcriptions with audio file."""
        # Create a fake WAV file
        audio_content = b"RIFF" + b"\x00" * 100  # Minimal WAV header

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "distil-large-v3"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data

    def test_transcription_requires_file(self, client):
        """Test POST /v1/audio/transcriptions requires file."""
        response = client.post(
            "/v1/audio/transcriptions",
            data={"model": "distil-large-v3"},
        )

        assert response.status_code == 400
        assert "Audio file is required" in response.json()["detail"]

    def test_transcription_text_format(self, client):
        """Test transcription with text response format."""
        audio_content = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "distil-large-v3", "response_format": "text"},
        )

        assert response.status_code == 200
        # Text format returns plain string
        assert isinstance(response.text, str)

    def test_transcription_verbose_json(self, client):
        """Test transcription with verbose_json response format."""
        audio_content = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "distil-large-v3", "response_format": "verbose_json"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "segments" in data
        assert "duration" in data


class TestTranslationEndpoint:
    """Test /v1/audio/translations endpoint."""

    def test_translation_with_file(self, client):
        """Test POST /v1/audio/translations with audio file."""
        audio_content = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "distil-large-v3"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "text" in data

    def test_translation_text_format(self, client):
        """Test translation with text response format."""
        audio_content = b"RIFF" + b"\x00" * 100

        response = client.post(
            "/v1/audio/translations",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "distil-large-v3", "response_format": "text"},
        )

        assert response.status_code == 200
        assert isinstance(response.text, str)


class TestRouterInitialization:
    """Test router initialization and dependency injection."""

    def test_speech_loader_not_set_raises_error(self):
        """Test that calling endpoint without setting loader raises error."""
        from routers.audio import router, set_speech_loader

        # Reset the loader
        set_speech_loader(None)

        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)

        audio_content = b"RIFF" + b"\x00" * 100
        response = client.post(
            "/v1/audio/transcriptions",
            files={"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")},
            data={"model": "distil-large-v3"},
        )

        assert response.status_code == 500
        assert "not initialized" in response.json()["detail"].lower()
