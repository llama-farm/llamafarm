"""
Tests for AudioModel (speech-to-text).
"""

import pytest
import numpy as np
import io
import wave
import base64
from models.audio_model import AudioModel


def create_test_audio_wav(duration=1.0, sample_rate=16000):
    """Create a test WAV file with silence."""
    # Generate silence
    audio_data = np.zeros(int(sample_rate * duration), dtype=np.int16)

    # Write to WAV format in memory
    wav_io = io.BytesIO()
    with wave.open(wav_io, "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

    wav_io.seek(0)
    return wav_io.read()


@pytest.fixture
def test_audio_wav():
    """Create test audio in WAV format."""
    return create_test_audio_wav(duration=0.5)  # 0.5 seconds for speed


@pytest.fixture
def test_audio_wav_base64(test_audio_wav):
    """Test audio as base64."""
    return base64.b64encode(test_audio_wav).decode("utf-8")


@pytest.mark.asyncio
@pytest.mark.slow  # Whisper download is large
async def test_audio_load(device, test_model_ids):
    """Test loading an audio model."""
    model = AudioModel(test_model_ids["audio"], device, task="transcribe")
    await model.load()

    assert model.pipe is not None
    assert model.model_type == "audio_transcribe"


@pytest.mark.asyncio
@pytest.mark.slow
async def test_audio_transcribe(device, test_model_ids, test_audio_wav):
    """Test audio transcription."""
    model = AudioModel(test_model_ids["audio"], device, task="transcribe")
    await model.load()

    result = await model.transcribe(audio=test_audio_wav, language="en")

    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)
    # Silence typically transcribes to empty or noise
    # Just check it doesn't crash


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skip(reason="Skipping base64 test for now due to speed issues")
async def test_audio_transcribe_from_base64(
    device, test_model_ids, test_audio_wav_base64
):
    """Test transcription from base64 audio."""
    model = AudioModel(test_model_ids["audio"], device, task="transcribe")
    await model.load()

    result = await model.transcribe(audio=test_audio_wav_base64)

    assert "text" in result
    assert isinstance(result["text"], str)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_audio_transcribe_with_timestamps(device, test_model_ids, test_audio_wav):
    """Test transcription with word timestamps."""
    model = AudioModel(test_model_ids["audio"], device, task="transcribe")
    await model.load()

    result = await model.transcribe(audio=test_audio_wav, return_timestamps=True)

    assert "text" in result
    # Timestamps might not be present for silence, but check it doesn't crash


@pytest.mark.asyncio
@pytest.mark.slow
@pytest.mark.skip(reason="Skipping temperature test for now due to speed issues")
async def test_audio_transcribe_with_temperature(
    device, test_model_ids, test_audio_wav
):
    """Test transcription with different temperatures."""
    model = AudioModel(test_model_ids["audio"], device, task="transcribe")
    await model.load()

    result = await model.transcribe(audio=test_audio_wav, temperature=0.0)

    assert "text" in result


@pytest.mark.asyncio
@pytest.mark.slow
async def test_audio_translate(device, test_model_ids, test_audio_wav):
    """Test audio translation (Whisper can translate to English)."""
    model = AudioModel(test_model_ids["audio"], device, task="translate")
    await model.load()

    result = await model.translate(audio=test_audio_wav)

    assert isinstance(result, dict)
    assert "text" in result
    assert isinstance(result["text"], str)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_audio_generate_alias(device, test_model_ids, test_audio_wav):
    """Test that generate() is an alias for transcribe()."""
    model = AudioModel(test_model_ids["audio"], device, task="transcribe")
    await model.load()

    result = await model.generate(audio=test_audio_wav)

    assert "text" in result


@pytest.mark.asyncio
@pytest.mark.slow
async def test_audio_model_info(device, test_model_ids):
    """Test getting model info."""
    model = AudioModel(test_model_ids["audio"], device, task="transcribe")
    await model.load()

    info = model.get_model_info()

    assert info["model_id"] == test_model_ids["audio"]
    assert info["model_type"] == "audio_transcribe"
    assert info["device"] == device
