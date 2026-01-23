"""
Tests for SpeechModel (speech-to-text with Whisper).

Tests the SpeechModel class and audio utility functions.
For integration tests with real audio files, install the speech dependencies
and run tests with: pytest tests/test_speech_model.py -v
"""

import pytest

# Check if we can import the speech model (requires torch)
try:
    from models.speech_model import SpeechModel

    SPEECH_MODEL_AVAILABLE = True
except ImportError:
    SPEECH_MODEL_AVAILABLE = False

# Skip marker for tests requiring speech model
requires_speech_model = pytest.mark.skipif(
    not SPEECH_MODEL_AVAILABLE,
    reason="SpeechModel not available (requires torch and faster-whisper)",
)


@requires_speech_model
class TestSpeechModel:
    """Tests for SpeechModel class.

    Note: Tests that require faster-whisper to be installed are marked
    with @pytest.mark.integration and skipped by default.
    """

    @pytest.mark.asyncio
    async def test_model_initialization(self):
        """Test SpeechModel initialization."""
        model = SpeechModel("distil-large-v3", "cpu")

        assert model.model_id == "distil-large-v3"
        assert model.device == "cpu"
        assert model.model_type == "speech"
        assert model.supports_streaming is True

    @pytest.mark.asyncio
    async def test_model_info(self):
        """Test getting model info."""
        model = SpeechModel("large-v3", "cuda", compute_type="float16")

        info = model.get_model_info()

        assert info["model_id"] == "large-v3"
        assert info["model_type"] == "speech"
        assert info["supports_streaming"] is True

    @pytest.mark.asyncio
    async def test_model_size_resolution(self):
        """Test that model size aliases are resolved."""
        model = SpeechModel("large", "cpu")
        assert model._resolved_model == "large-v3"

        model = SpeechModel("distil-large-v3", "cpu")
        assert model._resolved_model == "distil-large-v3"

    @pytest.mark.asyncio
    async def test_compute_type_auto_selection(self):
        """Test automatic compute type selection."""
        # CUDA should use float16
        model_cuda = SpeechModel("small", "cuda")
        assert model_cuda.compute_type == "float16"

        # CPU should use int8
        model_cpu = SpeechModel("small", "cpu")
        assert model_cpu.compute_type == "int8"

    @pytest.mark.asyncio
    async def test_error_on_unloaded_model(self):
        """Test error when transcribing without loading."""
        model = SpeechModel("small", "cpu")

        with pytest.raises(RuntimeError, match="Model not loaded"):
            await model.transcribe("test.wav")


class TestAudioBuffer:
    """Tests for AudioBuffer utility."""

    def test_buffer_initialization(self):
        """Test AudioBuffer initialization."""
        from utils.audio_buffer import AudioBuffer

        buffer = AudioBuffer()

        assert buffer.sample_rate == 16000
        assert buffer.sample_width == 2
        assert buffer.channels == 1
        assert buffer.is_empty

    def test_buffer_add_audio(self):
        """Test adding audio to buffer."""
        from utils.audio_buffer import AudioBuffer

        buffer = AudioBuffer()

        # Add 1 second of audio (16000 samples * 2 bytes)
        audio_data = bytes(16000 * 2)
        buffer.add(audio_data)

        assert not buffer.is_empty
        assert abs(buffer.duration - 1.0) < 0.01

    def test_buffer_has_enough_audio(self):
        """Test has_enough_audio check."""
        from utils.audio_buffer import AudioBuffer

        buffer = AudioBuffer(min_duration=0.5)

        # Add 0.3 seconds
        buffer.add(bytes(int(16000 * 0.3 * 2)))
        assert not buffer.has_enough_audio

        # Add more to reach 0.6 seconds
        buffer.add(bytes(int(16000 * 0.3 * 2)))
        assert buffer.has_enough_audio

    def test_buffer_is_full(self):
        """Test is_full check."""
        from utils.audio_buffer import AudioBuffer

        buffer = AudioBuffer(max_duration=1.0)

        # Add 0.5 seconds
        buffer.add(bytes(int(16000 * 0.5 * 2)))
        assert not buffer.is_full

        # Add more to exceed 1 second
        buffer.add(bytes(int(16000 * 0.6 * 2)))
        assert buffer.is_full

    def test_buffer_get_wav_bytes(self):
        """Test WAV output."""
        from utils.audio_buffer import AudioBuffer

        buffer = AudioBuffer()
        buffer.add(bytes(16000 * 2))  # 1 second

        wav_bytes = buffer.get_wav_bytes()

        # Check WAV header
        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_buffer_clear(self):
        """Test clearing buffer."""
        from utils.audio_buffer import AudioBuffer

        buffer = AudioBuffer()
        buffer.add(bytes(16000 * 2))

        buffer.clear()

        assert buffer.is_empty
        assert buffer.duration == 0.0

    def test_buffer_pop_audio(self):
        """Test popping audio from buffer."""
        from utils.audio_buffer import AudioBuffer

        buffer = AudioBuffer()
        buffer.add(bytes(16000 * 2))

        wav_bytes = buffer.pop_audio()

        assert wav_bytes[:4] == b"RIFF"
        assert buffer.is_empty


class TestStreamingAudioBuffer:
    """Tests for StreamingAudioBuffer with VAD."""

    def test_streaming_buffer_initialization(self):
        """Test StreamingAudioBuffer initialization."""
        from utils.audio_buffer import StreamingAudioBuffer

        buffer = StreamingAudioBuffer(use_vad=False)

        assert buffer.min_speech_duration == 0.5
        assert buffer.max_speech_duration == 30.0

    def test_streaming_buffer_add_without_vad(self):
        """Test adding audio without VAD."""
        from utils.audio_buffer import StreamingAudioBuffer

        buffer = StreamingAudioBuffer(
            min_speech_duration=0.5,
            max_speech_duration=1.0,
            use_vad=False,
        )

        # Add audio that doesn't trigger transcription
        should_transcribe, audio = buffer.add(bytes(int(16000 * 0.3 * 2)))
        assert not should_transcribe
        assert audio is None

        # Add more to exceed max_duration
        should_transcribe, audio = buffer.add(bytes(int(16000 * 0.8 * 2)))
        assert should_transcribe
        assert audio is not None

    def test_streaming_buffer_flush(self):
        """Test flushing buffer."""
        from utils.audio_buffer import StreamingAudioBuffer

        buffer = StreamingAudioBuffer(use_vad=False)

        # Add some audio
        buffer.add(bytes(int(16000 * 0.3 * 2)))

        # Flush should return audio
        audio = buffer.flush()
        assert audio is not None
        assert audio[:4] == b"RIFF"

    def test_streaming_buffer_clear(self):
        """Test clearing buffer."""
        from utils.audio_buffer import StreamingAudioBuffer

        buffer = StreamingAudioBuffer(use_vad=False)
        buffer.add(bytes(int(16000 * 0.3 * 2)))

        buffer.clear()

        assert buffer.flush() is None

    def test_streaming_buffer_chunk_interval(self):
        """Test time-based chunking mode."""
        from utils.audio_buffer import StreamingAudioBuffer

        # Create buffer with 1 second chunk interval
        buffer = StreamingAudioBuffer(chunk_interval=1.0)

        # Add 0.5 seconds - shouldn't trigger yet
        should_transcribe, audio = buffer.add(bytes(int(16000 * 0.5 * 2)))
        assert not should_transcribe
        assert audio is None

        # Add another 0.3 seconds (total 0.8s) - still not enough
        should_transcribe, audio = buffer.add(bytes(int(16000 * 0.3 * 2)))
        assert not should_transcribe
        assert audio is None

        # Add 0.3 more seconds (total 1.1s) - should trigger
        should_transcribe, audio = buffer.add(bytes(int(16000 * 0.3 * 2)))
        assert should_transcribe
        assert audio is not None
        assert audio[:4] == b"RIFF"

        # Buffer should be empty now, next add shouldn't trigger
        should_transcribe, audio = buffer.add(bytes(int(16000 * 0.3 * 2)))
        assert not should_transcribe
        assert audio is None

    def test_streaming_buffer_chunk_interval_multiple_triggers(self):
        """Test that chunk interval triggers multiple times."""
        from utils.audio_buffer import StreamingAudioBuffer

        buffer = StreamingAudioBuffer(chunk_interval=0.5)
        triggers = 0

        # Add 2 seconds of audio in small chunks
        for _ in range(20):  # 20 chunks of 0.1s each = 2 seconds
            should_transcribe, audio = buffer.add(bytes(int(16000 * 0.1 * 2)))
            if should_transcribe:
                triggers += 1
                assert audio is not None

        # Should have triggered ~4 times (2 seconds / 0.5 second interval)
        assert triggers >= 3  # Allow some margin for floating point


class TestAudioConversion:
    """Tests for audio conversion utilities."""

    def test_pcm_to_wav(self):
        """Test PCM to WAV conversion."""
        from utils.audio_buffer import pcm_to_wav

        # Create some PCM data
        pcm_data = bytes(16000 * 2)  # 1 second of silence

        wav_bytes = pcm_to_wav(pcm_data)

        assert wav_bytes[:4] == b"RIFF"
        assert wav_bytes[8:12] == b"WAVE"

    def test_float32_to_int16(self):
        """Test float32 to int16 conversion."""
        import struct

        from utils.audio_buffer import float32_to_int16

        # Create float32 samples
        floats = [0.0, 0.5, -0.5, 1.0, -1.0]
        float_bytes = struct.pack(f"{len(floats)}f", *floats)

        int16_bytes = float32_to_int16(float_bytes)

        # Unpack and verify
        int16_samples = struct.unpack(f"{len(floats)}h", int16_bytes)

        assert int16_samples[0] == 0  # 0.0 -> 0
        assert int16_samples[1] == 16383  # 0.5 -> ~16383
        assert int16_samples[2] == -16383  # -0.5 -> ~-16383

    def test_strip_wav_header(self):
        """Test WAV header stripping."""
        from utils.audio_buffer import pcm_to_wav, strip_wav_header

        # Create some PCM data
        pcm_data = bytes(range(256)) * 100  # Some non-zero PCM data

        # Convert to WAV
        wav_bytes = pcm_to_wav(pcm_data)

        # Strip the WAV header
        stripped = strip_wav_header(wav_bytes)

        # Should get back the original PCM data
        assert stripped == pcm_data

    def test_strip_wav_header_raw_pcm(self):
        """Test that raw PCM passes through unchanged."""
        from utils.audio_buffer import strip_wav_header

        # Raw PCM data (doesn't start with RIFF)
        pcm_data = bytes(range(256)) * 100

        # Should pass through unchanged
        result = strip_wav_header(pcm_data)
        assert result == pcm_data


class TestTimestampFormatting:
    """Tests for timestamp formatting utilities from audio router."""

    def test_format_timestamp_srt(self):
        """Test SRT timestamp formatting using production function."""
        from routers.audio.router import _format_timestamp_srt

        assert _format_timestamp_srt(0.0) == "00:00:00,000"
        assert _format_timestamp_srt(1.5) == "00:00:01,500"
        assert _format_timestamp_srt(65.123) == "00:01:05,123"
        # Use exact decimal to avoid floating point precision issues
        assert _format_timestamp_srt(3662.0) == "01:01:02,000"

    def test_format_timestamp_vtt(self):
        """Test VTT timestamp formatting using production function."""
        from routers.audio.router import _format_timestamp_vtt

        assert _format_timestamp_vtt(0.0) == "00:00:00.000"
        assert _format_timestamp_vtt(1.5) == "00:00:01.500"
        assert _format_timestamp_vtt(65.123) == "00:01:05.123"


class TestSilenceDetection:
    """Tests for silence detection utilities."""

    def test_silence_detection_with_zeros(self):
        """Test that all-zero audio is detected as silence."""
        from utils.audio_buffer import is_silence

        # All zeros = complete silence
        silent_audio = bytes(16000 * 2)  # 1 second of silence
        assert is_silence(silent_audio, threshold=0.01) is True

    def test_silence_detection_with_audio(self):
        """Test that actual audio is not detected as silence."""
        import struct

        from utils.audio_buffer import is_silence

        # Create audio with actual samples (sine wave approximation)
        samples = []
        for i in range(16000):
            # Generate values that vary significantly
            value = int(10000 * ((i % 100) / 50 - 1))  # Values between -10000 and 10000
            samples.append(value)
        audio_data = struct.pack(f"{len(samples)}h", *samples)

        assert is_silence(audio_data, threshold=0.01) is False

    def test_silence_detection_with_low_noise(self):
        """Test that very low amplitude noise is detected as silence."""
        import struct

        from utils.audio_buffer import is_silence

        # Very low amplitude values (noise floor)
        samples = [i % 10 - 5 for i in range(16000)]  # Values between -5 and 5
        audio_data = struct.pack(f"{len(samples)}h", *samples)

        assert is_silence(audio_data, threshold=0.01) is True

    def test_calculate_audio_energy(self):
        """Test RMS energy calculation."""
        import struct

        from utils.audio_buffer import calculate_audio_energy

        # Silence should have zero energy
        silent = bytes(1000)
        assert calculate_audio_energy(silent) == 0.0

        # Full-scale sine should have high energy
        # Samples at max amplitude
        max_samples = [32767] * 500 + [-32768] * 500
        loud_audio = struct.pack(f"{len(max_samples)}h", *max_samples)
        energy = calculate_audio_energy(loud_audio)
        assert energy > 0.9  # Should be close to 1.0


class TestAudioFormatDetection:
    """Tests for audio format detection utilities."""

    def test_detect_wav_format(self):
        """Test detection of WAV format."""
        from utils.audio_buffer import detect_audio_format, pcm_to_wav

        # Create a WAV file
        pcm_data = bytes(16000 * 2)  # 1 second of silence
        wav_data = pcm_to_wav(pcm_data)

        format_name, is_compressed = detect_audio_format(wav_data)
        assert format_name == "WAV"
        assert is_compressed is False

    def test_detect_raw_pcm(self):
        """Test detection of raw PCM (assumed)."""
        from utils.audio_buffer import detect_audio_format

        # Raw PCM data (doesn't match any known signature)
        pcm_data = bytes(range(256)) * 100

        format_name, is_compressed = detect_audio_format(pcm_data)
        assert format_name == "PCM (assumed)"
        assert is_compressed is False

    def test_detect_ogg_format(self):
        """Test detection of Ogg format."""
        from utils.audio_buffer import detect_audio_format

        # Ogg signature
        ogg_data = b"OggS" + bytes(100)

        format_name, is_compressed = detect_audio_format(ogg_data)
        assert format_name == "Ogg (Opus/Vorbis)"
        assert is_compressed is True

    def test_detect_webm_format(self):
        """Test detection of WebM/Matroska format."""
        from utils.audio_buffer import detect_audio_format

        # WebM EBML header signature
        webm_data = b"\x1a\x45\xdf\xa3" + bytes(100)

        format_name, is_compressed = detect_audio_format(webm_data)
        assert format_name == "WebM/Matroska"
        assert is_compressed is True

    def test_detect_mp3_id3_format(self):
        """Test detection of MP3 with ID3 tag."""
        from utils.audio_buffer import detect_audio_format

        # MP3 ID3 tag signature
        mp3_data = b"ID3" + bytes(100)

        format_name, is_compressed = detect_audio_format(mp3_data)
        assert format_name == "MP3 (ID3 tag)"
        assert is_compressed is True

    def test_detect_mp3_frame_sync(self):
        """Test detection of MP3 frame sync."""
        from utils.audio_buffer import detect_audio_format

        # MP3 frame sync bytes
        mp3_data = b"\xff\xfb" + bytes(100)

        format_name, is_compressed = detect_audio_format(mp3_data)
        assert format_name == "MP3 (frame sync)"
        assert is_compressed is True

    def test_detect_flac_format(self):
        """Test detection of FLAC format."""
        from utils.audio_buffer import detect_audio_format

        # FLAC signature
        flac_data = b"fLaC" + bytes(100)

        format_name, is_compressed = detect_audio_format(flac_data)
        assert format_name == "FLAC"
        assert is_compressed is True

    def test_detect_short_data(self):
        """Test detection with very short data."""
        from utils.audio_buffer import detect_audio_format

        # Too short to have a signature
        short_data = b"ab"

        format_name, is_compressed = detect_audio_format(short_data)
        assert "unknown" in format_name
        assert is_compressed is False
