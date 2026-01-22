"""
Audio buffering utilities for real-time speech-to-text streaming.

Provides:
- AudioBuffer: Accumulates audio chunks and provides complete audio for transcription
- VAD integration for detecting speech segments
- Audio format conversion utilities
"""

import io
import logging
import struct
import subprocess
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Audio format constants
SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
SAMPLE_WIDTH = 2  # 16-bit audio (2 bytes per sample)
CHANNELS = 1  # Mono audio

# WAV header signature
WAV_HEADER_RIFF = b"RIFF"
WAV_HEADER_WAVE = b"WAVE"


def strip_wav_header(audio_data: bytes) -> bytes:
    """Strip WAV header from audio data if present, returning raw PCM.

    Handles cases where:
    - Audio is raw PCM (no header) - returns as-is
    - Audio has a complete WAV header - strips it
    - Audio starts with partial WAV header from streaming - strips it

    Args:
        audio_data: Audio bytes that may or may not have a WAV header

    Returns:
        Raw PCM audio bytes without WAV header
    """
    if len(audio_data) < 44:
        # Too short to have a full WAV header, but check for RIFF signature
        if audio_data[:4] == WAV_HEADER_RIFF:
            logger.warning("Received partial WAV header, cannot strip")
        return audio_data

    # Check for WAV header signature
    if audio_data[:4] == WAV_HEADER_RIFF and audio_data[8:12] == WAV_HEADER_WAVE:
        # This is a WAV file, extract PCM data
        try:
            wav_buffer = io.BytesIO(audio_data)
            with wave.open(wav_buffer, "rb") as wav_file:
                pcm_data = wav_file.readframes(wav_file.getnframes())
                return pcm_data
        except Exception as e:
            # If wave parsing fails, try to find data chunk manually
            logger.warning(f"Failed to parse WAV header: {e}, trying manual extraction")
            # Look for 'data' chunk marker
            data_marker = audio_data.find(b"data")
            if data_marker != -1 and data_marker + 8 < len(audio_data):
                # Skip 'data' (4 bytes) + size (4 bytes)
                return audio_data[data_marker + 8 :]
            # Fallback: assume 44-byte header
            return audio_data[44:]

    return audio_data


def calculate_audio_energy(
    audio_data: bytes, sample_width: int = SAMPLE_WIDTH
) -> float:
    """Calculate the RMS energy of audio data.

    Args:
        audio_data: Raw PCM audio bytes
        sample_width: Bytes per sample (default: 2 for 16-bit)

    Returns:
        RMS energy value (0.0 to 1.0 normalized)
    """
    if not audio_data or len(audio_data) < sample_width:
        return 0.0

    # Unpack samples based on sample width
    if sample_width == 2:
        # 16-bit signed samples
        num_samples = len(audio_data) // 2
        if num_samples == 0:
            return 0.0
        samples = struct.unpack(f"{num_samples}h", audio_data[: num_samples * 2])
        max_val = 32767.0
    else:
        # Fallback: treat as 8-bit unsigned
        samples = audio_data
        max_val = 127.0

    # Calculate RMS
    sum_squares = sum(s * s for s in samples)
    rms = (sum_squares / len(samples)) ** 0.5

    # Normalize to 0-1 range
    return rms / max_val


def is_silence(
    audio_data: bytes,
    threshold: float = 0.01,
    sample_width: int = SAMPLE_WIDTH,
) -> bool:
    """Check if audio data is effectively silence.

    Args:
        audio_data: Raw PCM audio bytes
        threshold: Energy threshold below which audio is considered silence (0.0 to 1.0)
        sample_width: Bytes per sample

    Returns:
        True if audio energy is below threshold
    """
    energy = calculate_audio_energy(audio_data, sample_width)
    return energy < threshold


# Compressed audio format signatures
COMPRESSED_AUDIO_SIGNATURES = {
    b"OggS": "Ogg (Opus/Vorbis)",
    b"\x1a\x45\xdf\xa3": "WebM/Matroska",
    b"ID3": "MP3 (ID3 tag)",
    b"\xff\xfb": "MP3 (frame sync)",
    b"\xff\xfa": "MP3 (frame sync)",
    b"\xff\xf3": "MP3 (frame sync)",
    b"\xff\xf2": "MP3 (frame sync)",
    b"fLaC": "FLAC",
    b"FORM": "AIFF",
}

# WebM Cluster element ID (seen when streaming WebM chunks)
WEBM_CLUSTER_ID = b"\x1f\x43\xb6\x75"

# Allowed audio formats for FFmpeg input (whitelist for security)
ALLOWED_FFMPEG_FORMATS = frozenset(
    {
        "webm",
        "ogg",
        "mp3",
        "flac",
        "aiff",
        "wav",
        "m4a",
        "mp4",
        "opus",
    }
)


def detect_audio_format(audio_data: bytes) -> tuple[str, bool]:
    """Detect the format of audio data.

    Args:
        audio_data: Audio bytes to analyze

    Returns:
        Tuple of (format_name, is_compressed)
        - format_name: Human-readable format name
        - is_compressed: True if the format is compressed (not raw PCM)
    """
    if len(audio_data) < 4:
        return ("unknown (too short)", False)

    # Check for WAV header
    if (
        audio_data[:4] == WAV_HEADER_RIFF
        and len(audio_data) >= 12
        and audio_data[8:12] == WAV_HEADER_WAVE
    ):
        return ("WAV", False)

    # Check for compressed audio signatures
    for signature, format_name in COMPRESSED_AUDIO_SIGNATURES.items():
        if audio_data.startswith(signature):
            return (format_name, True)

    # Check for WebM cluster chunks (from streaming MediaRecorder)
    # Cluster ID is 0x1F43B675, but first byte might be truncated
    if audio_data[:4] == WEBM_CLUSTER_ID:
        return ("WebM Cluster", True)

    # Check for Matroska/WebM by looking for common element IDs
    # SimpleBlock (0xA3), Block (0xA1), Timecode (0xE7), etc.
    if len(audio_data) >= 2:
        first_byte = audio_data[0]
        # WebM/Matroska element IDs often start with these
        # Additional heuristic: check for patterns common in Opus/WebM
        # Opus frames often have 0xFF padding or specific patterns
        if (
            first_byte in (0x1A, 0x1F, 0xA3, 0xA1, 0xE7, 0x43)
            and b"\xff\xff\xff\xff" in audio_data[:20]
        ):
            return ("WebM/Opus (likely)", True)

    # Assume raw PCM if no compressed format detected
    return ("PCM (assumed)", False)


def _decode_with_pyav(
    audio_data: bytes,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
) -> bytes:
    """Decode compressed audio using PyAV (in-process, efficient).

    Args:
        audio_data: Compressed audio bytes
        sample_rate: Output sample rate
        channels: Output channels

    Returns:
        Raw PCM audio bytes (signed 16-bit little-endian)
    """
    import av

    # Open container from bytes using context manager to ensure cleanup
    with av.open(io.BytesIO(audio_data)) as container:
        # Find audio stream
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise RuntimeError("No audio stream found in container")

        # Set up resampler for 16kHz mono s16
        resampler = av.AudioResampler(
            format="s16",
            layout="mono" if channels == 1 else "stereo",
            rate=sample_rate,
        )

        # Decode and resample
        pcm_chunks = []
        for frame in container.decode(audio_stream):
            # Resample to target format
            resampled_frames = resampler.resample(frame)
            for resampled in resampled_frames:
                # Get raw bytes from plane
                pcm_chunks.append(bytes(resampled.planes[0]))

        return b"".join(pcm_chunks)


def _decode_with_ffmpeg(
    audio_data: bytes,
    input_format: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    channels: int = CHANNELS,
    timeout: float | None = None,
) -> bytes:
    """Decode compressed audio using FFmpeg subprocess (fallback).

    Args:
        audio_data: Compressed audio bytes
        input_format: Input format hint for FFmpeg (must be in ALLOWED_FFMPEG_FORMATS)
        sample_rate: Output sample rate
        channels: Output channels
        timeout: Timeout in seconds. If None, calculated as 30s + 2s per MB of input.

    Returns:
        Raw PCM audio bytes (signed 16-bit little-endian)

    Raises:
        ValueError: If input_format is not in the allowed formats whitelist
        RuntimeError: If FFmpeg decoding fails
    """
    # Calculate timeout based on input size if not specified
    if timeout is None:
        input_size_mb = len(audio_data) / (1024 * 1024)
        timeout = 30.0 + (input_size_mb * 2.0)  # 30s base + 2s per MB
    # Validate input_format against whitelist to prevent command injection
    if input_format is not None and input_format not in ALLOWED_FFMPEG_FORMATS:
        raise ValueError(
            f"Unsupported audio format: {input_format}. "
            f"Allowed formats: {', '.join(sorted(ALLOWED_FFMPEG_FORMATS))}"
        )

    # Write input to temp file (FFmpeg needs seekable input for some formats)
    input_suffix = f".{input_format}" if input_format else ".webm"
    with tempfile.NamedTemporaryFile(suffix=input_suffix, delete=False) as tmp_in:
        tmp_in.write(audio_data)
        tmp_in_path = tmp_in.name

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
        ]

        if input_format:
            cmd.extend(["-f", input_format])

        cmd.extend(
            [
                "-i",
                tmp_in_path,
                "-ar",
                str(sample_rate),
                "-ac",
                str(channels),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ]
        )

        result = subprocess.run(cmd, capture_output=True, timeout=timeout)

        if result.returncode != 0:
            error_msg = result.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"FFmpeg decoding failed: {error_msg}")

        return result.stdout

    except FileNotFoundError as e:
        raise RuntimeError(
            "FFmpeg not found. Install FFmpeg to decode compressed audio formats."
        ) from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("FFmpeg decoding timed out") from e
    finally:
        Path(tmp_in_path).unlink(missing_ok=True)


# Check if PyAV is available (preferred for efficiency)
try:
    import av as _av  # noqa: F401

    PYAV_AVAILABLE = True
except ImportError:
    PYAV_AVAILABLE = False


def decode_audio_to_pcm(
    audio_data: bytes,
    input_format: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    sample_width: int = SAMPLE_WIDTH,
    channels: int = CHANNELS,
) -> bytes:
    """Decode compressed audio to raw PCM.

    Uses PyAV for efficient in-process decoding if available,
    falls back to FFmpeg subprocess otherwise.

    Args:
        audio_data: Compressed audio bytes (WebM, Opus, MP3, etc.)
        input_format: Input format hint (e.g., "webm", "ogg", "mp3").
        sample_rate: Output sample rate (default: 16000 Hz for Whisper)
        sample_width: Output bytes per sample (default: 2 for 16-bit)
        channels: Output channels (default: 1 for mono)

    Returns:
        Raw PCM audio bytes (signed 16-bit little-endian)

    Raises:
        RuntimeError: If decoding fails
    """
    if PYAV_AVAILABLE:
        try:
            return _decode_with_pyav(audio_data, sample_rate, channels)
        except Exception as e:
            logger.warning(f"PyAV decoding failed, falling back to FFmpeg: {e}")
            # Fall through to FFmpeg

    return _decode_with_ffmpeg(audio_data, input_format, sample_rate, channels)


def decode_audio_bytes(
    audio_data: bytes,
    sample_rate: int = SAMPLE_RATE,
) -> bytes:
    """Auto-detect format and decode audio to raw PCM if needed.

    This is a convenience wrapper that:
    1. Detects if audio is compressed
    2. If compressed, decodes to raw PCM
    3. If already PCM or WAV, returns raw PCM (stripping WAV header if present)

    Args:
        audio_data: Audio bytes (compressed or raw PCM)
        sample_rate: Target sample rate for decoding

    Returns:
        Raw PCM audio bytes (16-bit mono)
    """
    format_name, is_compressed = detect_audio_format(audio_data)

    if is_compressed:
        logger.debug(f"Decoding compressed audio format: {format_name}")
        # Map detected format to FFmpeg input format
        format_map = {
            "Ogg (Opus/Vorbis)": "ogg",
            "WebM/Matroska": "webm",
            "WebM Cluster": "webm",
            "WebM/Opus (likely)": "webm",
            "MP3 (ID3 tag)": "mp3",
            "MP3 (frame sync)": "mp3",
            "FLAC": "flac",
            "AIFF": "aiff",
        }
        input_format = format_map.get(format_name, "webm")
        return decode_audio_to_pcm(audio_data, input_format=input_format)

    elif format_name == "WAV":
        # Strip WAV header to get raw PCM
        return strip_wav_header(audio_data)

    else:
        # Already raw PCM
        return audio_data


@dataclass
class AudioChunk:
    """A chunk of audio data with metadata."""

    data: bytes
    sample_rate: int = SAMPLE_RATE
    sample_width: int = SAMPLE_WIDTH
    channels: int = CHANNELS
    timestamp: float = 0.0  # Start time in seconds


@dataclass
class AudioBuffer:
    """
    Buffer for accumulating audio chunks for transcription.

    Accumulates raw PCM audio data and provides methods to:
    - Add audio chunks
    - Check if enough audio is available for transcription
    - Get accumulated audio as WAV bytes
    - Reset the buffer

    Attributes:
        sample_rate: Audio sample rate (default: 16000 Hz for Whisper)
        sample_width: Bytes per sample (default: 2 for 16-bit)
        channels: Number of audio channels (default: 1 for mono)
        min_duration: Minimum audio duration (seconds) before transcription
        max_duration: Maximum audio duration (seconds) to accumulate
    """

    sample_rate: int = SAMPLE_RATE
    sample_width: int = SAMPLE_WIDTH
    channels: int = CHANNELS
    min_duration: float = 0.5  # Minimum 0.5 seconds before transcribing
    max_duration: float = 30.0  # Maximum 30 seconds (Whisper's native chunk size)
    _chunks: list[bytes] = field(default_factory=list)
    _total_samples: int = 0

    def add(self, audio_data: bytes) -> None:
        """Add audio data to the buffer.

        Args:
            audio_data: Raw PCM audio bytes (16-bit, mono, 16kHz)
        """
        self._chunks.append(audio_data)
        # Calculate samples: bytes / (sample_width * channels)
        num_samples = len(audio_data) // (self.sample_width * self.channels)
        self._total_samples += num_samples

    def add_chunk(self, chunk: AudioChunk) -> None:
        """Add an AudioChunk to the buffer."""
        self.add(chunk.data)

    @property
    def duration(self) -> float:
        """Get the total duration of buffered audio in seconds."""
        return self._total_samples / self.sample_rate

    @property
    def has_enough_audio(self) -> bool:
        """Check if we have enough audio for transcription."""
        return self.duration >= self.min_duration

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached maximum duration."""
        return self.duration >= self.max_duration

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._total_samples == 0

    def get_audio_bytes(self) -> bytes:
        """Get all buffered audio as raw PCM bytes."""
        return b"".join(self._chunks)

    def get_wav_bytes(self) -> bytes:
        """Get buffered audio as WAV format bytes.

        Returns:
            WAV file bytes suitable for transcription
        """
        pcm_data = self.get_audio_bytes()

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(pcm_data)

        return wav_buffer.getvalue()

    def clear(self) -> None:
        """Clear the buffer."""
        self._chunks.clear()
        self._total_samples = 0

    def pop_audio(self) -> bytes:
        """Get all buffered audio and clear the buffer.

        Returns:
            WAV file bytes
        """
        wav_bytes = self.get_wav_bytes()
        self.clear()
        return wav_bytes


class VADProcessor:
    """
    Voice Activity Detection processor using webrtcvad.

    Detects speech segments in audio to avoid transcribing silence.
    """

    def __init__(
        self,
        aggressiveness: int = 3,
        sample_rate: int = SAMPLE_RATE,
        frame_duration_ms: int = 30,
    ):
        """Initialize VAD processor.

        Args:
            aggressiveness: VAD aggressiveness (0-3, higher = more aggressive filtering)
            sample_rate: Audio sample rate (8000, 16000, 32000, or 48000)
            frame_duration_ms: Frame duration in ms (10, 20, or 30)
        """
        self.aggressiveness = aggressiveness
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self._vad = None

        # Calculate frame size in bytes
        # frame_size = sample_rate * frame_duration_ms / 1000 * 2 (16-bit)
        self.frame_size = int(sample_rate * frame_duration_ms / 1000 * 2)

    def _ensure_vad(self):
        """Lazily initialize VAD to handle import errors gracefully."""
        if self._vad is None:
            try:
                import webrtcvad

                self._vad = webrtcvad.Vad(self.aggressiveness)
            except ImportError as e:
                raise ImportError(
                    "webrtcvad is not installed. "
                    "Install it with: uv pip install 'universal-runtime[speech]'"
                ) from e

    def is_speech(self, audio_frame: bytes) -> bool:
        """Check if an audio frame contains speech.

        Args:
            audio_frame: Raw PCM audio frame (must be 10, 20, or 30ms)

        Returns:
            True if speech is detected
        """
        self._ensure_vad()

        # Ensure frame is correct size
        if len(audio_frame) != self.frame_size:
            logger.warning(
                f"Frame size mismatch: got {len(audio_frame)}, "
                f"expected {self.frame_size}"
            )
            return True  # Assume speech if size is wrong

        return self._vad.is_speech(audio_frame, self.sample_rate)

    def filter_audio(self, audio_data: bytes) -> bytes:
        """Filter audio to keep only speech segments.

        Args:
            audio_data: Raw PCM audio bytes

        Returns:
            Audio bytes with non-speech segments removed
        """
        self._ensure_vad()

        speech_frames = []

        # Process audio in frames
        for i in range(0, len(audio_data) - self.frame_size + 1, self.frame_size):
            frame = audio_data[i : i + self.frame_size]
            if self.is_speech(frame):
                speech_frames.append(frame)

        return b"".join(speech_frames)

    def get_speech_ratio(self, audio_data: bytes) -> float:
        """Calculate the ratio of speech to total audio.

        Args:
            audio_data: Raw PCM audio bytes

        Returns:
            Ratio of speech frames (0.0 to 1.0)
        """
        self._ensure_vad()

        total_frames = 0
        speech_frames = 0

        for i in range(0, len(audio_data) - self.frame_size + 1, self.frame_size):
            frame = audio_data[i : i + self.frame_size]
            total_frames += 1
            if self.is_speech(frame):
                speech_frames += 1

        return speech_frames / total_frames if total_frames > 0 else 0.0


class StreamingAudioBuffer:
    """
    Advanced audio buffer for streaming transcription with VAD support.

    Combines audio buffering with voice activity detection to:
    - Accumulate audio during speech
    - Detect speech boundaries
    - Trigger transcription at natural pause points
    - Or use time-based chunking for continuous output

    Supports both raw PCM and compressed audio formats (WebM/Opus, etc.).
    Compressed audio is accumulated and decoded when transcription is triggered.
    """

    def __init__(
        self,
        min_speech_duration: float = 0.5,
        max_speech_duration: float = 30.0,
        silence_threshold: float = 0.5,
        vad_aggressiveness: int = 2,
        use_vad: bool = True,
        chunk_interval: float | None = None,
    ):
        """Initialize streaming buffer.

        Args:
            min_speech_duration: Minimum speech duration before transcription
            max_speech_duration: Maximum duration before forced transcription
            silence_threshold: Seconds of silence to trigger transcription
            vad_aggressiveness: VAD aggressiveness (0-3)
            use_vad: Whether to use VAD for speech detection
            chunk_interval: If set, forces transcription every N seconds regardless
                           of VAD state. Useful for continuous streaming output.
                           When set, overrides VAD-based triggering.
        """
        self.min_speech_duration = min_speech_duration
        self.max_speech_duration = max_speech_duration
        self.silence_threshold = silence_threshold
        self.use_vad = use_vad
        self.chunk_interval = chunk_interval

        self._buffer = AudioBuffer(
            min_duration=min_speech_duration,
            max_duration=max_speech_duration,
        )

        self._vad: VADProcessor | None = None
        if use_vad and chunk_interval is None:
            # Only use VAD if not in time-based chunking mode
            try:
                self._vad = VADProcessor(aggressiveness=vad_aggressiveness)
            except ImportError:
                logger.warning("VAD not available, speech detection disabled")
                self.use_vad = False

        self._silence_samples = 0
        self._in_speech = False

        # Compressed audio handling
        self._compressed_chunks: list[bytes] = []
        self._compressed_format: str | None = None
        self._is_compressed_mode = False
        self._compressed_bytes_count = 0

    def _estimate_compressed_duration(self) -> float:
        """Estimate duration of compressed audio based on byte count.

        Browser MediaRecorder with Opus typically uses 48-128 kbps,
        which translates to approximately 6-16 KB/second.
        We use a conservative estimate of 8 KB/second to trigger
        decoding slightly early rather than accumulating too much.

        Returns:
            Estimated duration in seconds. Returns 0.0 if no bytes accumulated.
        """
        # Estimate 8 KB/second for compressed audio (conservative for Opus)
        COMPRESSED_BYTES_PER_SECOND = 8000.0
        return self._compressed_bytes_count / COMPRESSED_BYTES_PER_SECOND

    def add(self, audio_data: bytes) -> tuple[bool, bytes | None]:
        """Add audio data and check if transcription should be triggered.

        Args:
            audio_data: Raw PCM audio bytes, WAV, or compressed audio (WebM/Opus)

        Returns:
            Tuple of (should_transcribe, audio_bytes or None)
        """
        # Detect audio format
        format_name, is_compressed = detect_audio_format(audio_data)

        # Handle compressed audio (WebM/Opus from browser MediaRecorder)
        if is_compressed or self._is_compressed_mode:
            return self._add_compressed(audio_data, format_name)

        # Handle raw PCM/WAV
        # Strip WAV header if present (browser MediaRecorder often sends WAV)
        audio_data = strip_wav_header(audio_data)

        # Add to buffer
        self._buffer.add(audio_data)

        # Time-based chunking mode: transcribe every chunk_interval seconds
        if self.chunk_interval is not None:
            if self._buffer.duration >= self.chunk_interval:
                audio_bytes = self._buffer.pop_audio()
                return True, audio_bytes
            return False, None

        # VAD-based mode: detect speech boundaries
        if self.use_vad and self._vad:
            speech_ratio = self._vad.get_speech_ratio(audio_data)
            is_speech = speech_ratio > 0.3  # At least 30% speech

            if is_speech:
                self._in_speech = True
                self._silence_samples = 0
            else:
                # Count silence
                num_samples = len(audio_data) // (SAMPLE_WIDTH * CHANNELS)
                self._silence_samples += num_samples

        # Check if we should transcribe
        should_transcribe = False

        # Force transcription if buffer is full
        if self._buffer.is_full:
            should_transcribe = True

        # Transcribe on silence after speech (natural pause)
        elif self._in_speech and self._silence_samples > 0:
            silence_duration = self._silence_samples / SAMPLE_RATE
            if (
                silence_duration >= self.silence_threshold
                and self._buffer.has_enough_audio
            ):
                should_transcribe = True

        if should_transcribe:
            audio_bytes = self._buffer.pop_audio()
            self._in_speech = False
            self._silence_samples = 0
            return True, audio_bytes

        return False, None

    def _add_compressed(
        self, audio_data: bytes, format_name: str
    ) -> tuple[bool, bytes | None]:
        """Handle compressed audio data (WebM/Opus, etc.).

        Accumulates compressed chunks and decodes when chunk_interval is reached.

        Args:
            audio_data: Compressed audio bytes
            format_name: Detected format name

        Returns:
            Tuple of (should_transcribe, audio_bytes or None)
        """
        # Enter compressed mode on first compressed chunk
        if not self._is_compressed_mode:
            self._is_compressed_mode = True
            self._compressed_format = format_name
            logger.info(f"Entering compressed audio mode: {format_name}")

        # Accumulate compressed chunks
        self._compressed_chunks.append(audio_data)
        self._compressed_bytes_count += len(audio_data)

        # Time-based chunking mode: decode and transcribe every chunk_interval
        if self.chunk_interval is not None:
            estimated_duration = self._estimate_compressed_duration()

            if estimated_duration >= self.chunk_interval:
                return self._decode_and_return()

        # VAD-based mode not supported for compressed audio
        # (would need to decode to check for speech, defeating the purpose)
        # Fall back to max duration
        elif self._estimate_compressed_duration() >= self.max_speech_duration:
            return self._decode_and_return()

        return False, None

    def _decode_and_return(self) -> tuple[bool, bytes | None]:
        """Decode accumulated compressed audio and return as WAV.

        Returns:
            Tuple of (True, wav_bytes) or (False, None) on error
        """
        if not self._compressed_chunks:
            return False, None

        # Combine all compressed chunks
        compressed_data = b"".join(self._compressed_chunks)

        # Clear compressed buffer
        self._compressed_chunks.clear()
        self._compressed_bytes_count = 0

        try:
            # Decode to raw PCM
            pcm_data = decode_audio_bytes(compressed_data)

            if not pcm_data:
                logger.warning("Decoded audio is empty")
                return False, None

            # Convert PCM to WAV
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(SAMPLE_WIDTH)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(pcm_data)

            return True, wav_buffer.getvalue()

        except Exception as e:
            logger.error(f"Failed to decode compressed audio: {e}")
            return False, None

    def flush(self) -> bytes | None:
        """Flush any remaining audio in the buffer.

        Returns:
            WAV bytes if buffer has audio, None otherwise
        """
        # Handle compressed audio flush
        if self._is_compressed_mode and self._compressed_chunks:
            should_transcribe, wav_bytes = self._decode_and_return()
            return wav_bytes if should_transcribe else None

        # Handle raw PCM/WAV flush
        if self._buffer.is_empty:
            return None
        return self._buffer.pop_audio()

    def clear(self) -> None:
        """Clear the buffer and reset state."""
        self._buffer.clear()
        self._in_speech = False
        self._silence_samples = 0
        # Clear compressed audio state
        self._compressed_chunks.clear()
        self._compressed_format = None
        self._is_compressed_mode = False
        self._compressed_bytes_count = 0


def convert_audio_format(
    audio_data: bytes,
    input_format: Literal["pcm", "wav", "webm", "opus"] = "pcm",
    input_sample_rate: int = 16000,
    input_channels: int = 1,
) -> bytes:
    """Convert audio to the format expected by Whisper (16kHz mono PCM).

    Args:
        audio_data: Input audio bytes
        input_format: Format of input audio
        input_sample_rate: Sample rate of input audio
        input_channels: Number of channels in input audio

    Returns:
        PCM audio bytes at 16kHz mono
    """
    # If already in correct format, return as-is
    if (
        input_format == "pcm"
        and input_sample_rate == SAMPLE_RATE
        and input_channels == CHANNELS
    ):
        return audio_data

    # For WAV, extract PCM data
    if input_format == "wav":
        wav_buffer = io.BytesIO(audio_data)
        with wave.open(wav_buffer, "rb") as wav_file:
            # Read all frames
            pcm_data = wav_file.readframes(wav_file.getnframes())
            actual_rate = wav_file.getframerate()
            actual_channels = wav_file.getnchannels()
            actual_width = wav_file.getsampwidth()

            # Simple case: already correct format
            if (
                actual_rate == SAMPLE_RATE
                and actual_channels == CHANNELS
                and actual_width == SAMPLE_WIDTH
            ):
                return pcm_data

            # Resampling required but not implemented - raise error
            # to avoid silent quality degradation
            if actual_rate != SAMPLE_RATE:
                raise ValueError(
                    f"Audio sample rate mismatch: got {actual_rate}Hz, "
                    f"expected {SAMPLE_RATE}Hz. Use FFmpeg or PyAV to resample: "
                    f"ffmpeg -i input.wav -ar {SAMPLE_RATE} output.wav"
                )

            # Channel/width mismatch
            if actual_channels != CHANNELS or actual_width != SAMPLE_WIDTH:
                raise ValueError(
                    f"Audio format mismatch: got {actual_channels}ch/{actual_width * 8}bit, "
                    f"expected {CHANNELS}ch/{SAMPLE_WIDTH * 8}bit. Convert audio format first."
                )

            return pcm_data

    # For opus/webm, would need ffmpeg or similar
    # For now, return as-is with warning
    logger.warning(f"Audio format conversion from {input_format} not implemented")
    return audio_data


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = SAMPLE_RATE,
    sample_width: int = SAMPLE_WIDTH,
    channels: int = CHANNELS,
) -> bytes:
    """Convert raw PCM data to WAV format.

    Args:
        pcm_data: Raw PCM audio bytes
        sample_rate: Sample rate
        sample_width: Bytes per sample
        channels: Number of channels

    Returns:
        WAV file bytes
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


def float32_to_int16(audio_data: bytes) -> bytes:
    """Convert 32-bit float audio to 16-bit integer.

    Args:
        audio_data: Audio as 32-bit floats (-1.0 to 1.0)

    Returns:
        Audio as 16-bit integers
    """
    # Unpack as floats (slice to exact size to handle non-aligned buffers)
    num_samples = len(audio_data) // 4
    floats = struct.unpack(f"{num_samples}f", audio_data[: num_samples * 4])

    # Convert to int16
    int16_samples = []
    for f in floats:
        # Clamp to -1.0 to 1.0
        f = max(-1.0, min(1.0, f))
        # Scale to int16 range
        int16_samples.append(int(f * 32767))

    return struct.pack(f"{num_samples}h", *int16_samples)
