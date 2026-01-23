"""
Audio format encoding utilities for TTS output.

Converts raw PCM audio to various output formats (MP3, Opus, WAV, FLAC, AAC).
Supports both complete encoding and streaming chunk encoding.
"""

import io
import logging
import wave
from typing import Literal

logger = logging.getLogger(__name__)

# Supported output formats and their MIME types
AUDIO_FORMATS = {
    "pcm": "audio/pcm",
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "flac": "audio/flac",
    "aac": "audio/aac",
}

AudioFormat = Literal["pcm", "wav", "mp3", "opus", "flac", "aac"]


def pcm_to_wav(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    """Convert raw PCM to WAV format.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed integers).
        sample_rate: Sample rate in Hz (default 24000 for Kokoro).
        channels: Number of audio channels (default 1 for mono).
        sample_width: Bytes per sample (default 2 for 16-bit).

    Returns:
        WAV file bytes including header.
    """
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return wav_buffer.getvalue()


def pcm_to_mp3(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    bitrate: int = 128,
) -> bytes:
    """Convert raw PCM to MP3 format using pydub/ffmpeg.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed integers).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        bitrate: MP3 bitrate in kbps.

    Returns:
        MP3 file bytes.

    Raises:
        ImportError: If pydub is not installed.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "pydub is not installed. "
            "Install it with: uv pip install 'universal-runtime[tts]'"
        ) from e

    audio = AudioSegment(
        data=pcm_data,
        sample_width=2,  # 16-bit
        frame_rate=sample_rate,
        channels=channels,
    )

    mp3_buffer = io.BytesIO()
    audio.export(mp3_buffer, format="mp3", bitrate=f"{bitrate}k")
    return mp3_buffer.getvalue()


def pcm_to_opus(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
) -> bytes:
    """Convert raw PCM to Opus format using PyAV.

    Opus is ideal for streaming due to low latency and good compression.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed integers).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        Opus/OGG file bytes.

    Raises:
        ImportError: If av (PyAV) is not installed.
    """
    try:
        import av
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "av (PyAV) is not installed. "
            "Install it with: uv pip install 'universal-runtime[tts]'"
        ) from e

    # Convert bytes to numpy array
    # Ensure buffer size is a multiple of 2 bytes (int16 = 2 bytes per sample)
    usable_bytes = len(pcm_data) - (len(pcm_data) % 2)
    if usable_bytes < 2:
        raise ValueError("PCM data too short - need at least 2 bytes")
    pcm_aligned = pcm_data[:usable_bytes] if usable_bytes < len(pcm_data) else pcm_data
    samples = np.frombuffer(pcm_aligned, dtype=np.int16)

    # Create output container in memory
    output = io.BytesIO()
    container = av.open(output, mode="w", format="ogg")
    stream = container.add_stream("libopus", rate=sample_rate)
    stream.channels = channels

    # Create audio frame
    # PyAV expects samples in shape (channels, samples)
    if channels == 1:
        samples_reshaped = samples.reshape(1, -1)
    else:
        # Interleaved stereo: reshape to (samples, channels) then transpose
        samples_reshaped = samples.reshape(-1, channels).T

    layout = "mono" if channels == 1 else "stereo"
    frame = av.AudioFrame.from_ndarray(samples_reshaped, format="s16", layout=layout)
    frame.sample_rate = sample_rate

    # Encode audio
    for packet in stream.encode(frame):
        container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return output.getvalue()


def pcm_to_flac(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
) -> bytes:
    """Convert raw PCM to FLAC format.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed integers).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.

    Returns:
        FLAC file bytes.
    """
    try:
        from pydub import AudioSegment
    except ImportError as e:
        raise ImportError(
            "pydub is not installed. "
            "Install it with: uv pip install 'universal-runtime[tts]'"
        ) from e

    audio = AudioSegment(
        data=pcm_data,
        sample_width=2,
        frame_rate=sample_rate,
        channels=channels,
    )

    flac_buffer = io.BytesIO()
    audio.export(flac_buffer, format="flac")
    return flac_buffer.getvalue()


def pcm_to_aac(
    pcm_data: bytes,
    sample_rate: int = 24000,
    channels: int = 1,
    bitrate: int = 128,
) -> bytes:
    """Convert raw PCM to AAC format.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed integers).
        sample_rate: Sample rate in Hz.
        channels: Number of audio channels.
        bitrate: AAC bitrate in kbps.

    Returns:
        AAC file bytes (in ADTS container).
    """
    try:
        import av
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "av (PyAV) is not installed. "
            "Install it with: uv pip install 'universal-runtime[tts]'"
        ) from e

    # Ensure buffer size is a multiple of 2 bytes (int16 = 2 bytes per sample)
    usable_bytes = len(pcm_data) - (len(pcm_data) % 2)
    if usable_bytes < 2:
        raise ValueError("PCM data too short - need at least 2 bytes")
    pcm_aligned = pcm_data[:usable_bytes] if usable_bytes < len(pcm_data) else pcm_data
    samples = np.frombuffer(pcm_aligned, dtype=np.int16)

    output = io.BytesIO()
    container = av.open(output, mode="w", format="adts")
    stream = container.add_stream("aac", rate=sample_rate)
    stream.channels = channels
    stream.bit_rate = bitrate * 1000

    if channels == 1:
        samples_reshaped = samples.reshape(1, -1)
    else:
        samples_reshaped = samples.reshape(-1, channels).T

    layout = "mono" if channels == 1 else "stereo"
    frame = av.AudioFrame.from_ndarray(samples_reshaped, format="s16", layout=layout)
    frame.sample_rate = sample_rate

    for packet in stream.encode(frame):
        container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)

    container.close()
    return output.getvalue()


def encode_audio(
    pcm_data: bytes,
    format: AudioFormat = "mp3",
    sample_rate: int = 24000,
    channels: int = 1,
    bitrate: int = 128,
) -> tuple[bytes, str]:
    """Encode PCM audio to the specified format.

    Args:
        pcm_data: Raw PCM audio bytes (16-bit signed integers).
        format: Output format ("pcm", "wav", "mp3", "opus", "flac", "aac").
        sample_rate: Sample rate in Hz (default 24000 for Kokoro).
        channels: Number of audio channels (default 1 for mono).
        bitrate: Bitrate for lossy formats in kbps (default 128).

    Returns:
        Tuple of (encoded_bytes, content_type).

    Raises:
        ValueError: If format is not supported.
    """
    if format not in AUDIO_FORMATS:
        supported = list(AUDIO_FORMATS.keys())
        raise ValueError(f"Unsupported format: {format}. Supported: {supported}")

    content_type = AUDIO_FORMATS[format]

    if format == "pcm":
        return pcm_data, content_type
    elif format == "wav":
        return pcm_to_wav(pcm_data, sample_rate, channels), content_type
    elif format == "mp3":
        return pcm_to_mp3(pcm_data, sample_rate, channels, bitrate), content_type
    elif format == "opus":
        return pcm_to_opus(pcm_data, sample_rate, channels), content_type
    elif format == "flac":
        return pcm_to_flac(pcm_data, sample_rate, channels), content_type
    elif format == "aac":
        return pcm_to_aac(pcm_data, sample_rate, channels, bitrate), content_type
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_content_type(format: AudioFormat) -> str:
    """Get the MIME content type for an audio format.

    Args:
        format: Audio format name.

    Returns:
        MIME content type string.
    """
    return AUDIO_FORMATS.get(format, "application/octet-stream")


def get_file_extension(format: AudioFormat) -> str:
    """Get the file extension for an audio format.

    Args:
        format: Audio format name.

    Returns:
        File extension (without dot).
    """
    extensions = {
        "pcm": "pcm",
        "wav": "wav",
        "mp3": "mp3",
        "opus": "opus",
        "flac": "flac",
        "aac": "aac",
    }
    return extensions.get(format, "bin")
