"""Audio types for speech-to-text transcription and translation endpoints."""

from typing import Literal

from pydantic import BaseModel

# =============================================================================
# Transcription Types
# =============================================================================


class TranscriptionSegment(BaseModel):
    """Segment of transcribed audio with timing."""

    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int] | None = None
    temperature: float | None = None
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


class TranscriptionWord(BaseModel):
    """Word-level timing in transcription."""

    word: str
    start: float
    end: float


class TranscriptionResponse(BaseModel):
    """OpenAI-compatible transcription response."""

    text: str
    task: str | None = None
    language: str | None = None
    duration: float | None = None
    segments: list[TranscriptionSegment] | None = None
    words: list[TranscriptionWord] | None = None


class TranscriptionRequest(BaseModel):
    """Transcription request parameters (for JSON body, not multipart)."""

    model: str = "distil-large-v3"  # Whisper model size
    language: str | None = None  # ISO language code
    prompt: str | None = None  # Optional conditioning text
    response_format: Literal["json", "text", "srt", "vtt", "verbose_json"] = "json"
    temperature: float = 0.0  # Sampling temperature
    timestamp_granularities: list[Literal["word", "segment"]] | None = None


# =============================================================================
# Translation Types
# =============================================================================


class TranslationResponse(BaseModel):
    """OpenAI-compatible translation response."""

    text: str
    task: str | None = None
    language: str | None = None
    duration: float | None = None


class TranslationRequest(BaseModel):
    """Translation request parameters (for JSON body, not multipart)."""

    model: str = "distil-large-v3"  # Whisper model size
    prompt: str | None = None  # Optional conditioning text
    response_format: Literal["json", "text"] = "json"
    temperature: float = 0.0  # Sampling temperature
