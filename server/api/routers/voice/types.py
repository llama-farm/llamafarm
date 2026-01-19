"""
Pydantic models for Voice chat WebSocket protocol.

Protocol messages for the real-time voice assistant pipeline:
- Speech In → STT → LLM → TTS → Speech Out
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class VoiceState(str, Enum):
    """Voice session state machine states."""

    IDLE = "idle"  # Waiting for input
    LISTENING = "listening"  # Receiving audio input
    PROCESSING = "processing"  # STT + LLM in progress
    SPEAKING = "speaking"  # TTS output playing
    INTERRUPTED = "interrupted"  # Barge-in occurred


# ============================================================================
# Client → Server Messages
# ============================================================================


class InterruptMessage(BaseModel):
    """Client request to interrupt current TTS playback (barge-in)."""

    type: Literal["interrupt"] = "interrupt"


class EndMessage(BaseModel):
    """Client signal to process accumulated audio."""

    type: Literal["end"] = "end"


class ConfigMessage(BaseModel):
    """Client request to update session configuration."""

    type: Literal["config"] = "config"
    stt_model: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    llm_model: str | None = None
    language: str | None = None
    speed: float | None = None
    sentence_boundary_only: bool | None = None


# ============================================================================
# Server → Client Messages
# ============================================================================


class SessionInfoMessage(BaseModel):
    """Session created or resumed."""

    type: Literal["session_info"] = "session_info"
    session_id: str


class TranscriptionMessage(BaseModel):
    """STT transcription result."""

    type: Literal["transcription"] = "transcription"
    text: str
    is_final: bool = True


class LLMTextMessage(BaseModel):
    """LLM response text (phrase for display)."""

    type: Literal["llm_text"] = "llm_text"
    text: str
    is_final: bool = False


class TTSStartMessage(BaseModel):
    """TTS synthesis starting for a phrase."""

    type: Literal["tts_start"] = "tts_start"
    phrase_index: int


class TTSDoneMessage(BaseModel):
    """TTS synthesis complete for a phrase."""

    type: Literal["tts_done"] = "tts_done"
    phrase_index: int
    duration: float


class StatusMessage(BaseModel):
    """Pipeline state change notification."""

    type: Literal["status"] = "status"
    state: VoiceState


class ErrorMessage(BaseModel):
    """Error occurred in pipeline."""

    type: Literal["error"] = "error"
    message: str


class ClosedMessage(BaseModel):
    """Session closed."""

    type: Literal["closed"] = "closed"


# ============================================================================
# Session Configuration
# ============================================================================


class VoiceSessionConfig(BaseModel):
    """Configuration for a voice chat session."""

    session_id: str | None = Field(
        default=None,
        description="Existing session ID to resume, or None for new session",
    )
    stt_model: str = Field(default="base", description="Whisper model size")
    tts_model: str = Field(default="kokoro", description="TTS model ID")
    tts_voice: str = Field(default="af_heart", description="TTS voice ID")
    llm_model: str = Field(default="", description="LLM model ID (required)")
    language: str = Field(default="en", description="STT language code")
    speed: float = Field(default=0.95, ge=0.5, le=2.0, description="TTS speed (0.95 for natural pace)")
    system_prompt: str | None = Field(
        default=None, description="System prompt for LLM"
    )
    enable_thinking: bool = Field(
        default=False,
        description="Enable LLM thinking/reasoning mode. Disabled by default for voice.",
    )
    sentence_boundary_only: bool = Field(
        default=True,
        description="Only split text on sentence boundaries (. ! ?) for natural speech. "
        "Set to False for aggressive chunking (lower latency but choppier speech).",
    )
