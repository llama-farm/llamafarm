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
    barge_in_enabled: bool | None = None
    barge_in_noise_filter: bool | None = None
    barge_in_min_chunks: int | None = None
    turn_detection_enabled: bool | None = None
    base_silence_duration: float | None = None
    thinking_silence_duration: float | None = None
    max_silence_duration: float | None = None
    emotion_detection_enabled: bool | None = None
    emotion_model: str | None = None
    emotion_confidence_threshold: float | None = None


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


class EmotionMessage(BaseModel):
    """Speech emotion recognition result."""

    type: Literal["emotion"] = "emotion"
    emotion: str
    confidence: float
    all_scores: dict[str, float]


class LLMTextMessage(BaseModel):
    """LLM response text (phrase for display)."""

    type: Literal["llm_text"] = "llm_text"
    text: str
    is_final: bool = False


class ToolCallMessage(BaseModel):
    """LLM tool call - sent via JSON only, never synthesized to speech."""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    function_name: str
    arguments: str  # JSON string of arguments


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
    llm_model: str = Field(default="", description="LLM model ID (required unless stt_only)")
    language: str = Field(default="en", description="STT language code")
    stt_only: bool = Field(
        default=False,
        description="STT-only mode - skip LLM and TTS, only return transcriptions.",
    )
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
    barge_in_enabled: bool = Field(
        default=True,
        description="Enable automatic barge-in detection. When enabled, incoming audio "
        "during TTS playback is analyzed and will interrupt if speech is detected. "
        "Requires client to handle echo cancellation.",
    )
    barge_in_noise_filter: bool = Field(
        default=True,
        description="Filter suspected background noise during barge-in detection. "
        "Requires multiple consecutive audio chunks above threshold to trigger interrupt. "
        "Set to False for more responsive (but potentially more false positive) detection.",
    )
    barge_in_min_chunks: int = Field(
        default=2,
        ge=1,
        le=10,
        description="Minimum consecutive chunks above speech threshold required to trigger "
        "barge-in when noise filter is enabled. Higher values reduce false positives.",
    )

    # End-of-turn detection settings
    turn_detection_enabled: bool = Field(
        default=True,
        description="Enable smart end-of-turn detection using linguistic analysis. "
        "When enabled, the system analyzes partial transcriptions to detect thinking "
        "pauses vs actual end of utterance, preventing premature LLM responses.",
    )
    base_silence_duration: float = Field(
        default=0.1,
        ge=0.1,
        le=2.0,
        description="Base silence duration for complete utterances (seconds). "
        "Used when the transcription appears linguistically complete.",
    )
    thinking_silence_duration: float = Field(
        default=0.7,
        ge=0.3,
        le=5.0,
        description="Extended silence duration for incomplete utterances (seconds). "
        "Used when linguistic analysis suggests the user is mid-thought.",
    )
    max_silence_duration: float = Field(
        default=1.5,
        ge=0.5,
        le=10.0,
        description="Maximum silence before forcing end-of-turn (seconds). "
        "Even if utterance seems incomplete, processing starts after this timeout.",
    )

    # Native audio support (for Omni models)
    use_native_audio: bool = Field(
        default=False,
        description="Use native audio input (skip STT). Enable this for models like "
        "Qwen2.5-Omni that can process audio directly. When enabled, audio is sent "
        "straight to the LLM without transcription.",
    )

    # Emotion detection settings
    emotion_detection_enabled: bool = Field(
        default=True,
        description="Enable speech emotion recognition. Analyzes audio to detect "
        "user emotional tone (angry, happy, sad, etc.) and includes it in the LLM "
        "context. Set to false in llamafarm.yaml to disable.",
    )
    emotion_model: str = Field(
        default="wav2vec2-lg-xlsr-en",
        description="Emotion recognition model ID. Options: wav2vec2-lg-xlsr-en (default), "
        "wav2vec2-base-superb. Custom HuggingFace model IDs also supported.",
    )
    emotion_confidence_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for emotion detection. "
        "Emotions below this threshold are reported as 'neutral'.",
    )
