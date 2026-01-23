"""Types for audio chat WebSocket protocol."""

from typing import Literal

from pydantic import BaseModel, Field


class AudioChatConfig(BaseModel):
    """Configuration message sent at start of audio chat session.

    Sent as JSON after WebSocket connection is established.
    """

    type: Literal["config"] = "config"
    model: str = Field(..., description="Model ID to use for inference")
    system_prompt: str | None = Field(
        default=None, description="System prompt for the conversation"
    )
    messages: list[dict] | None = Field(
        default=None, description="Previous conversation history (text only)"
    )
    max_tokens: int = Field(default=512, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, description="Sampling temperature")


class AudioDataMessage(BaseModel):
    """Audio data message containing PCM audio.

    The actual audio bytes are sent as binary WebSocket frames,
    not in this message. This is just for signaling.
    """

    type: Literal["audio"] = "audio"
    format: Literal["pcm", "wav"] = Field(
        default="pcm", description="Audio format"
    )
    sample_rate: int = Field(default=16000, description="Sample rate in Hz")
    channels: int = Field(default=1, description="Number of audio channels")
    sample_width: int = Field(default=2, description="Bytes per sample (2 = 16-bit)")


class AudioCompleteMessage(BaseModel):
    """Signal that audio upload is complete and inference should begin."""

    type: Literal["audio_complete"] = "audio_complete"


# Server -> Client messages


class TextChunkMessage(BaseModel):
    """Text chunk from LLM response."""

    type: Literal["text"] = "text"
    content: str
    is_final: bool = False


class ErrorMessage(BaseModel):
    """Error message."""

    type: Literal["error"] = "error"
    message: str
    code: str | None = None


class DoneMessage(BaseModel):
    """Session complete message."""

    type: Literal["done"] = "done"
    total_tokens: int | None = None
