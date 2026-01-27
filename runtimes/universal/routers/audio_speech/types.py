"""Pydantic models for text-to-speech API."""

from typing import Literal

from pydantic import BaseModel, Field


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request.

    Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """

    model: str = Field(
        default="kokoro",
        description="TTS model to use. Currently supports 'kokoro'.",
    )
    input: str = Field(
        ...,
        description="The text to synthesize into speech. Maximum 4096 characters.",
        max_length=4096,
    )
    voice: str = Field(
        default="af_heart",
        description="Voice ID to use for synthesis.",
    )
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = Field(
        default="mp3",
        description="Audio output format.",
    )
    speed: float = Field(
        default=1.0,
        ge=0.25,
        le=4.0,
        description="Speed of generated audio. 0.25 to 4.0.",
    )
    # LlamaFarm extension
    stream: bool = Field(
        default=False,
        description="Enable streaming response via Server-Sent Events.",
    )


class VoiceInfo(BaseModel):
    """Information about an available TTS voice."""

    id: str = Field(description="Unique voice identifier.")
    name: str = Field(description="Human-readable voice name.")
    language: str = Field(description="Language code (e.g., 'en-US', 'en-GB').")
    model: str = Field(description="Model this voice belongs to.")
    preview_url: str | None = Field(
        default=None,
        description="URL to a preview audio sample (if available).",
    )


class VoiceListResponse(BaseModel):
    """Response from the voices list endpoint."""

    object: Literal["list"] = "list"
    data: list[VoiceInfo]


class AudioChunkEvent(BaseModel):
    """Server-Sent Event for streaming audio chunk."""

    type: Literal["audio"] = "audio"
    data: str = Field(description="Base64-encoded audio chunk.")
    format: str = Field(description="Audio format of the chunk (e.g., 'pcm').")


class AudioDoneEvent(BaseModel):
    """Server-Sent Event indicating synthesis completion."""

    type: Literal["done"] = "done"
    duration: float = Field(description="Total duration of generated audio in seconds.")


class AudioErrorEvent(BaseModel):
    """Server-Sent Event indicating an error."""

    type: Literal["error"] = "error"
    message: str = Field(description="Error message.")
