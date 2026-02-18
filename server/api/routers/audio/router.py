"""
Audio Router - Text-to-Speech endpoints.

Provides project-scoped access to TTS functionality:
- Speech synthesis (text to audio)
- Voice listing

All endpoints are OpenAI-compatible where applicable.
"""

import logging
from typing import Literal

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from services.universal_runtime_service import UniversalRuntimeService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["audio"])


# =============================================================================
# Request/Response Models
# =============================================================================


class SpeechRequest(BaseModel):
    """OpenAI-compatible speech synthesis request."""

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


# =============================================================================
# Content type mapping
# =============================================================================

AUDIO_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "wav": "audio/wav",
    "pcm": "audio/pcm",
}

# Reverse mapping: content type -> format name
CONTENT_TYPE_TO_FORMAT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/opus": "opus",
    "audio/aac": "aac",
    "audio/flac": "flac",
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/pcm": "pcm",
    "audio/L16": "pcm",
}


def _parse_accept_header(accept: str | None) -> str | None:
    """Parse Accept header to determine requested audio format.

    Args:
        accept: Accept header value (e.g., "audio/pcm", "audio/mpeg, audio/*;q=0.5")

    Returns:
        Format name (mp3, pcm, etc.) or None if no supported format found.
    """
    if not accept:
        return None

    # Parse Accept header (simplified - handles basic cases)
    # Format: type/subtype;q=weight, type/subtype;q=weight, ...
    best_format = None
    best_quality = -1.0

    for part in accept.split(","):
        part = part.strip()
        if not part:
            continue

        # Split off quality parameter
        if ";" in part:
            media_type, params = part.split(";", 1)
            media_type = media_type.strip()
            # Extract q value
            quality = 1.0
            for param in params.split(";"):
                param = param.strip()
                if param.startswith("q="):
                    try:
                        quality = float(param[2:])
                    except ValueError:
                        quality = 1.0
        else:
            media_type = part
            quality = 1.0

        # Skip wildcards and non-audio types
        if media_type == "*/*" or media_type == "audio/*":
            continue

        # Check if this is a supported audio type
        fmt = CONTENT_TYPE_TO_FORMAT.get(media_type.lower())
        if fmt and quality > best_quality:
            best_format = fmt
            best_quality = quality

    return best_format


# =============================================================================
# Endpoints
# =============================================================================


@router.post("/{namespace}/{project}/audio/speech")
async def create_speech(
    namespace: str,
    project: str,
    request: SpeechRequest,
    accept: str | None = Header(default=None),
):
    """Generate audio from the input text.

    OpenAI-compatible text-to-speech endpoint.

    Path Parameters:
        namespace: Project namespace
        project: Project name

    Headers:
        Accept: Preferred audio format (e.g., "audio/pcm", "audio/mpeg").
            Overrides response_format in body if provided.

    Request Body:
        model: TTS model ID (default: "kokoro")
        input: Text to synthesize (max 4096 characters)
        voice: Voice ID (default: "af_heart")
        response_format: Audio format - mp3, opus, aac, flac, wav, pcm (default: "mp3")
        speed: Speech speed 0.25-4.0 (default: 1.0)

    Returns:
        Audio file in the requested format

    Example:
        ```bash
        # Using response_format in body
        curl -X POST "http://localhost:14345/v1/default/myproject/audio/speech" \\
          -H "Content-Type: application/json" \\
          -d '{"input": "Hello, world!", "voice": "af_heart"}' \\
          --output speech.mp3

        # Using Accept header
        curl -X POST "http://localhost:14345/v1/default/myproject/audio/speech" \\
          -H "Content-Type: application/json" \\
          -H "Accept: audio/pcm" \\
          -d '{"input": "Hello, world!"}' \\
          --output speech.pcm
        ```
    """
    # Accept header overrides response_format in body
    accept_format = _parse_accept_header(accept)
    response_format = accept_format if accept_format else request.response_format

    logger.debug(
        f"TTS request for {namespace}/{project}: "
        f"model={request.model}, voice={request.voice}, format={response_format}"
        f"{' (from Accept header)' if accept_format else ''}"
    )

    if not request.input.strip():
        raise HTTPException(
            status_code=400,
            detail="Input text cannot be empty",
        )

    audio_bytes = await UniversalRuntimeService.synthesize_speech(
        text=request.input,
        model=request.model,
        voice=request.voice,
        response_format=response_format,
        speed=request.speed,
    )

    content_type = AUDIO_CONTENT_TYPES.get(response_format, "audio/mpeg")

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{response_format}"',
        },
    )


@router.get(
    "/{namespace}/{project}/audio/voices",
    response_model=VoiceListResponse,
)
async def list_voices(
    namespace: str,
    project: str,
    model: str | None = None,
):
    """List available TTS voices.

    Path Parameters:
        namespace: Project namespace
        project: Project name

    Query Parameters:
        model: Filter by model ID (optional)

    Returns:
        List of available voices with metadata

    Example:
        ```bash
        curl "http://localhost:14345/v1/default/myproject/audio/voices"
        ```
    """
    result = await UniversalRuntimeService.list_tts_voices(model)
    return result
