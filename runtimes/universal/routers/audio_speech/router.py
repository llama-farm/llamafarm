"""FastAPI router for text-to-speech endpoints."""

import logging

from fastapi import APIRouter, WebSocket
from fastapi.responses import Response, StreamingResponse

from .service import TTSSynthesisService
from .types import SpeechRequest, VoiceListResponse

router = APIRouter(tags=["audio"])
logger = logging.getLogger(__name__)


def _get_service() -> TTSSynthesisService:
    """Get TTS service with load function from server."""
    # Import here to avoid circular import
    from server import load_tts

    return TTSSynthesisService(load_tts)


@router.post("/v1/audio/speech")
async def create_speech(request: SpeechRequest):
    """Generate audio from the input text.

    OpenAI-compatible text-to-speech endpoint.

    Reference: https://platform.openai.com/docs/api-reference/audio/createSpeech
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"TTS request: model={request.model}, voice={request.voice}, "
                     f"format={request.response_format}, stream={request.stream}")

    service = _get_service()

    if request.stream:
        # Streaming response via SSE
        return StreamingResponse(
            service.synthesize_stream_sse(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Non-streaming: return complete audio
    audio_bytes, content_type = await service.synthesize(request)

    return Response(
        content=audio_bytes,
        media_type=content_type,
        headers={
            "Content-Disposition": (
                f'attachment; filename="speech.{request.response_format}"'
            ),
        },
    )


@router.get("/v1/audio/voices", response_model=VoiceListResponse)
async def list_voices(model: str | None = None):
    """List available TTS voices.

    Args:
        model: Filter by model ID (optional).

    Returns:
        List of available voices with metadata.
    """
    service = _get_service()
    voices = service.list_voices(model)
    return {"object": "list", "data": voices}


@router.websocket("/v1/audio/speech/stream")
async def websocket_speech(
    websocket: WebSocket,
    model: str = "kokoro",
    voice: str = "af_heart",
    response_format: str = "pcm",
    sample_rate: int = 24000,
):
    """WebSocket endpoint for real-time TTS streaming.

    Protocol:
    1. Connect with query params (model, voice, format, sample_rate)
    2. Send JSON: {"text": "Hello world", "final": false}
    3. Receive binary audio chunks as they're generated
    4. Send {"text": "", "final": true} to close session

    Response messages:
    - Binary: Raw PCM audio bytes (16-bit signed, mono)
    - JSON: {"type": "done", "duration": 2.5} when synthesis complete
    - JSON: {"type": "error", "message": "..."} on error
    - JSON: {"type": "closed"} when session ends
    """
    service = _get_service()
    await service.handle_websocket(
        websocket=websocket,
        model_id=model,
        voice=voice,
        response_format=response_format,
        sample_rate=sample_rate,
    )
