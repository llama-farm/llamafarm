"""Business logic for text-to-speech synthesis."""

import asyncio
import base64
import contextlib
import json
import logging
from collections.abc import AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

from models.tts_model import DEFAULT_SAMPLE_RATE
from utils.audio_encoder import encode_audio

from .types import SpeechRequest

logger = logging.getLogger(__name__)


class TTSSynthesisService:
    """Service for text-to-speech synthesis operations."""

    def __init__(self, load_tts_func):
        """Initialize the TTS service.

        Args:
            load_tts_func: Function to load TTS models (from server.py).
        """
        self._load_tts = load_tts_func

    async def synthesize(self, request: SpeechRequest) -> tuple[bytes, str]:
        """Synthesize speech from text.

        Args:
            request: Speech synthesis request.

        Returns:
            Tuple of (audio_bytes, content_type).
        """
        # Load the TTS model
        model = await self._load_tts(
            model_id=request.model,
            voice=request.voice,
        )

        # Synthesize audio
        result = await model.synthesize(
            text=request.input,
            voice=request.voice,
            speed=request.speed,
        )

        # Encode to requested format
        audio_bytes, content_type = encode_audio(
            pcm_data=result.audio,
            format=request.response_format,
            sample_rate=result.sample_rate,
        )

        return audio_bytes, content_type

    async def synthesize_stream_sse(
        self,
        request: SpeechRequest,
    ) -> AsyncGenerator[str, None]:
        """Stream synthesized audio as Server-Sent Events.

        Yields audio chunks as base64-encoded SSE events for low-latency playback.

        Args:
            request: Speech synthesis request.

        Yields:
            SSE-formatted strings containing audio chunk events.
        """
        # Load the TTS model
        model = await self._load_tts(
            model_id=request.model,
            voice=request.voice,
        )

        total_samples = 0

        try:
            async for chunk in model.synthesize_stream(
                text=request.input,
                voice=request.voice,
                speed=request.speed,
            ):
                # Track samples for duration calculation
                # PCM is 16-bit (2 bytes per sample)
                total_samples += len(chunk) // 2

                # For streaming, we send raw PCM chunks regardless of requested format
                # The client can decode/convert as needed
                chunk_b64 = base64.b64encode(chunk).decode("ascii")

                event = {
                    "type": "audio",
                    "data": chunk_b64,
                    "format": "pcm",
                    "sample_rate": DEFAULT_SAMPLE_RATE,
                }
                yield f"data: {json.dumps(event)}\n\n"

                # Yield to event loop for responsive streaming
                await asyncio.sleep(0)

            # Send completion event
            duration = total_samples / DEFAULT_SAMPLE_RATE
            done_event = {
                "type": "done",
                "duration": duration,
            }
            yield f"data: {json.dumps(done_event)}\n\n"

        except Exception as e:
            logger.error(f"TTS streaming error: {e}", exc_info=True)
            error_event = {
                "type": "error",
                "message": "Speech synthesis failed",
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    async def handle_websocket(
        self,
        websocket: WebSocket,
        model_id: str,
        voice: str,
        response_format: str,
        sample_rate: int,
    ) -> None:
        """Handle WebSocket connection for real-time TTS streaming.

        Protocol:
        1. Client connects with query params (model, voice, format)
        2. Client sends JSON: {"text": "Hello", "final": false}
        3. Server sends binary audio chunks
        4. Client sends {"text": "", "final": true} to close

        Args:
            websocket: FastAPI WebSocket connection.
            model_id: TTS model to use.
            voice: Voice ID.
            response_format: Audio format for response.
            sample_rate: Output sample rate.
        """
        await websocket.accept()

        try:
            # Load the TTS model
            model = await self._load_tts(
                model_id=model_id,
                voice=voice,
            )

            while True:
                # Receive text from client
                message = await websocket.receive_json()

                text = message.get("text", "")
                is_final = message.get("final", False)

                if is_final and not text:
                    # Client signaling end of session
                    await websocket.send_json({"type": "closed"})
                    break

                if not text:
                    continue

                # Stream audio chunks back
                total_samples = 0
                async for chunk in model.synthesize_stream(
                    text=text,
                    voice=voice,
                    speed=message.get("speed", 1.0),
                ):
                    total_samples += len(chunk) // 2
                    await websocket.send_bytes(chunk)

                # Send completion marker
                duration = total_samples / DEFAULT_SAMPLE_RATE
                await websocket.send_json({
                    "type": "done",
                    "duration": duration,
                })

                if is_final:
                    logger.info("TTS WebSocket: final=True, closing connection")
                    break
                else:
                    logger.info("TTS WebSocket: final=False, waiting for next phrase")

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error(f"WebSocket TTS error: {e}")
            with contextlib.suppress(Exception):
                # Send generic error message to avoid exposing internal details
                await websocket.send_json({"type": "error", "message": "Speech synthesis failed"})

    def list_voices(self, model_id: str | None = None) -> list[dict]:
        """List available TTS voices.

        Args:
            model_id: Filter by model ID (optional).

        Returns:
            List of voice info dictionaries.
        """
        from models.tts_model import KOKORO_VOICES

        voices = []
        for voice_id, (name, lang_code) in KOKORO_VOICES.items():
            language = "en-US" if lang_code == "a" else "en-GB"
            voice_model = "kokoro"

            # Filter by model if specified
            if model_id and voice_model != model_id:
                continue

            voices.append({
                "id": voice_id,
                "name": name,
                "language": language,
                "model": voice_model,
                "preview_url": None,
            })

        return voices
