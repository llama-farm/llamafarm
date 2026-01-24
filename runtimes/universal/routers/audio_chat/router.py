"""FastAPI router for audio chat WebSocket endpoint."""

import logging

from fastapi import APIRouter, WebSocket

from .service import AudioChatService

router = APIRouter(tags=["audio"])
logger = logging.getLogger(__name__)


def _get_service() -> AudioChatService:
    """Get audio chat service with load functions from server."""
    # Import here to avoid circular import
    from server import load_language, load_speech

    return AudioChatService(
        load_language=load_language,
        load_speech=load_speech,
    )


@router.websocket("/v1/audio/chat")
async def websocket_audio_chat(websocket: WebSocket):
    """WebSocket endpoint for audio-to-LLM chat.

    Efficient binary protocol for voice applications - accepts raw PCM audio
    and streams text responses. No base64 encoding overhead.

    Protocol:
    1. Connect to WebSocket
    2. Send JSON config: {"type": "config", "model": "...", "system_prompt": "...", ...}
    3. Optionally send JSON audio format: {"type": "audio", "format": "pcm", "sample_rate": 16000}
    4. Send binary PCM audio frames (16-bit signed, mono, 16kHz)
    5. Send JSON: {"type": "audio_complete"} when done sending audio
    6. Receive JSON text chunks: {"type": "text", "content": "...", "is_final": false}
    7. Receive JSON done: {"type": "done", "total_tokens": 123}

    Config message fields:
    - model (required): Model ID to use
    - system_prompt: System prompt for the conversation
    - messages: Previous conversation history (text only)
    - max_tokens: Maximum tokens to generate (default: 512)
    - temperature: Sampling temperature (default: 0.7)

    Flow: Audio is transcribed via STT (Whisper), then sent to the LLM.

    Example client code:
    ```python
    import asyncio
    import websockets
    import json

    async def audio_chat(audio_bytes: bytes):
        async with websockets.connect("ws://localhost:9000/v1/audio/chat") as ws:
            # Send config
            await ws.send(json.dumps({
                "type": "config",
                "model": "qwen-omni",
                "system_prompt": "You are a helpful assistant.",
            }))

            # Send audio format (optional, defaults to PCM 16kHz mono)
            await ws.send(json.dumps({
                "type": "audio",
                "format": "pcm",
                "sample_rate": 16000,
            }))

            # Send raw PCM audio bytes
            await ws.send(audio_bytes)

            # Signal audio complete
            await ws.send(json.dumps({"type": "audio_complete"}))

            # Receive text stream
            response = ""
            async for msg in ws:
                data = json.loads(msg)
                if data["type"] == "text":
                    response += data["content"]
                    print(data["content"], end="", flush=True)
                elif data["type"] == "done":
                    print(f"\\nDone! {data['total_tokens']} tokens")
                    break
                elif data["type"] == "error":
                    print(f"Error: {data['message']}")
                    break

            return response
    ```
    """
    service = _get_service()
    await service.handle_websocket(websocket)
