"""Audio chat WebSocket service.

Handles WebSocket connections for audio-to-LLM communication.
Accepts raw PCM audio bytes, transcribes via STT, and streams text responses.
"""

from __future__ import annotations

import contextlib
import io
import logging
import wave
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from .types import (
    AudioChatConfig,
    AudioDataMessage,
    DoneMessage,
    ErrorMessage,
    TextChunkMessage,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

# Audio constants
WHISPER_SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1  # Mono


class AudioChatService:
    """Service for handling audio chat WebSocket connections.

    Efficient binary protocol for voice applications:
    - Accepts raw PCM audio bytes (no base64 encoding overhead)
    - Transcribes via STT (Whisper)
    - Streams LLM text responses

    Protocol:
    1. Client connects to WebSocket
    2. Client sends JSON config message (model, system prompt, history)
    3. Client sends binary PCM audio frames
    4. Client sends JSON audio_complete message
    5. Server transcribes audio via STT
    6. Server streams back text chunks as JSON
    7. Server sends done message when complete
    """

    def __init__(
        self,
        load_language: Callable,
        load_speech: Callable,
    ):
        """Initialize audio chat service.

        Args:
            load_language: Function to load text LLM
            load_speech: Function to load STT model
        """
        self._load_language = load_language
        self._load_speech = load_speech

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Handle a WebSocket connection for audio chat.

        Args:
            websocket: FastAPI WebSocket connection
        """
        await websocket.accept()
        logger.info("Audio chat WebSocket connected")

        config: AudioChatConfig | None = None
        audio_buffer = bytearray()
        audio_format = AudioDataMessage()  # Defaults

        try:
            # Main message loop
            while True:
                message = await websocket.receive()

                if message["type"] == "websocket.disconnect":
                    logger.info("Audio chat WebSocket disconnected")
                    break

                # Handle binary audio data
                if "bytes" in message:
                    audio_buffer.extend(message["bytes"])
                    continue

                # Handle JSON messages
                if "text" in message:
                    import json

                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "config":
                        config = AudioChatConfig(**data)
                        logger.info(
                            f"Audio chat config: model={config.model}, "
                            f"max_tokens={config.max_tokens}"
                        )

                    elif msg_type == "audio":
                        audio_format = AudioDataMessage(**data)
                        logger.debug(
                            f"Audio format: {audio_format.format}, "
                            f"rate={audio_format.sample_rate}"
                        )

                    elif msg_type == "audio_complete":
                        if config is None:
                            await websocket.send_json(
                                ErrorMessage(
                                    message="Config message required before audio",
                                    code="missing_config",
                                ).model_dump()
                            )
                            continue

                        if not audio_buffer:
                            await websocket.send_json(
                                ErrorMessage(
                                    message="No audio data received",
                                    code="no_audio",
                                ).model_dump()
                            )
                            continue

                        # Process the audio and stream response
                        await self._process_audio(
                            websocket=websocket,
                            config=config,
                            audio_bytes=bytes(audio_buffer),
                            audio_format=audio_format,
                        )

                        # Clear buffer for next turn
                        audio_buffer.clear()

                    else:
                        logger.warning(f"Unknown message type: {msg_type}")

        except WebSocketDisconnect:
            logger.info("Audio chat WebSocket disconnected")
        except Exception as e:
            logger.error(f"Audio chat WebSocket error: {e}", exc_info=True)
            with contextlib.suppress(Exception):
                await websocket.send_json(
                    ErrorMessage(message=str(e), code="internal_error").model_dump()
                )

    async def _process_audio(
        self,
        websocket: WebSocket,
        config: AudioChatConfig,
        audio_bytes: bytes,
        audio_format: AudioDataMessage,
    ) -> None:
        """Process audio and stream LLM response.

        Flow: Audio -> STT -> Text -> LLM -> Streaming text

        Args:
            websocket: WebSocket to send responses on
            config: Chat configuration
            audio_bytes: Raw PCM audio bytes
            audio_format: Audio format info
        """
        model_id = config.model

        logger.info(f"Processing {len(audio_bytes)} bytes of audio for model {model_id}")

        # Convert PCM to WAV for STT
        if audio_format.format == "pcm":
            audio_bytes = self._pcm_to_wav(
                audio_bytes,
                sample_rate=audio_format.sample_rate,
                channels=audio_format.channels,
                sample_width=audio_format.sample_width,
            )

        # Step 1: Transcribe audio via STT
        stt_model = await self._load_speech()
        result = await stt_model.transcribe_audio(audio_bytes)
        transcription = result.get("text", "").strip()

        if not transcription:
            await websocket.send_json(
                ErrorMessage(
                    message="Empty transcription",
                    code="empty_transcription",
                ).model_dump()
            )
            return

        logger.info(f"Transcribed: '{transcription[:100]}{'...' if len(transcription) > 100 else ''}'")

        # Build messages for the model
        messages = []

        # Add system prompt if provided
        if config.system_prompt:
            messages.append({"role": "system", "content": config.system_prompt})

        # Add history if provided
        if config.messages:
            messages.extend(config.messages)

        # Add transcribed text as user message
        messages.append({"role": "user", "content": transcription})

        # Step 2: Load LLM and stream response
        model = await self._load_language(model_id)

        token_count = 0
        async for token in model.generate_stream(
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        ):
            token_count += 1
            await websocket.send_json(
                TextChunkMessage(content=token, is_final=False).model_dump()
            )

        # Send done message
        await websocket.send_json(
            DoneMessage(total_tokens=token_count).model_dump()
        )

    def _pcm_to_wav(
        self,
        pcm_data: bytes,
        sample_rate: int = WHISPER_SAMPLE_RATE,
        channels: int = CHANNELS,
        sample_width: int = SAMPLE_WIDTH,
    ) -> bytes:
        """Convert raw PCM bytes to WAV format.

        Args:
            pcm_data: Raw PCM audio bytes
            sample_rate: Sample rate in Hz
            channels: Number of channels
            sample_width: Bytes per sample

        Returns:
            WAV file bytes
        """
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)
        return wav_buffer.getvalue()
