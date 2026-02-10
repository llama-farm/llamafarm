"""Audio router for speech-to-text transcription and translation endpoints.

This router provides OpenAI-compatible endpoints for:
- Audio transcription (speech-to-text)
- Audio translation (speech-to-English text)
- Streaming transcription via WebSocket
"""

import json
import logging
import tempfile
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
)

logger = logging.getLogger(__name__)

# Router for audio endpoints
router = APIRouter(tags=["audio"])

# =============================================================================
# Constants
# =============================================================================

# Safe audio file extensions (whitelist for security)
SAFE_AUDIO_EXTENSIONS = frozenset(
    {
        ".wav",
        ".mp3",
        ".m4a",
        ".webm",
        ".flac",
        ".ogg",
        ".mp4",
        ".opus",
    }
)

# =============================================================================
# Dependency Injection
# =============================================================================

# Injected speech loader function
_load_speech_fn: Callable[..., Coroutine[Any, Any, Any]] | None = None


def set_speech_loader(load_speech_fn: Callable[..., Coroutine[Any, Any, Any]] | None):
    """Set the speech model loading function.

    The function should have signature: async def load_speech(model_id, compute_type) -> SpeechModel
    """
    global _load_speech_fn
    _load_speech_fn = load_speech_fn


def _get_speech_loader():
    """Get the speech loader, raising error if not initialized."""
    if _load_speech_fn is None:
        raise HTTPException(
            status_code=500,
            detail="Speech loader not initialized. Server configuration error.",
        )
    return _load_speech_fn


# =============================================================================
# Helper Functions
# =============================================================================


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


# =============================================================================
# Transcription Endpoint
# =============================================================================


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = None,
    model: str = Form(default="distil-large-v3"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
    timestamp_granularities: str | None = Form(default=None),
    stream: bool = Form(default=False),
):
    """
    OpenAI-compatible audio transcription endpoint.

    Transcribe audio files to text using Whisper models. Supports multiple
    model sizes, languages, and output formats.

    **Supported audio formats:** mp3, wav, m4a, webm, flac, ogg, mp4

    **Model sizes:**
    - tiny, base, small: Fast, lower accuracy
    - medium: Good balance of speed and accuracy
    - large-v3: Best accuracy, slower
    - distil-large-v3: Near large-v3 accuracy, much faster (recommended)

    **Streaming:** Set `stream=true` to receive transcription segments via SSE
    as they're processed, rather than waiting for the complete transcription.

    Example with curl:
    ```bash
    curl -X POST http://localhost:11540/v1/audio/transcriptions \\
        -F "file=@audio.mp3" \\
        -F "model=distil-large-v3" \\
        -F "language=en" \\
        -F "response_format=json"
    ```

    Example streaming:
    ```bash
    curl -X POST http://localhost:11540/v1/audio/transcriptions \\
        -F "file=@audio.mp3" \\
        -F "model=distil-large-v3" \\
        -F "stream=true"
    ```
    """
    from fastapi.responses import StreamingResponse

    try:
        # Get audio content from file upload
        audio_bytes: bytes | None = None
        file_extension = ".wav"

        if file is not None:
            audio_bytes = await file.read()
            if file.filename:
                # Sanitize file extension against whitelist
                ext = Path(file.filename).suffix.lower()
                file_extension = ext if ext in SAFE_AUDIO_EXTENSIONS else ".wav"
        else:
            raise HTTPException(
                status_code=400,
                detail="Audio file is required. Upload via 'file' field.",
            )

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="Empty audio file",
            )

        # Try to decode compressed audio to WAV for reliable processing
        try:
            from utils.audio_buffer import (
                decode_audio_bytes,
                detect_audio_format,
                pcm_to_wav,
            )

            format_name, is_compressed = detect_audio_format(audio_bytes)
            logger.debug(
                f"Detected audio format: {format_name} (compressed={is_compressed})"
            )

            if is_compressed:
                try:
                    pcm_data = decode_audio_bytes(audio_bytes)
                    audio_bytes = pcm_to_wav(pcm_data)
                    file_extension = ".wav"
                    logger.debug(
                        f"Decoded {format_name} to WAV ({len(audio_bytes)} bytes)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to decode {format_name}: {e}, using original data"
                    )
        except ImportError:
            # Audio buffer utilities not available, use raw audio
            pass

        # Load speech model
        load_speech = _get_speech_loader()
        speech_model = await load_speech(model_id=model)
        if speech_model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load speech model",
            )

        # Parse timestamp granularities
        word_timestamps = False
        if timestamp_granularities:
            granularities = [g.strip() for g in timestamp_granularities.split(",")]
            word_timestamps = "word" in granularities

        # Write audio to temp file (faster-whisper requires file path)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            if stream:
                # Streaming response - yield segments as they're transcribed
                async def generate_sse():
                    async for segment in speech_model.transcribe_stream(
                        audio_path=tmp_path,
                        language=language,
                        word_timestamps=word_timestamps,
                        initial_prompt=prompt,
                    ):
                        segment_data = {
                            "id": segment.id,
                            "start": segment.start,
                            "end": segment.end,
                            "text": segment.text,
                        }
                        if segment.words:
                            segment_data["words"] = segment.words

                        yield f"data: {json.dumps(segment_data)}\n\n"

                    yield "data: [DONE]\n\n"

                # Use BackgroundTasks to ensure temp file cleanup even on client disconnect
                background_tasks.add_task(Path(tmp_path).unlink, missing_ok=True)

                return StreamingResponse(
                    generate_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                    background=background_tasks,
                )

            # Non-streaming response
            result = await speech_model.transcribe(
                audio_path=tmp_path,
                language=language,
                word_timestamps=word_timestamps,
                initial_prompt=prompt,
                temperature=[temperature]
                if temperature > 0
                else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )

            # Format response based on requested format
            if response_format == "text":
                return result.text

            if response_format == "srt":
                # Generate SRT subtitle format
                srt_lines = []
                for i, seg in enumerate(result.segments, 1):
                    start_time = _format_timestamp_srt(seg.start)
                    end_time = _format_timestamp_srt(seg.end)
                    srt_lines.append(f"{i}")
                    srt_lines.append(f"{start_time} --> {end_time}")
                    srt_lines.append(seg.text.strip())
                    srt_lines.append("")
                return "\n".join(srt_lines)

            if response_format == "vtt":
                # Generate WebVTT subtitle format
                vtt_lines = ["WEBVTT", ""]
                for seg in result.segments:
                    start_time = _format_timestamp_vtt(seg.start)
                    end_time = _format_timestamp_vtt(seg.end)
                    vtt_lines.append(f"{start_time} --> {end_time}")
                    vtt_lines.append(seg.text.strip())
                    vtt_lines.append("")
                return "\n".join(vtt_lines)

            if response_format == "verbose_json":
                # Detailed JSON with segments
                return {
                    "task": "transcribe",
                    "language": result.language,
                    "duration": result.duration,
                    "text": result.text,
                    "segments": [
                        {
                            "id": seg.id,
                            "start": seg.start,
                            "end": seg.end,
                            "text": seg.text,
                            "words": seg.words,
                            "avg_logprob": seg.avg_logprob,
                            "no_speech_prob": seg.no_speech_prob,
                        }
                        for seg in result.segments
                    ],
                }

            # Default: simple JSON
            return {
                "text": result.text,
            }

        finally:
            # Clean up temp file (if not streaming)
            if not stream and tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"Speech model dependencies not installed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Speech-to-text not available. Install with: uv pip install 'universal-runtime[speech]'. Error: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Error in create_transcription: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# Translation Endpoint
# =============================================================================


@router.post("/v1/audio/translations")
async def create_translation(
    file: UploadFile,
    model: str = Form(default="distil-large-v3"),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
):
    """
    OpenAI-compatible audio translation endpoint.

    Translate audio to English text. Works the same as transcription but
    always outputs English regardless of the input language.

    Example:
    ```bash
    curl -X POST http://localhost:11540/v1/audio/translations \\
        -F "file=@french_audio.mp3" \\
        -F "model=distil-large-v3"
    ```
    """
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        file_extension = Path(file.filename).suffix if file.filename else ".wav"

        # Try to decode compressed audio
        try:
            from utils.audio_buffer import (
                decode_audio_bytes,
                detect_audio_format,
                pcm_to_wav,
            )

            format_name, is_compressed = detect_audio_format(audio_bytes)
            logger.debug(
                f"Detected audio format: {format_name} (compressed={is_compressed})"
            )

            if is_compressed:
                try:
                    pcm_data = decode_audio_bytes(audio_bytes)
                    audio_bytes = pcm_to_wav(pcm_data)
                    file_extension = ".wav"
                    logger.debug(
                        f"Decoded {format_name} to WAV ({len(audio_bytes)} bytes)"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to decode {format_name}: {e}, using original data"
                    )
        except ImportError:
            pass

        # Load speech model
        load_speech = _get_speech_loader()
        speech_model = await load_speech(model_id=model)
        if speech_model is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to load speech model",
            )

        # Write to temp file
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=file_extension, delete=False
            ) as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(audio_bytes)

            # Transcribe with translation task
            result = await speech_model.transcribe(
                audio_path=tmp_path,
                task="translate",  # Translate to English
                initial_prompt=prompt,
                temperature=[temperature]
                if temperature > 0
                else [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            )

            if response_format == "text":
                return result.text

            return {
                "text": result.text,
            }

        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    except HTTPException:
        raise
    except ImportError as e:
        logger.error(f"Speech model dependencies not installed: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Speech-to-text not available. Install with: uv pip install 'universal-runtime[speech]'. Error: {e}",
        ) from e
    except Exception as e:
        logger.error(f"Error in create_translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) from e


# =============================================================================
# WebSocket Streaming Endpoint
# =============================================================================


@router.websocket("/v1/audio/transcriptions/stream")
async def websocket_transcription(
    websocket: WebSocket,
    model: str = "base",
    language: str | None = None,
    word_timestamps: bool = False,
    chunk_interval: float = 2.0,
):
    """
    WebSocket endpoint for real-time audio streaming transcription.

    Connect via WebSocket and send audio chunks to receive live transcription.
    Audio should be sent as binary messages (raw PCM: 16kHz, 16-bit, mono).

    **IMPORTANT - Model Selection for Real-Time:**
    For real-time on CPU, use small models:
    - "tiny": ~0.5s to process 2s audio - fastest, lower accuracy
    - "base": ~1-2s to process 2s audio - good balance (DEFAULT)
    - "small": ~3-4s to process 2s audio - better accuracy

    Connection parameters:
    - model: Whisper model size (default: "base")
    - language: Force specific language detection
    - word_timestamps: Include word-level timing
    - chunk_interval: Seconds between processing chunks (default: 2.0)

    Protocol:
    1. Connect to WebSocket
    2. Send audio as binary messages
    3. Receive JSON transcription segments
    4. Send "END" text message to finish
    5. Connection closes after final segment

    Example JavaScript:
    ```javascript
    const ws = new WebSocket('ws://localhost:11540/v1/audio/transcriptions/stream');
    ws.onmessage = (event) => {
        const segment = JSON.parse(event.data);
        console.log(segment.text);
    };
    // Send audio chunks from microphone
    ```
    """
    await websocket.accept()

    try:
        # Load speech model
        load_speech = _get_speech_loader()
        speech_model = await load_speech(model_id=model)

        if speech_model is None:
            await websocket.send_json({"type": "error", "message": "Failed to load speech model"})
            await websocket.close(code=1011)
            return

        # Audio buffer for accumulating chunks
        audio_buffer = bytearray()
        sample_rate = 16000
        bytes_per_second = sample_rate * 2  # 16-bit mono

        # Process audio chunks
        while True:
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            if "bytes" in message:
                audio_buffer.extend(message["bytes"])

                # Process when we have enough audio
                buffer_duration = len(audio_buffer) / bytes_per_second
                if buffer_duration >= chunk_interval:
                    # Write buffer to temp file and transcribe
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp_file:
                        tmp_path = tmp_file.name
                        # Write WAV header + PCM data
                        from utils.audio_buffer import pcm_to_wav

                        wav_data = pcm_to_wav(
                            bytes(audio_buffer), sample_rate=sample_rate
                        )
                        tmp_file.write(wav_data)

                    try:
                        result = await speech_model.transcribe(
                            audio_path=tmp_path,
                            language=language,
                            word_timestamps=word_timestamps,
                        )

                        # Send transcription result as segment
                        await websocket.send_json(
                            {
                                "type": "segment",
                                "text": result.text,
                                "duration": buffer_duration,
                                "is_final": False,
                            }
                        )
                    except Exception as transcribe_err:
                        # Handle errors gracefully - log and send empty segment
                        # This can happen when VAD removes all audio or audio is corrupted
                        logger.warning(
                            f"Transcription error (may be silence): {transcribe_err}"
                        )
                        # Send empty segment to indicate we processed but found no speech
                        await websocket.send_json(
                            {
                                "type": "segment",
                                "text": "",
                                "duration": buffer_duration,
                                "is_final": False,
                                "warning": "No speech detected",
                            }
                        )

                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

                    # Clear buffer
                    audio_buffer.clear()

            elif "text" in message and message["text"].upper() == "END":
                # Process remaining audio
                if audio_buffer:
                    with tempfile.NamedTemporaryFile(
                        suffix=".wav", delete=False
                    ) as tmp_file:
                        tmp_path = tmp_file.name
                        from utils.audio_buffer import pcm_to_wav

                        wav_data = pcm_to_wav(
                            bytes(audio_buffer), sample_rate=sample_rate
                        )
                        tmp_file.write(wav_data)

                    try:
                        result = await speech_model.transcribe(
                            audio_path=tmp_path,
                            language=language,
                            word_timestamps=word_timestamps,
                        )

                        # Send final segment
                        await websocket.send_json(
                            {
                                "type": "segment",
                                "text": result.text,
                                "duration": len(audio_buffer) / bytes_per_second,
                                "is_final": True,
                            }
                        )
                    except Exception as transcribe_err:
                        # Handle errors gracefully - log and send empty final segment
                        logger.warning(
                            f"Final transcription error (may be silence): {transcribe_err}"
                        )
                        await websocket.send_json(
                            {
                                "type": "segment",
                                "text": "",
                                "duration": len(audio_buffer) / bytes_per_second,
                                "is_final": True,
                                "warning": "No speech detected",
                            }
                        )

                    finally:
                        Path(tmp_path).unlink(missing_ok=True)

                # Signal completion
                await websocket.send_json({"type": "done"})
                break

        await websocket.close()

    except Exception as e:
        logger.error(f"WebSocket transcription error: {e}", exc_info=True)
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
            await websocket.close(code=1011)
        except Exception:
            pass
