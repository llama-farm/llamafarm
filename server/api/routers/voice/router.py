"""
FastAPI router for real-time voice chat WebSocket endpoint.

Provides full-duplex voice assistant functionality:
- Speech In → STT → LLM → TTS → Speech Out
- Stateful sessions with conversation history
- Barge-in support (interrupt TTS when user speaks)

Configuration:
- Voice settings can be configured in llamafarm.yaml under the `voice` section
- Query parameters override config defaults
- If no project is specified, hardcoded defaults are used
"""

import asyncio
import contextlib
import json
import logging

import httpx
from config.datamodel import LlamaFarmConfig
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.settings import settings
from services.model_service import ModelService
from services.project_service import ProjectService
from services.prompt_service import PromptService

from .service import VoiceChatService
from .session import get_session_manager
from .types import (
    ClosedMessage,
    ErrorMessage,
    SessionInfoMessage,
    StatusMessage,
    VoiceSessionConfig,
    VoiceState,
)

router = APIRouter(tags=["voice"])
logger = logging.getLogger(__name__)


# Security: Maximum length for user-provided text inputs
MAX_SYSTEM_PROMPT_LENGTH = 10000  # 10KB max for system prompts


def _sanitize_system_prompt(prompt: str | None) -> str | None:
    """Sanitize user-provided system prompt.

    Validates and sanitizes the system prompt to prevent injection attacks
    and excessive resource usage.

    Args:
        prompt: Raw system prompt from URL parameter.

    Returns:
        Sanitized prompt or None if invalid/empty.
    """
    if prompt is None:
        return None

    # Strip whitespace
    prompt = prompt.strip()
    if not prompt:
        return None

    # Enforce maximum length to prevent DoS
    if len(prompt) > MAX_SYSTEM_PROMPT_LENGTH:
        logger.warning(
            f"System prompt truncated from {len(prompt)} to {MAX_SYSTEM_PROMPT_LENGTH} chars"
        )
        prompt = prompt[:MAX_SYSTEM_PROMPT_LENGTH]

    # Remove null bytes and other control characters that could cause issues
    # Keep newlines and tabs as they're valid in prompts
    prompt = "".join(
        char for char in prompt
        if char == "\n" or char == "\t" or (ord(char) >= 32 and ord(char) != 127)
    )

    return prompt if prompt else None

# Cache for model capabilities (avoid querying runtime repeatedly)
_model_capabilities_cache: dict[str, dict] = {}

# Cache for available TTS models (refreshed periodically)
_available_tts_models: list[str] = []
_tts_models_cache_time: float = 0
TTS_CACHE_TTL = 60.0  # Refresh TTS model list every 60 seconds


# Default values when no config is provided
DEFAULT_STT_MODEL = "base"
DEFAULT_TTS_MODEL = "kokoro"
DEFAULT_TTS_VOICE = "af_heart"
DEFAULT_LANGUAGE = "en"
DEFAULT_SPEED = 0.95  # Slightly slower for more natural speech


async def _get_available_tts_models() -> list[str]:
    """Get list of TTS models available on the runtime.

    Queries the runtime's /v1/models endpoint and filters for TTS models.
    Results are cached for TTS_CACHE_TTL seconds.

    Returns:
        List of available TTS model IDs (e.g., ["kokoro", "pocket-tts"]).
    """
    import time

    global _available_tts_models, _tts_models_cache_time

    # Return cached result if fresh
    if _available_tts_models and (time.time() - _tts_models_cache_time) < TTS_CACHE_TTL:
        return _available_tts_models

    runtime_url = f"http://{settings.universal_host}:{settings.universal_port}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{runtime_url}/v1/models")
            if response.status_code == 200:
                data = response.json()
                models = data.get("data", [])
                # Filter for TTS models and extract the model name
                # Model IDs are like "tts:kokoro:af_heart" or "tts:pocket-tts:alba"
                tts_models = set()
                for model in models:
                    model_id = model.get("id", "")
                    model_type = model.get("type", "")
                    if model_type == "tts" and model_id.startswith("tts:"):
                        # Extract model name from "tts:model_name:voice"
                        parts = model_id.split(":")
                        if len(parts) >= 2:
                            tts_models.add(parts[1])
                _available_tts_models = sorted(tts_models)
                _tts_models_cache_time = time.time()
                logger.debug(f"Available TTS models: {_available_tts_models}")
                return _available_tts_models
    except Exception as e:
        logger.debug(f"Failed to query available TTS models: {e}")

    return _available_tts_models  # Return stale cache on error


async def _check_model_native_audio(model_id: str, base_url: str | None = None) -> bool:
    """Check if a model supports native audio input by querying the runtime.

    Queries the runtime's capabilities endpoint to determine if the loaded
    model supports direct audio input (for models like Qwen2.5-Omni).

    Results are cached to avoid repeated queries.

    Args:
        model_id: The model identifier (e.g., "ggml-org/Qwen2.5-Omni-3B-GGUF")
        base_url: Optional base URL for the runtime. If None, uses default runtime.

    Returns:
        True if the model supports native audio input.
    """
    # Check cache first
    if model_id in _model_capabilities_cache:
        return _model_capabilities_cache[model_id].get("native_audio", False)

    # Build runtime URL
    if base_url:
        runtime_url = base_url.rstrip("/")
    else:
        runtime_url = f"http://{settings.universal_host}:{settings.universal_port}"

    # Query capabilities endpoint
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{runtime_url}/v1/models/{model_id}/capabilities"
            )
            if response.status_code == 200:
                data = response.json()
                capabilities = data.get("capabilities", {})
                # Cache the result
                _model_capabilities_cache[model_id] = capabilities
                native_audio = capabilities.get("native_audio", False)
                if native_audio:
                    logger.info(f"Runtime reports model supports native audio: {model_id}")
                return native_audio
            else:
                logger.debug(f"Capabilities query failed: {response.status_code}")
    except Exception as e:
        logger.debug(f"Failed to query model capabilities: {e}")

    # Fall back to name-based detection as last resort
    model_lower = model_id.lower()
    native_audio = "omni" in model_lower
    if native_audio:
        logger.info(f"Using name-based detection for native audio: {model_id}")
    return native_audio


def _get_voice_config_defaults(
    project_config: LlamaFarmConfig | None,
) -> dict:
    """Extract voice config defaults from project config.

    Returns a dict with keys for all voice session configuration options.
    """
    defaults = {
        "stt_model": DEFAULT_STT_MODEL,
        "tts_model": DEFAULT_TTS_MODEL,
        "tts_voice": DEFAULT_TTS_VOICE,
        "llm_model": "",
        "language": DEFAULT_LANGUAGE,
        "speed": DEFAULT_SPEED,
        "enable_thinking": False,  # Disabled by default for voice
        "sentence_boundary_only": True,  # Natural speech by default
        # Turn detection defaults (enabled by default)
        "turn_detection_enabled": True,
        "base_silence_duration": 0.4,
        "thinking_silence_duration": 1.2,
        "max_silence_duration": 2.5,
    }

    if project_config and hasattr(project_config, 'voice') and project_config.voice:
        voice = project_config.voice

        # LLM model from voice config
        if voice.llm_model:
            defaults["llm_model"] = voice.llm_model

        # Thinking mode (explicitly set in config to enable)
        if voice.enable_thinking is not None:
            defaults["enable_thinking"] = voice.enable_thinking

        # STT settings
        if voice.stt:
            if voice.stt.model:
                defaults["stt_model"] = voice.stt.model
            if voice.stt.language:
                defaults["language"] = voice.stt.language

        # TTS settings
        if voice.tts:
            if voice.tts.model:
                defaults["tts_model"] = voice.tts.model
            if voice.tts.voice:
                defaults["tts_voice"] = voice.tts.voice
            if voice.tts.speed is not None:
                defaults["speed"] = voice.tts.speed

        # Turn detection settings
        if voice.turn_detection:
            td = voice.turn_detection
            if td.enabled is not None:
                defaults["turn_detection_enabled"] = td.enabled
            if td.base_silence_duration is not None:
                defaults["base_silence_duration"] = td.base_silence_duration
            if td.thinking_silence_duration is not None:
                defaults["thinking_silence_duration"] = td.thinking_silence_duration
            if td.max_silence_duration is not None:
                defaults["max_silence_duration"] = td.max_silence_duration

    return defaults


@router.websocket("/v1/{namespace}/{project}/voice/chat")
async def voice_chat_websocket(
    websocket: WebSocket,
    # Project context (required) - from URL path
    namespace: str,
    project: str,
    # Session management
    session_id: str | None = None,
    # Voice settings - query params override config defaults
    stt_model: str | None = None,
    tts_model: str | None = None,
    tts_voice: str | None = None,
    llm_model: str | None = None,
    language: str | None = None,
    speed: float | None = None,
    system_prompt: str | None = None,
    sentence_boundary_only: bool | None = None,
):
    """Real-time voice chat WebSocket endpoint.

    Full-duplex voice assistant that:
    1. Receives audio from client
    2. Transcribes via STT
    3. Generates response via LLM
    4. Synthesizes speech via TTS
    5. Streams audio back to client

    Path Parameters:
        namespace: Project namespace
        project: Project name

    Query Parameters:
        session_id: Resume existing session (optional)
        stt_model: Whisper model size (tiny, base, small, medium, large-v3)
        tts_model: TTS model ID (default: kokoro)
        tts_voice: TTS voice ID (default: af_heart)
        llm_model: LLM model ID (required unless configured in llamafarm.yaml)
        language: STT language code (default: en)
        speed: TTS speed multiplier (0.5-2.0, default: 1.0)
        system_prompt: System prompt for LLM (optional)
        sentence_boundary_only: Only split on sentence endings (. ! ?) for natural speech.
            Set to false for aggressive chunking (lower latency but choppier). Default: true.

    Configuration:
        Voice settings are loaded from the project's llamafarm.yaml `voice`
        section. Query parameters override config defaults.

    Client → Server Messages:
        - Binary: Audio data (PCM 16kHz 16-bit mono, or WebM/Opus)
        - {"type": "interrupt"}: Stop current TTS (barge-in)
        - {"type": "end"}: Force processing (optional - VAD auto-detects end of speech)
        - {"type": "config", ...}: Update session settings

    Server → Client Messages:
        - {"type": "session_info", "session_id": "..."}: Session created/resumed
        - {"type": "transcription", "text": "...", "is_final": bool}: STT result
        - {"type": "llm_text", "text": "...", "is_final": bool}: LLM phrase
        - Binary: TTS audio chunks (PCM 24kHz 16-bit mono)
        - {"type": "tts_start", "phrase_index": N}: Phrase synthesis starting
        - {"type": "tts_done", "phrase_index": N, "duration": N}: Phrase complete
        - {"type": "status", "state": "..."}: Pipeline state change
        - {"type": "error", "message": "..."}: Error occurred
        - {"type": "closed"}: Session ended
    """
    await websocket.accept()

    # Load project config
    project_config: LlamaFarmConfig | None = None
    try:
        project_config = ProjectService.load_config(namespace, project)
        logger.info(
            f"Loaded voice config from project {namespace}/{project}",
            extra={"has_voice_config": hasattr(project_config, 'voice') and project_config.voice is not None},
        )
    except Exception as e:
        logger.warning(
            f"Failed to load project config {namespace}/{project}: {e}. "
            "Using default voice settings."
        )

    # Get defaults from config (or hardcoded defaults if no config)
    defaults = _get_voice_config_defaults(project_config)

    # Apply query param overrides (explicit values override config defaults)
    effective_stt_model = stt_model if stt_model is not None else defaults["stt_model"]
    effective_tts_model = tts_model if tts_model is not None else defaults["tts_model"]
    effective_tts_voice = tts_voice if tts_voice is not None else defaults["tts_voice"]
    effective_llm_model = llm_model if llm_model is not None else defaults["llm_model"]
    effective_language = language if language is not None else defaults["language"]
    effective_speed = speed if speed is not None else defaults["speed"]
    # Thinking is controlled by config only (no query param override)
    effective_enable_thinking = defaults["enable_thinking"]
    # Sentence boundary only - can be overridden via query param
    effective_sentence_boundary_only = (
        sentence_boundary_only if sentence_boundary_only is not None
        else defaults["sentence_boundary_only"]
    )

    # Turn detection settings (from config only, no query param override)
    effective_turn_detection_enabled = defaults["turn_detection_enabled"]
    effective_base_silence_duration = defaults["base_silence_duration"]
    effective_thinking_silence_duration = defaults["thinking_silence_duration"]
    effective_max_silence_duration = defaults["max_silence_duration"]

    # Validate required parameters
    if not effective_llm_model:
        await websocket.send_json(
            ErrorMessage(
                message="llm_model is required (via query param or voice.llm_model in config)"
            ).model_dump()
        )
        await websocket.close(code=1008, reason="Missing llm_model")
        return

    # Check if voice is explicitly disabled in config
    if project_config and hasattr(project_config, 'voice') and project_config.voice and project_config.voice.enabled is False:
        await websocket.send_json(
            ErrorMessage(
                message="Voice chat is disabled for this project (voice.enabled=false)"
            ).model_dump()
        )
        await websocket.close(code=1008, reason="Voice chat disabled")
        return

    # Resolve LLM model name to actual model configuration
    # This converts e.g., "conversational" to the actual model ID and base_url
    try:
        llm_model_config = ModelService.get_model(project_config, effective_llm_model)
        logger.debug(
            f"Resolved LLM model '{effective_llm_model}' to '{llm_model_config.model}'"
        )
    except ValueError as e:
        await websocket.send_json(
            ErrorMessage(message=f"Invalid LLM model: {e}").model_dump()
        )
        await websocket.close(code=1008, reason="Invalid LLM model")
        return

    # Validate TTS model is available on the runtime
    # This prevents cryptic errors when the configured TTS model isn't loaded
    available_tts = await _get_available_tts_models()
    if available_tts and effective_tts_model not in available_tts:
        # TTS model not available - provide helpful error with alternatives
        available_list = ", ".join(available_tts) if available_tts else "none loaded"
        error_msg = (
            f"TTS model '{effective_tts_model}' is not loaded on the runtime. "
            f"Available TTS models: {available_list}. "
            f"Configure voice.tts.model in your project's llamafarm.yaml or pass tts_model query param."
        )
        logger.warning(error_msg)
        await websocket.send_json(ErrorMessage(message=error_msg).model_dump())
        await websocket.close(code=1008, reason="TTS model not available")
        return

    # Detect if model supports native audio input (e.g., Qwen2.5-Omni)
    # Query the runtime to check capabilities (falls back to name-based detection)
    use_native_audio = await _check_model_native_audio(
        llm_model_config.model,
        base_url=llm_model_config.base_url,
    )
    if use_native_audio:
        logger.info(
            f"Native audio mode enabled for model: {llm_model_config.model} "
            "(STT will be skipped, audio sent directly to LLM)"
        )

    # Create or resume session
    session_manager = get_session_manager()
    # Don't pass system_prompt to config - we handle prompt injection manually
    # to ensure model prompts come first, then query param system_prompt is appended
    config = VoiceSessionConfig(
        session_id=session_id,
        stt_model=effective_stt_model,
        tts_model=effective_tts_model,
        tts_voice=effective_tts_voice,
        llm_model=effective_llm_model,
        language=effective_language,
        speed=effective_speed,
        system_prompt=None,  # Handled below
        enable_thinking=effective_enable_thinking,
        sentence_boundary_only=effective_sentence_boundary_only,
        # Turn detection settings
        turn_detection_enabled=effective_turn_detection_enabled,
        base_silence_duration=effective_base_silence_duration,
        thinking_silence_duration=effective_thinking_silence_duration,
        max_silence_duration=effective_max_silence_duration,
        # Native audio for Omni models
        use_native_audio=use_native_audio,
    )

    session = await session_manager.get_or_create_session(session_id, config)

    # Inject prompts if session is new (no messages yet)
    # Order: 1) Model config prompts, 2) Query param system_prompt
    if not session.messages:
        # First: inject prompts from model config (runtime.models[].prompts)
        if project_config:
            resolved_prompts = PromptService.resolve_prompts_for_model(
                project_config, llm_model_config
            )
            if resolved_prompts:
                for prompt_msg in resolved_prompts:
                    session.messages.append({
                        "role": prompt_msg.role,
                        "content": prompt_msg.content,
                    })
                logger.info(
                    f"Injected {len(resolved_prompts)} prompt messages from model config",
                    extra={"prompt_count": len(resolved_prompts)},
                )

        # Second: append system_prompt from query param (adds to model prompts)
        # Sanitize to prevent injection attacks and excessive resource usage
        sanitized_prompt = _sanitize_system_prompt(system_prompt)
        if sanitized_prompt:
            session.messages.append({
                "role": "system",
                "content": sanitized_prompt,
            })
            logger.info("Appended sanitized system_prompt from query param")

    service = VoiceChatService(session, llm_model_config)

    # Pre-warm connections in background (don't block session start)
    asyncio.create_task(service.warm_up())

    # Send session info
    await websocket.send_json(
        SessionInfoMessage(session_id=session.session_id).model_dump()
    )
    await websocket.send_json(
        StatusMessage(state=session.state).model_dump()
    )

    logger.info(
        f"Voice chat session started: {session.session_id}",
        extra={
            "llm_model_name": effective_llm_model,
            "llm_model_id": llm_model_config.model,
            "tts_voice": effective_tts_voice,
            "stt_model": effective_stt_model if not use_native_audio else "(native audio)",
            "native_audio": use_native_audio,
            "project": f"{namespace}/{project}" if namespace and project else None,
        },
    )

    # Track background task for process_turn so we can receive audio during TTS
    current_turn_task: asyncio.Task | None = None

    async def run_process_turn(audio_bytes: bytes, native_audio: bool) -> None:
        """Run process_turn in background, allowing receive loop to continue."""
        nonlocal current_turn_task
        try:
            if native_audio:
                await service.process_turn_native_audio(websocket, audio_bytes)
            else:
                await service.process_turn(websocket, audio_bytes)
        except asyncio.CancelledError:
            logger.info("process_turn cancelled by interrupt")
        except Exception as e:
            logger.error(f"process_turn error: {e}", exc_info=True)
        finally:
            current_turn_task = None

    try:
        while True:
            # Receive message from client
            message = await websocket.receive()

            if message["type"] == "websocket.disconnect":
                break

            # Handle binary audio data
            if "bytes" in message:
                audio_data = message["bytes"]

                # During SPEAKING state, check for barge-in (user interrupting)
                # Assumes client handles echo cancellation, so detected speech
                # is genuine user input, not TTS playback feedback.
                if session.state == VoiceState.SPEAKING:
                    logger.info(f"Received audio during SPEAKING: {len(audio_data)} bytes")
                    if session.detect_barge_in(audio_data):
                        logger.info("Auto-interrupt triggered by barge-in detection")
                        # Cancel the background task if running
                        if current_turn_task and not current_turn_task.done():
                            current_turn_task.cancel()
                        await service.handle_interrupt(websocket)
                    continue
                else:
                    logger.info(f"Received audio during {session.state.value}: {len(audio_data)} bytes (not SPEAKING)")

                # Update state to listening if idle
                if session.state == VoiceState.IDLE:
                    session.set_state(VoiceState.LISTENING)
                    await websocket.send_json(
                        StatusMessage(state=VoiceState.LISTENING).model_dump()
                    )
                    # Log VAD config when starting to listen
                    vad_cfg = session._vad.config
                    logger.info(
                        f"VAD config: threshold={vad_cfg.speech_threshold}, "
                        f"silence_duration={vad_cfg.silence_duration}s, "
                        f"min_speech={vad_cfg.min_speech_duration}s"
                    )

                # Append audio and check for end-of-speech via VAD
                # Note: append_audio returns True based on default silence threshold
                vad_speech_ended = session.append_audio(audio_data)

                # Debug logging for VAD diagnostics
                vad = session._vad
                fmt = session._audio_format.value if session._audio_format else "unknown"
                logger.debug(
                    f"Audio[{fmt}]: {len(audio_data)} bytes, "
                    f"energy={vad.get_average_energy():.4f}, "
                    f"state={vad.state.value}, "
                    f"speech={vad.get_speech_duration():.2f}s, "
                    f"silence={session.get_silence_duration():.2f}s, "
                    f"ended={vad_speech_ended}"
                )

                # Determine if we should process the turn
                should_process = False

                if session.config.turn_detection_enabled and session.is_in_silence_window():
                    silence_dur = session.get_silence_duration()
                    base_silence = session.config.base_silence_duration

                    # For native audio models, skip STT-based linguistic analysis
                    # Just use silence duration threshold
                    if session.config.use_native_audio:
                        # Simple silence-based turn detection for native audio
                        if silence_dur >= base_silence:
                            should_process = True
                    else:
                        # Smart turn detection: Use STT to analyze linguistic completeness
                        # Only do transcription when we have significant silence
                        if silence_dur >= base_silence and not session.get_partial_transcript():
                            # Do partial transcription for linguistic analysis
                            try:
                                audio_so_far = bytes(session._audio_buffer)
                                if len(audio_so_far) > 0:
                                    partial_text = await service.transcribe_audio(audio_so_far)
                                    session.set_partial_transcript(partial_text)
                                    logger.info(
                                        f"Partial transcription for turn detection: "
                                        f"'{partial_text[:100]}...' "
                                        f"(silence={silence_dur:.2f}s)"
                                    )
                            except Exception as e:
                                logger.warning(f"Partial transcription failed: {e}")

                        # Check if turn should end using linguistic analysis
                        should_process = session.check_end_of_turn_with_analysis()

                elif vad_speech_ended:
                    # Fallback: use default VAD behavior when turn detection disabled
                    should_process = True

                # Auto-trigger processing when turn should end
                if should_process and session.has_audio():
                    logger.info(
                        f"End of turn detected "
                        f"(turn_detection={session.config.turn_detection_enabled}, "
                        f"native_audio={session.config.use_native_audio}, "
                        f"silence={session.get_silence_duration():.2f}s), processing..."
                    )
                    audio_bytes = session.get_audio_buffer()
                    # Run process_turn as background task so we can receive audio during TTS
                    # This enables barge-in detection while TTS is streaming
                    current_turn_task = asyncio.create_task(
                        run_process_turn(audio_bytes, session.config.use_native_audio)
                    )

            # Handle JSON messages
            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    msg_type = data.get("type")

                    if msg_type == "interrupt":
                        # Barge-in: stop current TTS
                        if current_turn_task and not current_turn_task.done():
                            current_turn_task.cancel()
                        await service.handle_interrupt(websocket)

                    elif msg_type == "end":
                        # Process accumulated audio as background task
                        if session.has_audio():
                            audio_bytes = session.get_audio_buffer()
                            current_turn_task = asyncio.create_task(
                                run_process_turn(audio_bytes, session.config.use_native_audio)
                            )

                    elif msg_type == "config":
                        # Update session configuration
                        session.update_config(
                            stt_model=data.get("stt_model"),
                            tts_model=data.get("tts_model"),
                            tts_voice=data.get("tts_voice"),
                            llm_model=data.get("llm_model"),
                            language=data.get("language"),
                            speed=data.get("speed"),
                            sentence_boundary_only=data.get("sentence_boundary_only"),
                            barge_in_enabled=data.get("barge_in_enabled"),
                            barge_in_noise_filter=data.get("barge_in_noise_filter"),
                            barge_in_min_chunks=data.get("barge_in_min_chunks"),
                            turn_detection_enabled=data.get("turn_detection_enabled"),
                            base_silence_duration=data.get("base_silence_duration"),
                            thinking_silence_duration=data.get("thinking_silence_duration"),
                            max_silence_duration=data.get("max_silence_duration"),
                        )
                        logger.debug(f"Session config updated: {session.session_id}")

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON from client")
                    await websocket.send_json(
                        ErrorMessage(message="Invalid JSON message").model_dump()
                    )

    except WebSocketDisconnect:
        logger.info(f"Voice chat client disconnected: {session.session_id}")
    except Exception as e:
        logger.error(f"Voice chat error: {e}", exc_info=True)
        with contextlib.suppress(Exception):
            # Send sanitized error to client - don't expose internal details
            await websocket.send_json(
                ErrorMessage(
                    message="A server error occurred. Please reconnect and try again."
                ).model_dump()
            )
    finally:
        # Cancel any running process_turn task
        if current_turn_task and not current_turn_task.done():
            current_turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await current_turn_task

        # Clean up TTS WebSocket connection
        with contextlib.suppress(Exception):
            await service.cleanup()

        # Send closed message if connection still open
        with contextlib.suppress(Exception):
            await websocket.send_json(ClosedMessage().model_dump())

        logger.info(f"Voice chat session ended: {session.session_id}")
