"""
Voice chat service - orchestrates the STT → LLM → TTS pipeline.

Provides real-time voice conversation by:
1. Transcribing audio via Universal Runtime STT
2. Streaming LLM responses with phrase boundary detection
3. Synthesizing speech via Universal Runtime TTS WebSocket
"""

import json
import logging
import re
from collections.abc import AsyncGenerator

import httpx
import websockets
from fastapi import WebSocket

from config.datamodel import Model
from core.settings import settings
from services.universal_runtime_service import UniversalRuntimeService

from .phrase_detector import PhraseBoundaryDetector
from .session import VoiceSession
from .types import (
    ErrorMessage,
    LLMTextMessage,
    StatusMessage,
    TranscriptionMessage,
    TTSDoneMessage,
    TTSStartMessage,
    VoiceState,
)

logger = logging.getLogger(__name__)


class VoiceChatService:
    """Orchestrates the voice assistant pipeline.

    Pipeline flow:
    1. Receive audio from client WebSocket
    2. Transcribe via STT (Universal Runtime)
    3. Send to LLM with conversation history
    4. Detect phrase boundaries in LLM stream
    5. Synthesize each phrase via TTS (parallel to LLM)
    6. Stream audio back to client
    """

    def __init__(self, session: VoiceSession, llm_model_config: Model):
        """Initialize voice chat service.

        Args:
            session: Voice session with conversation state.
            llm_model_config: Resolved LLM model configuration from project.
                              Contains the actual model ID, base_url, etc.
        """
        self.session = session
        self._llm_model_config = llm_model_config

        # LLM endpoint - use model's base_url if specified, otherwise runtime default
        if llm_model_config.base_url:
            self._llm_url = llm_model_config.base_url.rstrip("/")
        else:
            self._llm_url = f"http://{settings.universal_host}:{settings.universal_port}/v1"

        # Runtime URLs for STT/TTS (always use Universal Runtime)
        self._runtime_url = (
            f"http://{settings.universal_host}:{settings.universal_port}"
        )
        self._runtime_ws_url = (
            f"ws://{settings.universal_host}:{settings.universal_port}"
        )

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Transcribe audio to text via Universal Runtime STT.

        Args:
            audio_bytes: Raw audio data (PCM 16kHz 16-bit mono or WebM/Opus).
                         Raw PCM is preferred for optimal performance.

        Returns:
            Transcribed text.
        """
        # Use .pcm extension to hint at raw PCM format (though detection is content-based)
        result = await UniversalRuntimeService.transcribe_audio(
            audio_bytes=audio_bytes,
            filename="audio.pcm",
            model=self.session.config.stt_model,
            language=self.session.config.language,
        )
        return result.get("text", "")

    def _inject_thinking_control(self, messages: list[dict]) -> list[dict]:
        """Inject thinking control into messages.

        When thinking is disabled (default for voice), appends /no_think
        to the last user message to instruct models like Qwen3 to skip
        chain-of-thought reasoning.

        Args:
            messages: List of chat messages.

        Returns:
            Modified messages list.
        """
        if self.session.config.enable_thinking:
            # Thinking enabled - don't modify
            return messages

        # Make a copy to avoid modifying session history
        messages = [dict(m) for m in messages]

        # Find the last user message and append /no_think
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                # Only add if not already present
                if "/think" not in content and "/no_think" not in content:
                    messages[i]["content"] = f"{content} /no_think"
                break

        return messages

    def _filter_thinking_tags(self, text: str) -> str:
        """Filter out <think>...</think> tags from text.

        Even with /no_think, some models may still output thinking tags.
        This ensures TTS never speaks the thinking content.

        Args:
            text: Raw LLM output.

        Returns:
            Text with thinking tags removed.
        """
        # Remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Remove orphaned closing tags
        text = re.sub(r"</think>", "", text, flags=re.IGNORECASE)
        # Remove orphaned opening tags (incomplete thinking)
        text = re.sub(r"<think>", "", text, flags=re.IGNORECASE)
        return text

    def _preprocess_for_speech(self, text: str) -> str:
        """Preprocess text to sound more natural when spoken.

        Applies transformations that make TTS output sound more human:
        - Expand common abbreviations
        - Convert symbols to spoken form
        - Normalize whitespace
        - Remove markdown formatting

        Args:
            text: Raw text from LLM.

        Returns:
            Text optimized for speech synthesis.
        """
        # Remove markdown bold/italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
        text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
        text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_

        # Remove markdown headers
        text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)

        # Remove markdown links, keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove code blocks and inline code
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove bullet points
        text = re.sub(r"^\s*[-*•]\s*", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s*", "", text, flags=re.MULTILINE)

        # Expand common abbreviations for natural speech
        abbreviations = {
            r"\bDr\.": "Doctor",
            r"\bMr\.": "Mister",
            r"\bMrs\.": "Misses",
            r"\bMs\.": "Miss",
            r"\bProf\.": "Professor",
            r"\betc\.": "etcetera",
            r"\be\.g\.": "for example",
            r"\bi\.e\.": "that is",
            r"\bvs\.": "versus",
            r"\bw/": "with",
            r"\bw/o": "without",
            r"\b&\b": "and",
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Convert common symbols
        text = text.replace("->", " to ")
        text = text.replace("=>", " implies ")
        text = text.replace("<=", " less than or equal to ")
        text = text.replace(">=", " greater than or equal to ")
        text = text.replace(" < ", " less than ")
        text = text.replace(" > ", " greater than ")
        text = text.replace("%", " percent")
        text = text.replace("$", " dollars ")
        text = text.replace("€", " euros ")
        text = text.replace("£", " pounds ")

        # Add slight pause after colons for better pacing
        text = re.sub(r":\s*", ": ... ", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    async def stream_llm_response(
        self, user_text: str
    ) -> AsyncGenerator[str, None]:
        """Stream LLM response tokens.

        Args:
            user_text: User's transcribed text.

        Yields:
            LLM response tokens (with thinking tags filtered out).
        """
        # Add user message to history
        self.session.add_user_message(user_text)

        # Prepare messages with thinking control
        messages = self._inject_thinking_control(self.session.messages)
        if not self.session.config.enable_thinking:
            logger.debug("Thinking disabled for voice - injected /no_think")

        # Prepare request using resolved model config
        # Use the actual model ID (e.g., "unsloth/Qwen3-4B-GGUF:Q4_K_M"), not the project name
        url = f"{self._llm_url}/chat/completions"
        payload = {
            "model": self._llm_model_config.model,  # Actual model ID
            "messages": messages,  # Use modified messages with thinking control
            "stream": True,
        }

        # Add any model-specific parameters
        if self._llm_model_config.model_api_parameters:
            payload.update(self._llm_model_config.model_api_parameters)

        accumulated_response = ""

        try:
            async with (
                httpx.AsyncClient(timeout=300.0) as client,
                client.stream("POST", url, json=payload) as response,
            ):
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            accumulated_response += content
                            yield content
                    except json.JSONDecodeError:
                        continue

            # Add complete response to history (with thinking tags filtered)
            if accumulated_response:
                clean_response = self._filter_thinking_tags(accumulated_response).strip()
                if clean_response:
                    self.session.add_assistant_message(clean_response)

        except httpx.HTTPStatusError as e:
            logger.error(f"LLM request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM streaming error: {e}")
            raise

    async def synthesize_phrase_stream(
        self, phrase: str, phrase_index: int
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize a phrase via TTS WebSocket and yield audio chunks.

        Args:
            phrase: Text to synthesize.
            phrase_index: Index for coordination.

        Yields:
            PCM audio chunks.
        """
        ws_url = (
            f"{self._runtime_ws_url}/v1/audio/speech/stream"
            f"?model={self.session.config.tts_model}"
            f"&voice={self.session.config.tts_voice}"
            f"&response_format=pcm"
        )

        try:
            async with websockets.connect(ws_url) as ws:
                # Send synthesis request
                await ws.send(json.dumps({
                    "text": phrase,
                    "speed": self.session.config.speed,
                    "final": True,
                }))

                # Receive audio chunks
                while True:
                    message = await ws.recv()

                    if isinstance(message, bytes):
                        # Audio chunk
                        yield message
                    else:
                        # JSON message
                        data = json.loads(message)
                        msg_type = data.get("type")

                        if msg_type == "done":
                            break
                        elif msg_type == "error":
                            logger.error(f"TTS error: {data.get('message')}")
                            break
                        elif msg_type == "closed":
                            break

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"TTS WebSocket closed for phrase {phrase_index}")
        except Exception as e:
            logger.error(f"TTS synthesis error for phrase {phrase_index}: {e}")

    async def process_turn(
        self, websocket: WebSocket, audio_bytes: bytes
    ) -> None:
        """Process a single conversational turn.

        1. Transcribe audio
        2. Send to LLM
        3. Stream TTS for each phrase
        4. Handle interrupts

        Args:
            websocket: Client WebSocket connection.
            audio_bytes: User's audio input.
        """
        # Clear any previous interrupt
        self.session.clear_interrupt()
        self.session.reset_phrase_counter()

        # Update state to processing
        self.session.set_state(VoiceState.PROCESSING)
        await websocket.send_json(StatusMessage(state=VoiceState.PROCESSING).model_dump())

        try:
            # Step 1: Transcribe audio
            transcription = await self.transcribe_audio(audio_bytes)

            if not transcription.strip():
                logger.debug("Empty transcription, skipping")
                self.session.set_state(VoiceState.IDLE)
                await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())
                return

            # Send transcription to client
            await websocket.send_json(
                TranscriptionMessage(text=transcription, is_final=True).model_dump()
            )

            # Step 2 & 3: Stream LLM and TTS in parallel
            self.session.set_state(VoiceState.SPEAKING)
            await websocket.send_json(StatusMessage(state=VoiceState.SPEAKING).model_dump())

            # Use phrase detector to accumulate LLM tokens
            phrase_detector = PhraseBoundaryDetector()
            full_response = ""

            async for token in self.stream_llm_response(transcription):
                # Check for interrupt
                if self.session.is_interrupted():
                    logger.info("Turn interrupted by user")
                    break

                full_response += token

                # Detect phrase boundaries
                phrase = phrase_detector.add_token(token)
                if phrase:
                    # Send LLM text to client for display
                    await websocket.send_json(
                        LLMTextMessage(text=phrase, is_final=False).model_dump()
                    )

                    # Synthesize and stream audio for this phrase
                    await self._synthesize_and_stream_phrase(websocket, phrase)

                    # Check interrupt again after TTS
                    if self.session.is_interrupted():
                        break

            # Flush remaining text
            if not self.session.is_interrupted():
                remaining = phrase_detector.flush()
                if remaining:
                    await websocket.send_json(
                        LLMTextMessage(text=remaining, is_final=True).model_dump()
                    )
                    await self._synthesize_and_stream_phrase(websocket, remaining)

            # Done with this turn
            self.session.set_state(VoiceState.IDLE)
            await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())

        except Exception as e:
            logger.error(f"Error processing turn: {e}", exc_info=True)
            await websocket.send_json(
                ErrorMessage(message=f"Processing error: {str(e)}").model_dump()
            )
            self.session.set_state(VoiceState.IDLE)

    async def _synthesize_and_stream_phrase(
        self, websocket: WebSocket, phrase: str
    ) -> None:
        """Synthesize a phrase and stream audio to client.

        Args:
            websocket: Client WebSocket connection.
            phrase: Text to synthesize.
        """
        # Filter thinking tags and preprocess for natural speech
        phrase = self._filter_thinking_tags(phrase)
        phrase = self._preprocess_for_speech(phrase)
        if not phrase:
            # Skip empty phrases (e.g., if it was all thinking/markdown content)
            return

        phrase_index = self.session.next_phrase_index()

        # Notify phrase TTS starting
        await websocket.send_json(
            TTSStartMessage(phrase_index=phrase_index).model_dump()
        )

        total_samples = 0

        async for audio_chunk in self.synthesize_phrase_stream(phrase, phrase_index):
            # Check for interrupt
            if self.session.is_interrupted():
                break

            # Send audio chunk as binary
            await websocket.send_bytes(audio_chunk)

            # Track duration (PCM 16-bit = 2 bytes per sample)
            total_samples += len(audio_chunk) // 2

        # Calculate duration (24kHz sample rate)
        duration = total_samples / 24000.0

        # Notify phrase TTS complete
        await websocket.send_json(
            TTSDoneMessage(phrase_index=phrase_index, duration=duration).model_dump()
        )

    async def handle_interrupt(self, websocket: WebSocket) -> None:
        """Handle barge-in interrupt from client.

        Args:
            websocket: Client WebSocket connection.
        """
        self.session.request_interrupt()
        self.session.set_state(VoiceState.INTERRUPTED)
        await websocket.send_json(
            StatusMessage(state=VoiceState.INTERRUPTED).model_dump()
        )

        # Transition to listening for new input
        self.session.set_state(VoiceState.LISTENING)
        await websocket.send_json(
            StatusMessage(state=VoiceState.LISTENING).model_dump()
        )
