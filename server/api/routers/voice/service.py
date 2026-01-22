"""
Voice chat service - orchestrates the STT → LLM → TTS pipeline.

Provides real-time voice conversation by:
1. Transcribing audio via Universal Runtime STT
2. Streaming LLM responses with phrase boundary detection
3. Synthesizing speech via Universal Runtime TTS WebSocket
"""

import asyncio
import json
import logging
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx
import websockets
from fastapi import WebSocket


@dataclass
class LLMToolCall:
    """Represents a tool call from the LLM stream."""

    id: str
    name: str
    arguments: str  # JSON string of arguments


@dataclass
class LLMContent:
    """Represents regular text content from the LLM stream."""

    text: str


# Type alias for stream output
LLMStreamOutput = LLMContent | LLMToolCall

from config.datamodel import Model
from core.settings import settings
from services.universal_runtime_service import UniversalRuntimeService

from .phrase_detector import PhraseBoundaryDetector
from .session import VoiceSession
from .types import (
    ErrorMessage,
    LLMTextMessage,
    StatusMessage,
    ToolCallMessage,
    TranscriptionMessage,
    TTSDoneMessage,
    TTSStartMessage,
    VoiceState,
)

logger = logging.getLogger(__name__)


class StreamingThinkingFilter:
    """Filters <think>...</think> blocks from a token stream.

    Tracks state across token boundaries to handle cases where tags
    are split across multiple tokens.
    """

    def __init__(self):
        self._in_thinking = False
        self._buffer = ""
        # Patterns to detect tag boundaries
        self._open_tag = re.compile(r"<think>", re.IGNORECASE)
        self._close_tag = re.compile(r"</think>", re.IGNORECASE)

    def filter_token(self, token: str) -> str:
        """Filter a token, removing thinking content.

        Args:
            token: Incoming token from LLM stream.

        Returns:
            Filtered token (may be empty if inside thinking block).
        """
        # Add token to buffer for pattern matching
        self._buffer += token

        # Process buffer to extract non-thinking content
        result = ""
        while True:
            if self._in_thinking:
                # Look for closing tag
                match = self._close_tag.search(self._buffer)
                if match:
                    # Found closing tag - exit thinking mode
                    self._in_thinking = False
                    self._buffer = self._buffer[match.end():]
                else:
                    # Still in thinking mode, discard buffer but keep last 8 chars
                    # (to catch split </think> tag)
                    if len(self._buffer) > 8:
                        self._buffer = self._buffer[-8:]
                    break
            else:
                # Look for opening tag
                match = self._open_tag.search(self._buffer)
                if match:
                    # Found opening tag - emit content before it, enter thinking mode
                    result += self._buffer[:match.start()]
                    self._in_thinking = True
                    self._buffer = self._buffer[match.end():]
                else:
                    # No tag found - emit most of buffer, keep last 7 chars
                    # (to catch split <think> tag)
                    if len(self._buffer) > 7:
                        emit_len = len(self._buffer) - 7
                        result += self._buffer[:emit_len]
                        self._buffer = self._buffer[emit_len:]
                    break

        return result

    def flush(self) -> str:
        """Flush remaining buffer content.

        Call at end of stream to get any remaining non-thinking content.
        """
        if self._in_thinking:
            return ""
        result = self._buffer
        self._buffer = ""
        return result


class StreamingToolCallFilter:
    """Filters JSON-like tool call content from a token stream.

    Detects JSON objects that look like tool calls (containing keys like
    'name', 'function', 'arguments', 'tool_call', etc.) and filters them
    out so they don't get sent to TTS.
    """

    # Keys that indicate a JSON object is a tool call
    TOOL_CALL_KEYS = {
        "name", "function", "arguments", "tool_call", "tool_calls",
        "function_call", "type", "id", "parameters",
    }

    def __init__(self):
        self._buffer = ""
        self._in_json = False
        self._brace_depth = 0
        self._bracket_depth = 0
        self._in_string = False
        self._escape_next = False
        self._detected_tool_calls: list[str] = []

    def _is_tool_call_json(self, json_str: str) -> bool:
        """Check if a JSON string looks like a tool call."""
        try:
            data = json.loads(json_str)
            if isinstance(data, dict):
                keys = set(data.keys())
                logger.debug(f"🔍 JSON object keys: {keys}")
                # Check if any tool call keys are present
                if keys & self.TOOL_CALL_KEYS:
                    logger.info(f"✅ Detected tool call JSON with keys: {keys & self.TOOL_CALL_KEYS}")
                    return True
                else:
                    logger.debug(f"❌ JSON object not a tool call, keys: {keys}")
            elif isinstance(data, list) and data:
                # Check first element if it's an array
                if isinstance(data[0], dict):
                    keys = set(data[0].keys())
                    logger.debug(f"🔍 JSON array[0] keys: {keys}")
                    if keys & self.TOOL_CALL_KEYS:
                        logger.info(f"✅ Detected tool call JSON array with keys: {keys & self.TOOL_CALL_KEYS}")
                        return True
        except (json.JSONDecodeError, TypeError, IndexError) as e:
            logger.debug(f"❌ JSON parse failed for: {json_str[:100]}... error: {e}")
        return False

    def filter_token(self, token: str) -> str:
        """Filter a token, removing tool call JSON content.

        Args:
            token: Incoming token from LLM stream.

        Returns:
            Filtered token (may be empty if inside tool call JSON).
        """
        result = ""

        for char in token:
            # Handle escape sequences in strings
            if self._escape_next:
                self._escape_next = False
                if self._in_json:
                    self._buffer += char
                else:
                    result += char
                continue

            if char == "\\" and self._in_string:
                self._escape_next = True
                if self._in_json:
                    self._buffer += char
                else:
                    result += char
                continue

            # Track string boundaries
            if char == '"' and not self._escape_next:
                self._in_string = not self._in_string
                if self._in_json:
                    self._buffer += char
                else:
                    result += char
                continue

            # Don't process braces inside strings
            if self._in_string:
                if self._in_json:
                    self._buffer += char
                else:
                    result += char
                continue

            # Track JSON object/array boundaries
            if char == "{":
                if not self._in_json and self._brace_depth == 0 and self._bracket_depth == 0:
                    # Starting a new potential JSON object
                    self._in_json = True
                    self._buffer = char
                    self._brace_depth = 1
                elif self._in_json:
                    self._buffer += char
                    self._brace_depth += 1
                else:
                    result += char
            elif char == "}":
                if self._in_json:
                    self._buffer += char
                    self._brace_depth -= 1
                    if self._brace_depth == 0 and self._bracket_depth == 0:
                        # Completed a JSON object - check if it's a tool call
                        logger.info(f"🔶 Complete JSON object detected: {self._buffer[:150]}...")
                        if self._is_tool_call_json(self._buffer):
                            logger.info(f"🚫 Filtered tool call JSON: {self._buffer[:100]}...")
                            self._detected_tool_calls.append(self._buffer)
                            # Don't emit the tool call content
                        else:
                            # Not a tool call, emit the buffered content
                            logger.debug(f"✓ JSON not a tool call, emitting: {self._buffer[:100]}...")
                            result += self._buffer
                        self._buffer = ""
                        self._in_json = False
                else:
                    result += char
            elif char == "[":
                if not self._in_json and self._brace_depth == 0 and self._bracket_depth == 0:
                    # Starting a new potential JSON array
                    self._in_json = True
                    self._buffer = char
                    self._bracket_depth = 1
                elif self._in_json:
                    self._buffer += char
                    self._bracket_depth += 1
                else:
                    result += char
            elif char == "]":
                if self._in_json:
                    self._buffer += char
                    self._bracket_depth -= 1
                    if self._brace_depth == 0 and self._bracket_depth == 0:
                        # Completed a JSON array - check if it's a tool call
                        if self._is_tool_call_json(self._buffer):
                            logger.debug(f"Filtered tool call JSON array: {self._buffer[:100]}...")
                            self._detected_tool_calls.append(self._buffer)
                        else:
                            result += self._buffer
                        self._buffer = ""
                        self._in_json = False
                else:
                    result += char
            else:
                if self._in_json:
                    self._buffer += char
                else:
                    result += char

        return result

    def flush(self) -> str:
        """Flush remaining buffer content.

        If we have incomplete JSON, emit it (it's probably not a tool call).
        """
        result = self._buffer
        self._buffer = ""
        self._in_json = False
        self._brace_depth = 0
        self._bracket_depth = 0
        return result

    def get_detected_tool_calls(self) -> list[str]:
        """Return list of detected tool call JSON strings."""
        return self._detected_tool_calls


class VoiceChatService:
    """Orchestrates the voice assistant pipeline.

    Pipeline flow:
    1. Receive audio from client WebSocket
    2. Transcribe via STT (Universal Runtime)
    3. Send to LLM with conversation history
    4. Detect phrase boundaries in LLM stream
    5. Synthesize each phrase via TTS (parallel to LLM)
    6. Stream audio back to client

    Performance optimizations:
    - Persistent HTTP client for LLM (avoids connection overhead)
    - Reusable TTS WebSocket connection
    - Early LLM start on partial transcription
    """

    # Shared HTTP client for LLM requests (connection pooling)
    _http_client: httpx.AsyncClient | None = None

    @classmethod
    def get_http_client(cls) -> httpx.AsyncClient:
        """Get or create shared HTTP client with connection pooling."""
        if cls._http_client is None or cls._http_client.is_closed:
            cls._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=5.0,  # Fast connect timeout
                    read=300.0,   # Long read timeout for streaming
                    write=10.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(
                    max_keepalive_connections=5,
                    max_connections=10,
                    keepalive_expiry=30.0,
                ),
            )
        return cls._http_client

    def __init__(self, session: VoiceSession, llm_model_config: Model | None):
        """Initialize voice chat service.

        Args:
            session: Voice session with conversation state.
            llm_model_config: Resolved LLM model configuration from project.
                              Contains the actual model ID, base_url, etc.
                              None when running in stt_only mode.
        """
        self.session = session
        self._llm_model_config = llm_model_config

        # LLM endpoint - use model's base_url if specified, otherwise runtime default
        # Note: In stt_only mode, llm_model_config is None and LLM won't be called
        if llm_model_config and llm_model_config.base_url:
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

        # Reusable TTS WebSocket connection
        self._tts_ws: websockets.WebSocketClientProtocol | None = None

    async def warm_up(self) -> None:
        """Pre-warm connections to minimize first-request latency.

        Call this when a session starts to establish connections before
        the user speaks.
        """
        try:
            # Pre-establish TTS WebSocket (skip in stt_only mode)
            if not self.session.config.stt_only:
                ws = await self._get_tts_websocket()
                logger.debug(f"TTS WebSocket pre-warmed: {ws.remote_address}")

                # Pre-warm HTTP connection pool with a lightweight request
                client = self.get_http_client()
                # Just establish the TCP connection, don't make a full request
                logger.debug("HTTP client pool pre-warmed")
            else:
                logger.debug("STT-only mode: skipping TTS/LLM connection warm-up")
        except Exception as e:
            logger.warning(f"Connection pre-warm failed (non-fatal): {e}")

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

    async def transcribe_audio_stream(
        self, audio_bytes: bytes
    ) -> AsyncGenerator[str, None]:
        """Stream transcription segments as they're processed.

        Enables parallel processing - start LLM on first segment while
        STT continues processing remaining audio.

        Args:
            audio_bytes: Raw PCM audio (16kHz 16-bit mono).

        Yields:
            Transcribed text segments.
        """
        async for segment in UniversalRuntimeService.transcribe_audio_stream(
            audio_bytes=audio_bytes,
            model=self.session.config.stt_model,
            language=self.session.config.language,
        ):
            text = segment.get("text", "").strip()
            if text:
                yield text

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
        - Expand contractions for clearer pronunciation
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

        # NOTE: We do NOT expand contractions - they sound more natural in speech.
        # TTS models are trained on natural spoken language which includes contractions.

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
            r"\bAI\b": "A.I.",  # Spell out for clearer pronunciation
        }
        for pattern, replacement in abbreviations.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # NOTE: We do NOT expand ordinals (1st, 2nd) or most symbols -
        # modern TTS models handle these naturally.

        # Remove URLs (TTS can't pronounce them naturally)
        text = re.sub(r"https?://\S+", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    async def stream_llm_response(
        self, user_text: str
    ) -> AsyncGenerator[LLMStreamOutput, None]:
        """Stream LLM response tokens and tool calls.

        Args:
            user_text: User's transcribed text.

        Yields:
            LLMContent for regular text tokens, LLMToolCall for tool calls.
            Tool calls are yielded complete (not streamed token-by-token).
        """
        # === TIMING INSTRUMENTATION ===
        t_start = time.perf_counter()
        t_connected = None
        t_first_token = None
        first_token_logged = False

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
            # Speed optimizations for voice
            "temperature": 0.7,  # Slightly lower for faster sampling
            "max_tokens": 500,   # Limit response length for voice
        }

        # Add any model-specific parameters (may override above)
        if self._llm_model_config.model_api_parameters:
            payload.update(self._llm_model_config.model_api_parameters)

        accumulated_response = ""
        token_count = 0

        # Tool call tracking - tool calls can be spread across multiple chunks
        # Structure: {index: {"id": str, "name": str, "arguments": str}}
        pending_tool_calls: dict[int, dict] = {}

        try:
            # Use shared HTTP client with connection pooling for lower latency
            client = self.get_http_client()
            async with client.stream("POST", url, json=payload) as response:
                t_connected = time.perf_counter()
                logger.info(f"⏱️ LLM: HTTP stream connected in {(t_connected - t_start)*1000:.1f}ms")
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        break

                    try:
                        chunk = json.loads(data)
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})

                        # Handle regular content tokens
                        content = delta.get("content", "")
                        if content:
                            if not first_token_logged:
                                t_first_token = time.perf_counter()
                                first_token_logged = True
                                logger.info(f"⏱️ LLM: First token in {(t_first_token - t_start)*1000:.1f}ms total, {(t_first_token - t_connected)*1000:.1f}ms after connect")
                            token_count += 1
                            accumulated_response += content
                            yield LLMContent(text=content)

                        # Handle tool calls (accumulate across chunks)
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            idx = tc.get("index", 0)

                            # Initialize tracking for new tool call
                            if idx not in pending_tool_calls:
                                pending_tool_calls[idx] = {
                                    "id": "",
                                    "name": "",
                                    "arguments": "",
                                }

                            # Accumulate tool call data
                            if tc.get("id"):
                                pending_tool_calls[idx]["id"] = tc["id"]
                            if tc.get("function", {}).get("name"):
                                pending_tool_calls[idx]["name"] = tc["function"]["name"]
                            if tc.get("function", {}).get("arguments"):
                                pending_tool_calls[idx]["arguments"] += tc["function"]["arguments"]

                        # Check if stream indicates finish_reason for tool_calls
                        finish_reason = choice.get("finish_reason")
                        if finish_reason == "tool_calls" and pending_tool_calls:
                            # Yield all completed tool calls
                            for idx in sorted(pending_tool_calls.keys()):
                                tc_data = pending_tool_calls[idx]
                                if tc_data["id"] and tc_data["name"]:
                                    logger.info(
                                        f"Tool call detected: {tc_data['name']}({tc_data['arguments'][:100]}...)"
                                    )
                                    yield LLMToolCall(
                                        id=tc_data["id"],
                                        name=tc_data["name"],
                                        arguments=tc_data["arguments"],
                                    )
                            pending_tool_calls.clear()

                    except json.JSONDecodeError:
                        continue

            # Yield any remaining tool calls that weren't finalized with finish_reason
            for idx in sorted(pending_tool_calls.keys()):
                tc_data = pending_tool_calls[idx]
                if tc_data["id"] and tc_data["name"]:
                    logger.info(
                        f"Tool call (end of stream): {tc_data['name']}({tc_data['arguments'][:100]}...)"
                    )
                    yield LLMToolCall(
                        id=tc_data["id"],
                        name=tc_data["name"],
                        arguments=tc_data["arguments"],
                    )

            # Log final stats
            t_done = time.perf_counter()
            logger.info(f"⏱️ LLM: Complete in {(t_done - t_start)*1000:.1f}ms, {token_count} tokens, {len(accumulated_response)} chars")

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

    async def _get_tts_websocket(self) -> websockets.WebSocketClientProtocol:
        """Get or create TTS WebSocket connection.

        Reuses existing connection to avoid handshake overhead (~50-100ms per connection).
        """
        from websockets import State
        if self._tts_ws is None:
            logger.info("TTS WebSocket: creating new connection (was None)")
        elif self._tts_ws.state != State.OPEN:
            logger.info(f"TTS WebSocket: creating new connection (state={self._tts_ws.state})")
        else:
            logger.info("TTS WebSocket: reusing existing connection")
            return self._tts_ws

        ws_url = (
            f"{self._runtime_ws_url}/v1/audio/speech/stream"
            f"?model={self.session.config.tts_model}"
            f"&voice={self.session.config.tts_voice}"
            f"&response_format=pcm"
        )
        self._tts_ws = await websockets.connect(ws_url)
        return self._tts_ws

    async def _close_tts_websocket(self) -> None:
        """Close TTS WebSocket connection."""
        from websockets import State
        if self._tts_ws is not None and self._tts_ws.state == State.OPEN:
            await self._tts_ws.close()
            self._tts_ws = None

    async def invalidate_tts_connection(self) -> None:
        """Invalidate TTS WebSocket connection.

        Call this when TTS config (voice or model) changes to force
        recreation with new parameters on next synthesis.
        """
        await self._close_tts_websocket()
        logger.info("TTS WebSocket invalidated due to config change")

    async def synthesize_phrase_stream(
        self, phrase: str, phrase_index: int
    ) -> AsyncGenerator[bytes, None]:
        """Synthesize a phrase via TTS WebSocket and yield audio chunks.

        Uses a reusable WebSocket connection to minimize latency.

        Args:
            phrase: Text to synthesize.
            phrase_index: Index for coordination.

        Yields:
            PCM audio chunks.
        """
        try:
            ws = await self._get_tts_websocket()

            # Send synthesis request
            # Use final=False to keep WebSocket connection open for reuse
            # across multiple phrases (reduces ~50-100ms overhead per phrase)
            await ws.send(json.dumps({
                "text": phrase,
                "speed": self.session.config.speed,
                "final": False,
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
                        logger.info(f"TTS phrase {phrase_index}: received 'done', keeping connection open")
                        break
                    elif msg_type == "error":
                        logger.error(f"TTS error: {data.get('message')}")
                        # Reset connection on error
                        self._tts_ws = None
                        break
                    elif msg_type == "closed":
                        logger.info(f"TTS phrase {phrase_index}: server sent 'closed'")
                        self._tts_ws = None
                        break

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"TTS WebSocket closed for phrase {phrase_index}")
            self._tts_ws = None
        except Exception as e:
            logger.error(f"TTS synthesis error for phrase {phrase_index}: {e}")
            self._tts_ws = None

    async def process_turn(
        self, websocket: WebSocket, audio_bytes: bytes
    ) -> None:
        """Process a single conversational turn with parallel STT+LLM.

        Optimized pipeline:
        1. Start streaming STT
        2. As soon as first segment arrives, start LLM (parallel with remaining STT)
        3. Stream TTS for each LLM phrase
        4. Handle interrupts

        This reduces time-to-first-audio by starting LLM before STT completes.

        Args:
            websocket: Client WebSocket connection.
            audio_bytes: User's audio input.
        """
        # === TIMING INSTRUMENTATION ===
        t_start = time.perf_counter()
        t_first_stt_segment = None
        t_stt_complete = None
        t_llm_first_token = None
        t_first_phrase = None
        t_first_tts_audio = None

        # Clear any previous interrupt
        self.session.clear_interrupt()
        self.session.reset_phrase_counter()

        # Update state to processing
        self.session.set_state(VoiceState.PROCESSING)
        await websocket.send_json(StatusMessage(state=VoiceState.PROCESSING).model_dump())

        try:
            # PARALLEL STT+LLM: Collect first segment(s) quickly, then start LLM
            # while STT continues in background
            transcription_parts: list[str] = []
            llm_started = False
            # Start LLM very early - even 5 chars is enough (e.g., "Hello" or "Hi!")
            min_chars_for_llm = 5

            # Try streaming transcription with timeout, fall back to HTTP if needed
            # Streaming allows parallel LLM start but HTTP is more reliable
            try:
                async with asyncio.timeout(2.0):  # 2 second timeout for streaming
                    async for segment_text in self.transcribe_audio_stream(audio_bytes):
                        if t_first_stt_segment is None:
                            t_first_stt_segment = time.perf_counter()
                            logger.info(f"⏱️ TIMING: First STT segment: {(t_first_stt_segment - t_start)*1000:.1f}ms")

                        transcription_parts.append(segment_text)
                        current_text = " ".join(transcription_parts)

                        # Send incremental transcription to client
                        await websocket.send_json(
                            TranscriptionMessage(
                                text=current_text,
                                is_final=False
                            ).model_dump()
                        )

                        # Start LLM as soon as we have enough text
                        if not llm_started and len(current_text) >= min_chars_for_llm:
                            llm_started = True
                            # Break to start LLM - remaining STT will be appended to display
                            break
            except TimeoutError:
                logger.debug("STT streaming timeout, using collected segments")

            # If no segments received, fall back to non-streaming HTTP endpoint
            # This is faster for short utterances where streaming overhead dominates
            if not transcription_parts:
                logger.debug("No STT segments, falling back to HTTP endpoint")
                t_http_start = time.perf_counter()
                transcription = await self.transcribe_audio(audio_bytes)
                t_stt_complete = time.perf_counter()
                logger.info(f"⏱️ TIMING: STT HTTP fallback: {(t_stt_complete - t_http_start)*1000:.1f}ms")

                if not transcription.strip():
                    logger.debug("Empty transcription, skipping")
                    self.session.set_state(VoiceState.IDLE)
                    await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())
                    return
                transcription_parts = [transcription]
            else:
                t_stt_complete = time.perf_counter()
                logger.info(f"⏱️ TIMING: STT streaming complete: {(t_stt_complete - t_start)*1000:.1f}ms")

            # Use what we have so far for LLM (first segment(s))
            transcription_for_llm = " ".join(transcription_parts).strip()

            if not transcription_for_llm:
                logger.debug("Empty transcription, skipping")
                self.session.set_state(VoiceState.IDLE)
                await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())
                return

            # Send final transcription (what LLM will use)
            await websocket.send_json(
                TranscriptionMessage(text=transcription_for_llm, is_final=True).model_dump()
            )

            # STT-only mode: skip LLM and TTS, return to idle
            if self.session.config.stt_only:
                logger.debug("STT-only mode, skipping LLM/TTS")
                self.session.set_state(VoiceState.IDLE)
                await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())
                return

            # Step 2 & 3: Stream LLM and TTS in parallel
            self.session.set_state(VoiceState.SPEAKING)
            await websocket.send_json(StatusMessage(state=VoiceState.SPEAKING).model_dump())

            # Use phrase detector to accumulate LLM tokens
            # Pass sentence_boundary_only config for natural speech (avoids mid-sentence breaks)
            phrase_detector = PhraseBoundaryDetector(
                sentence_boundary_only=self.session.config.sentence_boundary_only,
            )
            # Filter out <think>...</think> blocks at the token level
            thinking_filter = StreamingThinkingFilter()
            # Filter out JSON-like tool call content from text stream
            tool_call_filter = StreamingToolCallFilter()
            full_response = ""
            first_token_logged = False
            first_phrase_logged = False

            t_llm_start = time.perf_counter()
            logger.info(f"⏱️ TIMING: Starting LLM request: {(t_llm_start - t_start)*1000:.1f}ms from turn start")

            # Track if we've already spoken a tool call placeholder this turn
            tool_call_placeholder_spoken = False

            async for output in self.stream_llm_response(transcription_for_llm):
                if not first_token_logged:
                    t_llm_first_token = time.perf_counter()
                    first_token_logged = True
                    logger.info(f"⏱️ TIMING: LLM first token: {(t_llm_first_token - t_start)*1000:.1f}ms total, {(t_llm_first_token - t_llm_start)*1000:.1f}ms from request")

                # Check for interrupt
                if self.session.is_interrupted():
                    logger.info("Turn interrupted by user")
                    break

                # Handle tool calls - send via JSON only, never TTS
                if isinstance(output, LLMToolCall):
                    logger.info(f"Tool call received: {output.name} - sending via JSON only")

                    # Send tool call as JSON for the client
                    await websocket.send_json(
                        ToolCallMessage(
                            tool_call_id=output.id,
                            function_name=output.name,
                            arguments=output.arguments,
                        ).model_dump()
                    )

                    # Speak a brief placeholder if we haven't already this turn
                    if not tool_call_placeholder_spoken:
                        placeholder = "One moment."
                        await websocket.send_json(
                            LLMTextMessage(text=placeholder, is_final=False).model_dump()
                        )
                        await self._synthesize_and_stream_phrase(
                            websocket, placeholder,
                            track_first_audio=(t_first_tts_audio is None),
                            turn_start_time=t_start
                        )
                        tool_call_placeholder_spoken = True

                    continue

                # Handle regular content tokens
                if isinstance(output, LLMContent):
                    token = output.text

                    # DEBUG: Log raw token content
                    if token.strip():
                        logger.debug(f"🔤 RAW TOKEN: {repr(token)}")

                    # Filter thinking content at token level (before phrase detection)
                    filtered_token = thinking_filter.filter_token(token)
                    full_response += token  # Keep full response for history

                    # Skip empty tokens (thinking content filtered out)
                    if not filtered_token:
                        continue

                    # DEBUG: Log after thinking filter
                    if filtered_token != token:
                        logger.debug(f"🧠 AFTER THINKING FILTER: {repr(filtered_token)}")

                    # Filter out JSON-like tool call content from text
                    # (some models output tool calls as text, not structured)
                    before_tool_filter = filtered_token
                    filtered_token = tool_call_filter.filter_token(filtered_token)

                    # DEBUG: Log if tool call filter changed anything
                    if filtered_token != before_tool_filter:
                        logger.debug(f"🔧 TOOL FILTER removed: {repr(before_tool_filter)} -> {repr(filtered_token)}")

                    # Check if tool call filter detected any tool calls - speak placeholder
                    if tool_call_filter.get_detected_tool_calls() and not tool_call_placeholder_spoken:
                        placeholder = "One moment."
                        await websocket.send_json(
                            LLMTextMessage(text=placeholder, is_final=False).model_dump()
                        )
                        await self._synthesize_and_stream_phrase(
                            websocket, placeholder,
                            track_first_audio=(t_first_tts_audio is None),
                            turn_start_time=t_start
                        )
                        tool_call_placeholder_spoken = True

                    # Skip empty tokens (tool call content filtered out)
                    if not filtered_token:
                        continue

                    # Detect phrase boundaries on filtered content
                    phrase = phrase_detector.add_token(filtered_token)
                    if phrase:
                        if not first_phrase_logged:
                            t_first_phrase = time.perf_counter()
                            first_phrase_logged = True
                            logger.info(f"⏱️ TIMING: First phrase detected: {(t_first_phrase - t_start)*1000:.1f}ms total, phrase='{phrase[:50]}...'")

                        # Send LLM text to client for display
                        await websocket.send_json(
                            LLMTextMessage(text=phrase, is_final=False).model_dump()
                        )

                        # Synthesize and stream audio for this phrase
                        # Pass timing context for first TTS audio tracking
                        tts_timing = await self._synthesize_and_stream_phrase(
                            websocket, phrase,
                            track_first_audio=(t_first_tts_audio is None),
                            turn_start_time=t_start
                        )
                        if t_first_tts_audio is None and tts_timing:
                            t_first_tts_audio = tts_timing

                        # Check interrupt again after TTS
                        if self.session.is_interrupted():
                            break

            # Flush remaining text
            if not self.session.is_interrupted():
                # First flush any remaining content from thinking filter
                remaining_filtered = thinking_filter.flush()
                if remaining_filtered:
                    # Also filter through tool call filter
                    remaining_filtered = tool_call_filter.filter_token(remaining_filtered)

                # Flush tool call filter as well
                tool_call_remaining = tool_call_filter.flush()
                if tool_call_remaining:
                    remaining_filtered = (remaining_filtered or "") + tool_call_remaining

                if remaining_filtered:
                    # Add filtered content to phrase detector
                    phrase = phrase_detector.add_token(remaining_filtered)
                    if phrase:
                        await websocket.send_json(
                            LLMTextMessage(text=phrase, is_final=False).model_dump()
                        )
                        await self._synthesize_and_stream_phrase(websocket, phrase)

                # Then flush the phrase detector
                remaining = phrase_detector.flush()
                if remaining:
                    await websocket.send_json(
                        LLMTextMessage(text=remaining, is_final=True).model_dump()
                    )
                    await self._synthesize_and_stream_phrase(websocket, remaining)
                else:
                    # Always send is_final=True to signal end of LLM response
                    # This ensures the client can add the user message to history
                    await websocket.send_json(
                        LLMTextMessage(text="", is_final=True).model_dump()
                    )

            # === TIMING SUMMARY ===
            t_end = time.perf_counter()
            logger.info(f"⏱️ TIMING SUMMARY for turn:")
            logger.info(f"  Total turn duration: {(t_end - t_start)*1000:.1f}ms")
            if t_first_stt_segment:
                logger.info(f"  First STT segment: {(t_first_stt_segment - t_start)*1000:.1f}ms")
            if t_stt_complete:
                logger.info(f"  STT complete: {(t_stt_complete - t_start)*1000:.1f}ms")
            if t_llm_first_token:
                logger.info(f"  LLM first token: {(t_llm_first_token - t_start)*1000:.1f}ms")
            if t_first_phrase:
                logger.info(f"  First phrase boundary: {(t_first_phrase - t_start)*1000:.1f}ms")
            if t_first_tts_audio:
                logger.info(f"  First TTS audio chunk: {(t_first_tts_audio - t_start)*1000:.1f}ms ⭐ TIME TO FIRST AUDIO")

            # Done with this turn
            self.session.set_state(VoiceState.IDLE)
            await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())

        except Exception as e:
            logger.error(f"Error processing turn: {e}", exc_info=True)
            await websocket.send_json(
                ErrorMessage(message=f"Processing error: {str(e)}").model_dump()
            )
            self.session.set_state(VoiceState.IDLE)

    async def process_text_turn(self, websocket: WebSocket, text: str) -> None:
        """Process a text input turn (bypasses STT).

        This is similar to process_turn but skips the transcription step,
        directly using the provided text for LLM generation and TTS.

        Args:
            websocket: Client WebSocket connection.
            text: User's text input.
        """
        t_start = time.perf_counter()

        # Clear any previous interrupt
        self.session.clear_interrupt()
        self.session.reset_phrase_counter()

        # Update state to processing
        self.session.set_state(VoiceState.PROCESSING)
        await websocket.send_json(StatusMessage(state=VoiceState.PROCESSING).model_dump())

        try:
            # Send the text as a "transcription" so client can display it
            await websocket.send_json(
                TranscriptionMessage(text=text, is_final=True).model_dump()
            )

            # STT-only mode: skip LLM and TTS, return to idle
            if self.session.config.stt_only:
                logger.debug("STT-only mode, skipping LLM/TTS for text input")
                self.session.set_state(VoiceState.IDLE)
                await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())
                return

            # Go directly to LLM + TTS
            self.session.set_state(VoiceState.SPEAKING)
            await websocket.send_json(StatusMessage(state=VoiceState.SPEAKING).model_dump())

            # Use phrase detector to accumulate LLM tokens
            phrase_detector = PhraseBoundaryDetector(
                sentence_boundary_only=self.session.config.sentence_boundary_only,
            )
            thinking_filter = StreamingThinkingFilter()
            full_response = ""

            async for output in self.stream_llm_response(text):
                # Check for interrupt
                if self.session.is_interrupted():
                    logger.info("Turn interrupted by user")
                    break

                # Skip tool calls in text input mode (for now)
                if isinstance(output, LLMToolCall):
                    continue

                # Handle regular content tokens
                if isinstance(output, LLMContent):
                    token = output.text

                    # Filter thinking content at token level
                    filtered_token = thinking_filter.filter_token(token)
                    full_response += token

                    if not filtered_token:
                        continue

                    # Detect phrase boundaries
                    phrase = phrase_detector.add_token(filtered_token)
                    if phrase:
                        await websocket.send_json(
                            LLMTextMessage(text=phrase, is_final=False).model_dump()
                        )
                        await self._synthesize_and_stream_phrase(websocket, phrase)

                        if self.session.is_interrupted():
                            break

            # Flush remaining text
            if not self.session.is_interrupted():
                remaining_filtered = thinking_filter.flush()
                if remaining_filtered:
                    phrase = phrase_detector.add_token(remaining_filtered)
                    if phrase:
                        await websocket.send_json(
                            LLMTextMessage(text=phrase, is_final=False).model_dump()
                        )
                        await self._synthesize_and_stream_phrase(websocket, phrase)

                remaining = phrase_detector.flush()
                if remaining:
                    await websocket.send_json(
                        LLMTextMessage(text=remaining, is_final=True).model_dump()
                    )
                    await self._synthesize_and_stream_phrase(websocket, remaining)
                else:
                    # Always send is_final=True to signal end of LLM response
                    # This ensures the client can add the user message to history
                    await websocket.send_json(
                        LLMTextMessage(text="", is_final=True).model_dump()
                    )

            t_end = time.perf_counter()
            logger.info(f"Text turn completed in {(t_end - t_start)*1000:.1f}ms")

            self.session.set_state(VoiceState.IDLE)
            await websocket.send_json(StatusMessage(state=VoiceState.IDLE).model_dump())

        except Exception as e:
            logger.error(f"Error processing text turn: {e}", exc_info=True)
            await websocket.send_json(
                ErrorMessage(message=f"Processing error: {str(e)}").model_dump()
            )
            self.session.set_state(VoiceState.IDLE)

    async def _synthesize_and_stream_phrase(
        self,
        websocket: WebSocket,
        phrase: str,
        track_first_audio: bool = False,
        turn_start_time: float | None = None,
    ) -> float | None:
        """Synthesize a phrase and stream audio to client.

        Args:
            websocket: Client WebSocket connection.
            phrase: Text to synthesize.
            track_first_audio: If True, log timing for first audio chunk.
            turn_start_time: Start time of the turn for timing calculations.

        Returns:
            Time of first audio chunk if track_first_audio is True, None otherwise.
        """
        # Filter thinking tags and preprocess for natural speech
        original_phrase = phrase
        phrase = self._filter_thinking_tags(phrase)
        phrase = self._preprocess_for_speech(phrase)
        if not phrase:
            # Skip empty phrases (e.g., if it was all thinking/markdown content)
            return None

        # DEBUG: Log what's being sent to TTS
        logger.info(f"🔊 TTS INPUT: {repr(phrase[:200])}{'...' if len(phrase) > 200 else ''}")
        if phrase != original_phrase:
            logger.debug(f"🔊 TTS (original was): {repr(original_phrase[:200])}")

        phrase_index = self.session.next_phrase_index()
        t_first_audio = None
        first_chunk_logged = False

        # Timing for TTS request
        t_tts_start = time.perf_counter()

        # Notify phrase TTS starting
        await websocket.send_json(
            TTSStartMessage(phrase_index=phrase_index).model_dump()
        )

        total_samples = 0

        async for audio_chunk in self.synthesize_phrase_stream(phrase, phrase_index):
            # Track first audio chunk timing
            if track_first_audio and not first_chunk_logged:
                t_first_audio = time.perf_counter()
                first_chunk_logged = True
                tts_latency = (t_first_audio - t_tts_start) * 1000
                if turn_start_time:
                    total_latency = (t_first_audio - turn_start_time) * 1000
                    logger.info(f"⏱️ TIMING: First TTS audio: {total_latency:.1f}ms total, {tts_latency:.1f}ms TTS latency")
                else:
                    logger.info(f"⏱️ TIMING: TTS latency to first chunk: {tts_latency:.1f}ms")

            # Check for interrupt
            if self.session.is_interrupted():
                break

            # Send audio chunk as binary
            await websocket.send_bytes(audio_chunk)

            # Track duration (PCM 16-bit = 2 bytes per sample)
            total_samples += len(audio_chunk) // 2

        # Calculate duration (24kHz sample rate)
        duration = total_samples / 24000.0

        # Log TTS synthesis time
        t_tts_end = time.perf_counter()
        logger.debug(f"⏱️ TTS phrase {phrase_index}: {(t_tts_end - t_tts_start)*1000:.1f}ms for {len(phrase)} chars, {duration:.2f}s audio")

        # Notify phrase TTS complete
        await websocket.send_json(
            TTSDoneMessage(phrase_index=phrase_index, duration=duration).model_dump()
        )

        return t_first_audio

    async def handle_interrupt(self, websocket: WebSocket) -> None:
        """Handle barge-in interrupt from client.

        Args:
            websocket: Client WebSocket connection.
        """
        self.session.request_interrupt()
        self.session.reset_barge_in_state()
        self.session.set_state(VoiceState.INTERRUPTED)
        await websocket.send_json(
            StatusMessage(state=VoiceState.INTERRUPTED).model_dump()
        )

        # Discard any audio that was buffered during TTS playback
        # This prevents echo/stale audio from being processed
        self.session.discard_audio()

        # Transition to listening for new input
        self.session.set_state(VoiceState.LISTENING)
        await websocket.send_json(
            StatusMessage(state=VoiceState.LISTENING).model_dump()
        )

    async def cleanup(self) -> None:
        """Clean up service resources.

        Call this when the voice session ends to properly close
        the TTS WebSocket connection.
        """
        await self._close_tts_websocket()
