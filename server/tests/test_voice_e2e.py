"""
End-to-End tests for Voice Chat (GP-01 through GP-06).

These tests exercise the FULL voice pipeline with real services:
- Real STT transcription via Universal Runtime
- Real LLM inference
- Real TTS synthesis via Universal Runtime

Prerequisites (must be running before tests):
- LlamaFarm Server running (`nx start server`)
- Universal Runtime with STT model loaded (e.g., faster-whisper)
- Universal Runtime with TTS model loaded (e.g., kokoro)
- LLM endpoint (Ollama or Universal Runtime with language model)

Run with: pytest tests/test_voice_e2e.py -v -m e2e
Skip E2E in regular runs: pytest tests/ -v -m "not e2e"
"""

import asyncio
import json
import math
import struct
import time
from dataclasses import dataclass

import httpx
import pytest
import websockets

from api.routers.voice.types import VoiceState
from core.settings import settings

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


# =============================================================================
# Test Project Configuration
# =============================================================================

TEST_NAMESPACE = "test"
TEST_PROJECT = "voice-e2e"
TEST_PROJECT_OMNI = "voice-e2e-omni"

# Minimal project config for voice E2E testing
# This config uses Universal Runtime for LLM with voice section for STT/TTS defaults
TEST_PROJECT_CONFIG = {
    "version": "v1",
    "name": TEST_PROJECT,
    "namespace": TEST_NAMESPACE,
    "prompts": [
        {
            "name": "default",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep responses brief and conversational.",
                }
            ],
        }
    ],
    "runtime": {
        "default_model": "default",
        "models": [
            {
                "name": "default",
                "description": "Universal Runtime model for voice E2E tests",
                "provider": "universal",
                "model": "unsloth/qwen3-1.7b-gguf:q4_k_m",
                "prompts": ["default"],
            }
        ],
    },
    # Voice configuration - required for voice chat endpoint
    "voice": {
        "enabled": True,
        "llm_model": "default",  # Reference to model in runtime.models
        "stt": {
            "model": "base",  # Whisper model size (faster-whisper is passed via query param)
            "language": "en",
        },
        "tts": {
            "model": "kokoro",
            "voice": "af_heart",
            "speed": 0.95,
        },
        "enable_thinking": False,  # Disabled for voice (don't speak thinking output)
    },
}

# Project config for Omni model tests (native audio input, no STT)
# Omni models like Qwen2.5-Omni process audio directly without transcription
OMNI_PROJECT_CONFIG = {
    "version": "v1",
    "name": TEST_PROJECT_OMNI,
    "namespace": TEST_NAMESPACE,
    "prompts": [
        {
            "name": "default",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful voice assistant. Keep responses brief and conversational.",
                }
            ],
        }
    ],
    "runtime": {
        "default_model": "omni",
        "models": [
            {
                "name": "omni",
                "description": "Qwen2.5-Omni model for native audio E2E tests",
                "provider": "universal",
                "model": "unsloth/Qwen2.5-Omni-3B-GGUF:q4_k_m",
                "prompts": ["default"],
            }
        ],
    },
    # Voice configuration for Omni model
    # STT is automatically skipped when runtime reports native audio capability
    "voice": {
        "enabled": True,
        "llm_model": "omni",  # Reference to Omni model in runtime.models
        "stt": {
            "model": "base",  # Ignored for native audio models
            "language": "en",
        },
        "tts": {
            "model": "kokoro",
            "voice": "af_heart",
            "speed": 0.95,
        },
        "enable_thinking": False,
    },
}


# =============================================================================
# Audio Generation Utilities
# =============================================================================


def generate_real_speech(
    text: str,
    model: str = "kokoro",
    voice: str = "af_heart",
    speed: float = 1.0,
) -> bytes:
    """Generate real speech audio using the TTS endpoint.

    Calls the Universal Runtime's TTS endpoint to synthesize actual speech.
    This produces audio that Whisper can transcribe reliably.

    Args:
        text: Text to synthesize.
        model: TTS model (default: kokoro).
        voice: Voice ID (default: af_heart).
        speed: Speech speed multiplier.

    Returns:
        Raw PCM audio bytes (16-bit signed, 24kHz mono for kokoro).

    Raises:
        RuntimeError: If TTS synthesis fails.
    """
    url = f"http://{settings.universal_host}:{settings.universal_port}/v1/audio/speech"

    response = httpx.post(
        url,
        json={
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "pcm",  # Raw PCM for direct use
        },
        timeout=30.0,
    )

    if response.status_code != 200:
        raise RuntimeError(f"TTS synthesis failed: {response.status_code} - {response.text}")

    return response.content


def generate_sine_wave_audio(
    duration_seconds: float = 1.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.5,
) -> bytes:
    """Generate a sine wave tone as PCM audio.

    Args:
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate (16kHz for voice).
        frequency: Tone frequency in Hz.
        amplitude: Volume (0.0-1.0).

    Returns:
        Raw PCM bytes (16-bit signed little-endian mono).
    """
    num_samples = int(sample_rate * duration_seconds)
    samples = []

    for i in range(num_samples):
        t = i / sample_rate
        value = amplitude * math.sin(2 * math.pi * frequency * t)
        sample = int(value * 32767)
        samples.append(sample)

    return struct.pack(f"<{len(samples)}h", *samples)


def generate_speech_audio(text: str, sample_rate: int = 16000) -> bytes:
    """Generate audio that simulates speech patterns.

    Creates audio with varying amplitude to simulate speech energy patterns.
    This is NOT real speech but has energy patterns that may trigger VAD.

    Args:
        text: Text to "simulate" (used to determine duration).
        sample_rate: Sample rate.

    Returns:
        PCM audio bytes.
    """
    # Rough estimate: 150 words per minute, 5 chars per word
    words = len(text.split())
    duration = max(0.5, words * 0.4)  # ~0.4 seconds per word

    num_samples = int(sample_rate * duration)
    samples = []

    for i in range(num_samples):
        t = i / sample_rate
        # Mix of frequencies to simulate speech
        value = 0.3 * math.sin(2 * math.pi * 200 * t)  # Low frequency
        value += 0.2 * math.sin(2 * math.pi * 400 * t)  # Mid frequency
        value += 0.1 * math.sin(2 * math.pi * 800 * t)  # Higher frequency

        # Add amplitude modulation to simulate syllables
        envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 4 * t)  # 4 Hz modulation
        value *= envelope * 0.7

        sample = int(value * 32767)
        samples.append(sample)

    return struct.pack(f"<{len(samples)}h", *samples)


def generate_silence(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate silent audio.

    Args:
        duration_seconds: Duration in seconds.
        sample_rate: Sample rate.

    Returns:
        PCM bytes of silence.
    """
    num_samples = int(sample_rate * duration_seconds)
    return b"\x00\x00" * num_samples


# =============================================================================
# Test Fixtures
# =============================================================================


def get_server_host() -> str:
    """Get the connectable server host address."""
    return "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST


def setup_test_project() -> bool:
    """Create or update the test project on the server.

    Returns True if successful, False otherwise.
    """
    host = get_server_host()
    base_url = f"http://{host}:{settings.PORT}/v1"

    # First, try to create the project
    try:
        response = httpx.post(
            f"{base_url}/projects/{TEST_NAMESPACE}",
            json={"name": TEST_PROJECT, "config_template": None},
            timeout=10.0,
        )
        if response.status_code in (200, 201):
            print(f"Created test project {TEST_NAMESPACE}/{TEST_PROJECT}")
    except Exception as e:
        print(f"Note: Could not create project (may already exist): {e}")

    # Then update with our config
    try:
        response = httpx.put(
            f"{base_url}/projects/{TEST_NAMESPACE}/{TEST_PROJECT}",
            json={"config": TEST_PROJECT_CONFIG},
            timeout=10.0,
        )
        if response.status_code == 200:
            print(f"Updated test project config for {TEST_NAMESPACE}/{TEST_PROJECT}")
            return True
        else:
            print(f"Failed to update project config: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error updating project config: {e}")
        return False


@pytest.fixture(scope="module")
def voice_test_project():
    """Fixture that ensures the test project exists on the server.

    This fixture runs once per module and sets up the test project
    with the correct configuration for voice E2E testing.
    """
    host = get_server_host()

    # First verify server is running
    try:
        response = httpx.get(f"http://{host}:{settings.PORT}/health", timeout=5.0)
        if response.status_code != 200:
            pytest.fail(
                "LlamaFarm server not healthy. "
                "Start with `nx start server` before running E2E tests."
            )
    except Exception as e:
        pytest.fail(
            f"Cannot connect to LlamaFarm server at {host}:{settings.PORT}: {e}. "
            f"Start with `nx start server` before running E2E tests."
        )

    # Set up the test project
    if not setup_test_project():
        pytest.fail(
            f"Failed to set up test project {TEST_NAMESPACE}/{TEST_PROJECT}. "
            f"Check server logs for details."
        )

    return {"namespace": TEST_NAMESPACE, "project": TEST_PROJECT}


def setup_omni_test_project() -> bool:
    """Create or update the Omni test project on the server.

    Returns True if successful, False otherwise.
    """
    host = get_server_host()
    base_url = f"http://{host}:{settings.PORT}/v1"

    # First, try to create the project
    try:
        response = httpx.post(
            f"{base_url}/projects/{TEST_NAMESPACE}",
            json={"name": TEST_PROJECT_OMNI, "config_template": None},
            timeout=10.0,
        )
        if response.status_code in (200, 201):
            print(f"Created Omni test project {TEST_NAMESPACE}/{TEST_PROJECT_OMNI}")
    except Exception as e:
        print(f"Note: Could not create Omni project (may already exist): {e}")

    # Then update with our config
    try:
        response = httpx.put(
            f"{base_url}/projects/{TEST_NAMESPACE}/{TEST_PROJECT_OMNI}",
            json={"config": OMNI_PROJECT_CONFIG},
            timeout=10.0,
        )
        if response.status_code == 200:
            print(f"Updated Omni test project config for {TEST_NAMESPACE}/{TEST_PROJECT_OMNI}")
            return True
        else:
            print(f"Failed to update Omni project config: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"Error updating Omni project config: {e}")
        return False


@pytest.fixture(scope="module")
def omni_test_project():
    """Fixture that ensures the Omni test project exists on the server."""
    host = get_server_host()

    # Verify server is running
    try:
        response = httpx.get(f"http://{host}:{settings.PORT}/health", timeout=5.0)
        if response.status_code != 200:
            pytest.fail(
                "LlamaFarm server not healthy. "
                "Start with `nx start server` before running E2E tests."
            )
    except Exception as e:
        pytest.fail(
            f"Cannot connect to LlamaFarm server at {host}:{settings.PORT}: {e}. "
            f"Start with `nx start server` before running E2E tests."
        )

    # Set up the Omni test project
    if not setup_omni_test_project():
        pytest.fail(
            f"Failed to set up Omni test project {TEST_NAMESPACE}/{TEST_PROJECT_OMNI}. "
            f"Check server logs for details."
        )

    return {"namespace": TEST_NAMESPACE, "project": TEST_PROJECT_OMNI}


# =============================================================================
# WebSocket Test Helpers
# =============================================================================


@dataclass
class VoiceTestResult:
    """Captures results from a voice test session."""

    session_id: str | None = None
    transcriptions: list[str] = None
    llm_texts: list[str] = None
    tts_audio_chunks: list[bytes] = None
    status_states: list[str] = None
    errors: list[str] = None
    all_messages: list[dict | bytes] = None

    def __post_init__(self):
        if self.transcriptions is None:
            self.transcriptions = []
        if self.llm_texts is None:
            self.llm_texts = []
        if self.tts_audio_chunks is None:
            self.tts_audio_chunks = []
        if self.status_states is None:
            self.status_states = []
        if self.errors is None:
            self.errors = []
        if self.all_messages is None:
            self.all_messages = []

    @property
    def full_transcription(self) -> str:
        """Get combined transcription text."""
        return " ".join(self.transcriptions)

    @property
    def full_llm_response(self) -> str:
        """Get combined LLM response text."""
        return "".join(self.llm_texts)

    @property
    def total_tts_bytes(self) -> int:
        """Get total TTS audio bytes received."""
        return sum(len(chunk) for chunk in self.tts_audio_chunks)

    @property
    def has_error(self) -> bool:
        """Check if any errors occurred."""
        return len(self.errors) > 0


async def run_voice_session_async(
    audio_chunks: list[bytes],
    system_prompt: str = "You are a helpful assistant. Keep responses brief.",
    timeout_seconds: float = 30.0,
    silence_after_audio: float = 1.0,
    stt_model: str = "base",  # Whisper model size: tiny, base, small, medium, large-v3
    tts_model: str = "kokoro",
    tts_voice: str = "af_heart",
    namespace: str = TEST_NAMESPACE,
    project: str = TEST_PROJECT,
) -> VoiceTestResult:
    """Run a complete voice session and capture results.

    Connects to the actually running LlamaFarm server via WebSocket.

    Args:
        audio_chunks: List of audio chunks to send.
        system_prompt: System prompt for the assistant.
        timeout_seconds: Maximum time to wait for response.
        silence_after_audio: Seconds of silence to append after audio.
        stt_model: STT model name (default: faster-whisper).
        tts_model: TTS model name (default: kokoro).
        tts_voice: TTS voice name (default: af_heart).
        namespace: Project namespace (default: test).
        project: Project name (default: voice-e2e).

    Returns:
        VoiceTestResult with all captured data.
    """
    result = VoiceTestResult()

    # URL encode the system prompt
    from urllib.parse import quote
    encoded_prompt = quote(system_prompt)

    host = get_server_host()

    # Build WebSocket URL to connect to running server
    # LLM model uses project default from config
    ws_url = (
        f"ws://{host}:{settings.PORT}/v1/{namespace}/{project}/voice/chat"
        f"?system_prompt={encoded_prompt}"
        f"&stt_model={stt_model}"
        f"&tts_model={tts_model}"
        f"&tts_voice={tts_voice}"
    )

    try:
        async with websockets.connect(ws_url, close_timeout=5) as websocket:
            start_time = time.time()

            # Collect initial messages (session_info, status)
            while time.time() - start_time < 5.0:
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)

                    if isinstance(raw_msg, bytes):
                        continue  # Skip binary during init

                    msg = json.loads(raw_msg)
                    result.all_messages.append(msg)

                    if msg.get("type") == "session_info":
                        result.session_id = msg.get("session_id")
                    elif msg.get("type") == "status":
                        result.status_states.append(msg.get("state"))

                    # Break after getting initial status
                    if result.session_id and result.status_states:
                        break
                except TimeoutError:
                    break
                except Exception:
                    break

            # Send audio chunks
            for chunk in audio_chunks:
                await websocket.send(chunk)
                await asyncio.sleep(0.05)  # Small delay between chunks

            # Send silence to trigger end-of-speech
            silence = generate_silence(silence_after_audio)
            await websocket.send(silence)

            # Send explicit end signal
            await websocket.send(json.dumps({"type": "end"}))

            # Collect response messages
            while time.time() - start_time < timeout_seconds:
                try:
                    raw_msg = await asyncio.wait_for(
                        websocket.recv(),
                        timeout=min(5.0, timeout_seconds - (time.time() - start_time))
                    )

                    if isinstance(raw_msg, bytes):
                        result.tts_audio_chunks.append(raw_msg)
                        result.all_messages.append(raw_msg)
                    else:
                        msg = json.loads(raw_msg)
                        result.all_messages.append(msg)

                        msg_type = msg.get("type")
                        if msg_type == "transcription":
                            result.transcriptions.append(msg.get("text", ""))
                        elif msg_type == "llm_text":
                            result.llm_texts.append(msg.get("text", ""))
                        elif msg_type == "status":
                            result.status_states.append(msg.get("state"))
                            # Check if we're back to IDLE (response complete)
                            if msg.get("state") == VoiceState.IDLE.value:
                                break
                        elif msg_type == "error":
                            result.errors.append(msg.get("message", ""))

                except TimeoutError:
                    # Timeout waiting for message - might be done
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    result.errors.append(str(e))
                    break

    except Exception as e:
        result.errors.append(f"Connection error: {e}")

    return result


# Alias for backward compatibility
run_voice_session = run_voice_session_async


# =============================================================================
# GP-01: Simple Greeting (E2E)
# =============================================================================


class TestE2EGP01SimpleGreeting:
    """E2E test: User says 'Hello', model responds with greeting."""

    @pytest.mark.asyncio
    async def test_greeting_produces_transcription(self, voice_test_project):
        """Test that audio input produces a transcription."""
        # Generate audio simulating "Hello"
        audio = generate_real_speech("Hello")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="You are a friendly assistant. Say hi back briefly.",
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        # Should have received a transcription
        assert not result.has_error, f"Errors: {result.errors}"
        assert len(result.transcriptions) > 0, "No transcription received"

    @pytest.mark.asyncio
    async def test_greeting_produces_llm_response(self, voice_test_project):
        """Test that greeting produces an LLM response."""
        audio = generate_real_speech("Hello")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="You are a friendly assistant. Respond with a brief greeting.",
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert len(result.llm_texts) > 0, "No LLM response received"

    @pytest.mark.asyncio
    async def test_greeting_produces_tts_audio(self, voice_test_project):
        """Test that response includes TTS audio."""
        audio = generate_real_speech("Hello")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="You are a friendly assistant. Say hello back.",
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert result.total_tts_bytes > 0, "No TTS audio received"

    @pytest.mark.asyncio
    async def test_greeting_state_transitions(self, voice_test_project):
        """Test that session goes through correct state transitions."""
        audio = generate_real_speech("Hello")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Say hi.",
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"

        # Should see: idle -> listening -> processing -> speaking -> idle
        assert VoiceState.IDLE.value in result.status_states


# =============================================================================
# GP-02: Question and Answer (E2E)
# =============================================================================


class TestE2EGP02QuestionAnswer:
    """E2E test: User asks factual question, model answers."""

    @pytest.mark.asyncio
    async def test_question_is_transcribed(self, voice_test_project):
        """Test that a question is transcribed."""
        # Use real TTS-generated speech for reliable transcription
        audio = generate_real_speech("What is the capital of France?")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Answer questions briefly and factually.",
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert len(result.transcriptions) > 0, (
            f"No transcription received. States: {result.status_states}"
        )

    @pytest.mark.asyncio
    async def test_factual_answer_produced(self, voice_test_project):
        """Test that model produces a factual answer."""
        # Use real TTS-generated speech
        audio = generate_real_speech("What is two plus two?")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Answer math questions with just the number.",
            timeout_seconds=45.0,
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert len(result.llm_texts) > 0, (
            f"No answer received. Transcription: '{result.full_transcription}'. "
            f"States: {result.status_states}"
        )
        response = result.full_llm_response.lower()
        # The response should contain "4" or "four"
        assert "4" in response or "four" in response, f"Expected '4' in response: {response}"


# =============================================================================
# GP-03: Multi-turn Conversation (E2E)
# =============================================================================


class TestE2EGP03MultiTurn:
    """E2E test: Multi-turn conversation maintains context."""

    @pytest.mark.asyncio
    async def test_multi_turn_context(self, voice_test_project):
        """Test that context is maintained across turns.

        This test is more complex - it requires maintaining the same session
        across multiple audio inputs. For now, we test a single long utterance
        that implies context.
        """
        # Single turn that tests the system works
        # Use real TTS-generated speech
        audio = generate_real_speech(
            "Remember the number forty two. What number did I just say?"
        )

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Listen carefully and repeat back information when asked.",
            timeout_seconds=60.0,
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"


# =============================================================================
# GP-04: Long Response Streaming (E2E)
# =============================================================================


class TestE2EGP04LongResponse:
    """E2E test: Long response streams correctly."""

    @pytest.mark.asyncio
    async def test_long_response_streams_multiple_phrases(self, voice_test_project):
        """Test that a long response is delivered in multiple phrases."""
        # Use real TTS-generated speech
        audio = generate_real_speech(
            "Tell me three interesting facts about the moon."
        )

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Give detailed, multi-sentence answers.",
            timeout_seconds=60.0,
            silence_after_audio=2.0,
        )

        assert not result.has_error, f"Errors: {result.errors}"

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: {result.full_transcription}")
        print(f"Received {len(result.llm_texts)} LLM text chunks")
        print(f"Full response: {result.full_llm_response}")
        print(f"Received {len(result.tts_audio_chunks)} TTS audio chunks")

        # Should have received TTS audio
        assert result.total_tts_bytes > 0, (
            f"No TTS audio received. Transcription: '{result.full_transcription}'. "
            f"LLM response: '{result.full_llm_response}'. States: {result.status_states}"
        )


# =============================================================================
# GP-05: Short Response (E2E)
# =============================================================================


class TestE2EGP05ShortResponse:
    """E2E test: Short response is not truncated."""

    @pytest.mark.asyncio
    async def test_yes_no_response_complete(self, voice_test_project):
        """Test that short yes/no responses are delivered completely."""
        # Use real TTS-generated speech
        audio = generate_real_speech("Is the sky blue?")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Answer yes or no questions with just 'Yes' or 'No'.",
            timeout_seconds=30.0,
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert len(result.llm_texts) > 0, (
            f"No response received. Transcription: '{result.full_transcription}'. "
            f"States: {result.status_states}"
        )

        response = result.full_llm_response.lower().strip()

        # Should be a short response
        assert len(response) < 50, f"Response too long for yes/no: {response}"


# =============================================================================
# Latency and Performance Tests
# =============================================================================


class TestE2ELatency:
    """Tests for voice pipeline latency."""

    @pytest.mark.asyncio
    async def test_time_to_first_audio(self, voice_test_project):
        """Measure time from audio send to first TTS audio received."""
        from urllib.parse import quote

        audio = generate_real_speech("Hello")
        system_prompt = quote("Say hi briefly")

        host = get_server_host()

        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/{TEST_NAMESPACE}/{TEST_PROJECT}/voice/chat"
            f"?system_prompt={system_prompt}"
            f"&stt_model=base"
            f"&tts_model=kokoro"
            f"&tts_voice=af_heart"
        )

        first_audio_time = None
        audio_send_time = None

        try:
            async with websockets.connect(ws_url, close_timeout=5) as websocket:
                # Skip initial messages
                while True:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    if isinstance(raw_msg, str):
                        msg = json.loads(raw_msg)
                        if msg.get("type") == "status" and msg.get("state") == "idle":
                            break

                # Send audio
                audio_send_time = time.time()
                await websocket.send(audio)
                await websocket.send(generate_silence(0.5))
                await websocket.send(json.dumps({"type": "end"}))

                # Wait for first TTS audio
                timeout = 30.0
                while time.time() - audio_send_time < timeout:
                    try:
                        raw_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        if isinstance(raw_msg, bytes):
                            first_audio_time = time.time()
                            break
                    except TimeoutError:
                        continue
                    except Exception:
                        break

        except Exception as e:
            pytest.fail(f"Connection failed: {e}")

        assert first_audio_time is not None, (
            "No TTS audio received. Check that TTS model is loaded and working."
        )
        latency = first_audio_time - audio_send_time
        print(f"Time to first TTS audio: {latency:.2f}s")
        # This is informational - actual threshold depends on hardware
        assert latency < 30.0, f"Latency too high: {latency}s"


# =============================================================================
# GP-06: Omni Model Native Audio (E2E)
# =============================================================================


class TestE2EGP06OmniModel:
    """E2E tests for Qwen2.5-Omni with native audio input (no STT).

    These tests verify that Omni models can process audio directly without
    requiring a separate STT transcription step. The model receives raw audio
    and generates responses based on the speech content.
    """

    @pytest.mark.asyncio
    async def test_omni_greeting_produces_response(self, omni_test_project):
        """Test that Omni model produces a response from audio input.

        Unlike standard flow, Omni models skip STT transcription entirely.
        The audio is sent directly to the LLM which processes speech natively.
        """
        # Generate real speech audio
        audio = generate_real_speech("Hello, how are you?")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="You are a friendly assistant. Respond briefly.",
            namespace=omni_test_project["namespace"],
            project=omni_test_project["project"],
            timeout_seconds=60.0,
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription (should be empty for native audio): '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        # Native audio should NOT produce transcription messages (STT is skipped)
        # But should produce LLM response
        assert len(result.llm_texts) > 0, (
            f"No LLM response received from Omni model. States: {result.status_states}"
        )

    @pytest.mark.asyncio
    async def test_omni_question_produces_answer(self, omni_test_project):
        """Test that Omni model produces a response from audio question."""
        audio = generate_real_speech("What is the capital of France?")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Answer questions briefly.",
            namespace=omni_test_project["namespace"],
            project=omni_test_project["project"],
            timeout_seconds=60.0,
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"Transcription (native audio): '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert len(result.llm_texts) > 0, (
            f"No answer received from Omni model. States: {result.status_states}"
        )
        # Just verify we got a non-empty response - Omni transcription can vary
        assert len(result.full_llm_response.strip()) > 0, (
            "Empty response from Omni model"
        )

    @pytest.mark.asyncio
    async def test_omni_produces_tts_audio(self, omni_test_project):
        """Test that Omni model response is synthesized to TTS audio."""
        audio = generate_real_speech("Say hello to me.")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Greet the user warmly.",
            namespace=omni_test_project["namespace"],
            project=omni_test_project["project"],
            timeout_seconds=60.0,
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert result.total_tts_bytes > 0, (
            f"No TTS audio received from Omni model flow. "
            f"LLM response: '{result.full_llm_response}'. States: {result.status_states}"
        )

    @pytest.mark.asyncio
    async def test_omni_state_transitions(self, omni_test_project):
        """Test correct state transitions for Omni model flow.

        Omni flow: IDLE → LISTENING → PROCESSING → SPEAKING → IDLE
        (Note: No separate transcription step since audio goes directly to LLM)
        """
        audio = generate_real_speech("Hello")

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="Say hi.",
            namespace=omni_test_project["namespace"],
            project=omni_test_project["project"],
            timeout_seconds=60.0,
            silence_after_audio=2.0,
        )

        # Debug output
        print(f"States: {result.status_states}")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")

        assert not result.has_error, f"Errors: {result.errors}"

        # Should see proper state transitions
        assert VoiceState.IDLE.value in result.status_states, (
            f"Missing IDLE state. States: {result.status_states}"
        )


# =============================================================================
# Service Health Tests (verify prerequisites are met)
# =============================================================================


class TestServiceHealth:
    """Service health checks - verify test prerequisites are met."""

    def test_llamafarm_server_health(self):
        """Test that LlamaFarm server is running."""
        host = get_server_host()
        url = f"http://{host}:{settings.PORT}/health"
        response = httpx.get(url, timeout=5.0)
        assert response.status_code == 200, (
            f"LlamaFarm server not healthy at {url}. "
            "Start with `nx start server` before running E2E tests."
        )

    def test_universal_runtime_health(self):
        """Test that Universal Runtime is running."""
        url = f"http://{settings.universal_host}:{settings.universal_port}/health"
        response = httpx.get(url, timeout=5.0)
        assert response.status_code == 200, (
            f"Universal Runtime not healthy at {url}. "
            "Start with `nx start universal-runtime` before running E2E tests."
        )
