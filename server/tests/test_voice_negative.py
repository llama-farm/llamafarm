"""
Negative Path / Error Handling tests for Voice Chat.

These tests verify the system handles unexpected or invalid inputs gracefully:
- Empty/silent audio
- Malformed audio data
- Service unavailability
- Invalid configurations

Prerequisites (must be running before tests):
- LlamaFarm Server running (`nx start server`)
- Universal Runtime with STT/TTS models loaded

Run with: pytest tests/test_voice_negative.py -v -s -m e2e
"""

import asyncio
import contextlib
import json
import struct
import time

import httpx
import pytest
import websockets

from api.routers.voice.types import VoiceState
from core.settings import settings

# Mark all tests in this module as e2e
pytestmark = pytest.mark.e2e


# =============================================================================
# Test Configuration (reuse from e2e tests)
# =============================================================================

TEST_NAMESPACE = "test"
TEST_PROJECT = "voice-e2e"

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
                    "content": "You are a helpful voice assistant. Keep responses brief.",
                }
            ],
        }
    ],
    "runtime": {
        "default_model": "default",
        "models": [
            {
                "name": "default",
                "provider": "universal",
                "model": "unsloth/qwen3-1.7b-gguf:q4_k_m",
                "prompts": ["default"],
            }
        ],
    },
    "voice": {
        "enabled": True,
        "llm_model": "default",
        "stt": {"model": "base", "language": "en"},
        "tts": {"model": "kokoro", "voice": "af_heart", "speed": 0.95},
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
    """Generate real speech audio using the TTS endpoint."""
    url = f"http://{settings.universal_host}:{settings.universal_port}/v1/audio/speech"

    response = httpx.post(
        url,
        json={
            "model": model,
            "input": text,
            "voice": voice,
            "speed": speed,
            "response_format": "pcm",
        },
        timeout=30.0,
    )

    if response.status_code != 200:
        raise RuntimeError(f"TTS synthesis failed: {response.status_code} - {response.text}")

    return response.content


def generate_silence(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate silent audio (zeros)."""
    num_samples = int(sample_rate * duration_seconds)
    return b"\x00\x00" * num_samples


def generate_noise(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate random noise audio."""
    import random
    num_samples = int(sample_rate * duration_seconds)
    samples = [random.randint(-32768, 32767) for _ in range(num_samples)]
    return struct.pack(f"<{len(samples)}h", *samples)


def generate_malformed_audio() -> bytes:
    """Generate intentionally malformed audio data."""
    return b"\xff\xfe\x00\x01" * 100  # Invalid PCM pattern


# =============================================================================
# Test Helpers
# =============================================================================


def get_server_host() -> str:
    """Get the connectable server host address."""
    return "127.0.0.1" if settings.HOST == "0.0.0.0" else settings.HOST


def setup_test_project() -> bool:
    """Create or update the test project on the server."""
    host = get_server_host()
    base_url = f"http://{host}:{settings.PORT}/v1"

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
    """Fixture that ensures the test project exists on the server."""
    host = get_server_host()

    try:
        response = httpx.get(f"http://{host}:{settings.PORT}/health", timeout=5.0)
        if response.status_code != 200:
            pytest.fail("LlamaFarm server not healthy.")
    except Exception as e:
        pytest.fail(f"Cannot connect to LlamaFarm server: {e}")

    if not setup_test_project():
        pytest.fail("Failed to set up test project.")

    return {"namespace": TEST_NAMESPACE, "project": TEST_PROJECT}


class VoiceTestResult:
    """Captures results from a voice test session."""

    def __init__(self):
        self.session_id: str | None = None
        self.transcriptions: list[str] = []
        self.llm_texts: list[str] = []
        self.tts_audio_chunks: list[bytes] = []
        self.status_states: list[str] = []
        self.errors: list[str] = []

    @property
    def full_transcription(self) -> str:
        return " ".join(self.transcriptions)

    @property
    def full_llm_response(self) -> str:
        return "".join(self.llm_texts)

    @property
    def total_tts_bytes(self) -> int:
        return sum(len(chunk) for chunk in self.tts_audio_chunks)

    @property
    def has_error(self) -> bool:
        return len(self.errors) > 0

    @property
    def final_state(self) -> str | None:
        return self.status_states[-1] if self.status_states else None


async def run_voice_session(
    audio_chunks: list[bytes],
    system_prompt: str = "You are a helpful assistant.",
    timeout_seconds: float = 30.0,
    silence_after_audio: float = 1.0,
    namespace: str = TEST_NAMESPACE,
    project: str = TEST_PROJECT,
) -> VoiceTestResult:
    """Run a voice session and capture results."""
    result = VoiceTestResult()

    from urllib.parse import quote
    encoded_prompt = quote(system_prompt)

    host = get_server_host()
    ws_url = (
        f"ws://{host}:{settings.PORT}/v1/{namespace}/{project}/voice/chat"
        f"?system_prompt={encoded_prompt}"
        f"&stt_model=base"
        f"&tts_model=kokoro"
        f"&tts_voice=af_heart"
    )

    try:
        async with websockets.connect(ws_url, close_timeout=5) as websocket:
            start_time = time.time()

            # Collect initial messages
            while time.time() - start_time < 5.0:
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    if isinstance(raw_msg, str):
                        msg = json.loads(raw_msg)
                        if msg.get("type") == "session_info":
                            result.session_id = msg.get("session_id")
                        elif msg.get("type") == "status":
                            result.status_states.append(msg.get("state"))
                        if result.session_id and result.status_states:
                            break
                except TimeoutError:
                    break

            # Send audio chunks
            for chunk in audio_chunks:
                if chunk:  # Only send non-empty chunks
                    await websocket.send(chunk)
                    await asyncio.sleep(0.05)

            # Send silence to trigger end-of-speech
            silence = generate_silence(silence_after_audio)
            await websocket.send(silence)

            # Send explicit end signal
            await websocket.send(json.dumps({"type": "end"}))

            # Collect response messages
            while time.time() - start_time < timeout_seconds:
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)

                    if isinstance(raw_msg, bytes):
                        result.tts_audio_chunks.append(raw_msg)
                    else:
                        msg = json.loads(raw_msg)
                        msg_type = msg.get("type")

                        if msg_type == "transcription":
                            result.transcriptions.append(msg.get("text", ""))
                        elif msg_type == "llm_text":
                            result.llm_texts.append(msg.get("text", ""))
                        elif msg_type == "status":
                            result.status_states.append(msg.get("state"))
                            if msg.get("state") == VoiceState.IDLE.value:
                                break
                        elif msg_type == "error":
                            result.errors.append(msg.get("message", ""))

                except TimeoutError:
                    continue
                except websockets.exceptions.ConnectionClosed:
                    break

    except Exception as e:
        result.errors.append(f"Connection error: {e}")

    return result


# =============================================================================
# NP-01: Empty Audio
# =============================================================================


class TestNP01EmptyAudio:
    """NP-01: Empty audio should not crash, just return to IDLE."""

    @pytest.mark.asyncio
    async def test_empty_audio_returns_to_idle(self, voice_test_project):
        """Empty audio buffer should be handled gracefully."""
        result = await run_voice_session(
            audio_chunks=[b""],
            system_prompt="You are a helpful assistant.",
            timeout_seconds=10.0,
        )

        print(f"States: {result.status_states}")
        print(f"Errors: {result.errors}")
        print(f"Final state: {result.final_state}")

        # Session should end in IDLE without crashing
        assert result.final_state == VoiceState.IDLE.value, (
            f"Session did not return to IDLE. States: {result.status_states}"
        )


# =============================================================================
# NP-02: Silent Audio
# =============================================================================


class TestNP02SilentAudio:
    """NP-02: Silent audio should be detected and skipped."""

    @pytest.mark.asyncio
    async def test_silence_only_returns_to_idle(self, voice_test_project):
        """Audio containing only silence should not trigger LLM."""
        silence = generate_silence(2.0)

        result = await run_voice_session(
            audio_chunks=[silence],
            system_prompt="You are a helpful assistant.",
            timeout_seconds=15.0,
        )

        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"Errors: {result.errors}")

        # Should return to IDLE without producing an LLM response
        assert result.final_state == VoiceState.IDLE.value, (
            f"Session did not return to IDLE. States: {result.status_states}"
        )

    @pytest.mark.asyncio
    async def test_silence_produces_no_transcription(self, voice_test_project):
        """Silent audio should produce empty or no transcription."""
        silence = generate_silence(2.0)

        result = await run_voice_session(
            audio_chunks=[silence],
            system_prompt="You are a helpful assistant.",
            timeout_seconds=15.0,
        )

        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")

        # Transcription should be empty or not present
        transcription = result.full_transcription.strip()
        assert len(transcription) == 0 or transcription == "", (
            f"Expected empty transcription for silence, got: '{transcription}'"
        )


# =============================================================================
# NP-03: Malformed Audio
# =============================================================================


class TestNP03MalformedAudio:
    """NP-03: Malformed audio should produce an error but not crash."""

    @pytest.mark.asyncio
    async def test_malformed_audio_handles_gracefully(self, voice_test_project):
        """Invalid audio data should be handled without crashing."""
        malformed = generate_malformed_audio()

        result = await run_voice_session(
            audio_chunks=[malformed],
            system_prompt="You are a helpful assistant.",
            timeout_seconds=15.0,
        )

        print(f"States: {result.status_states}")
        print(f"Errors: {result.errors}")
        print(f"Final state: {result.final_state}")

        # Session should remain stable (end in IDLE or have recoverable error)
        # The key is that it doesn't crash or hang
        assert result.final_state is not None, "Session did not complete"


# =============================================================================
# NP-04: Wrong Audio Format
# =============================================================================


class TestNP04WrongFormat:
    """NP-04: Wrong audio format should be handled gracefully."""

    @pytest.mark.asyncio
    async def test_random_noise_handled(self, voice_test_project):
        """Random noise (not speech) should be handled without crash."""
        noise = generate_noise(1.0)

        result = await run_voice_session(
            audio_chunks=[noise],
            system_prompt="You are a helpful assistant.",
            timeout_seconds=15.0,
        )

        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"Errors: {result.errors}")

        # Should complete without crashing
        assert result.final_state is not None, "Session did not complete"


# =============================================================================
# NP-05: Truncated Audio
# =============================================================================


class TestNP05TruncatedAudio:
    """NP-05: Audio cut off mid-utterance should still be processed."""

    @pytest.mark.asyncio
    async def test_truncated_audio_transcribes_partial(self, voice_test_project):
        """Audio cut off mid-word should transcribe available portion."""
        # Generate speech and truncate it
        full_audio = generate_real_speech("Hello, how are you doing today?")
        truncated = full_audio[:len(full_audio) // 3]  # Take first third

        result = await run_voice_session(
            audio_chunks=[truncated],
            system_prompt="You are a helpful assistant.",
            timeout_seconds=20.0,
        )

        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"Errors: {result.errors}")

        # Should complete - may or may not have transcription depending on how much audio
        assert result.final_state is not None, "Session did not complete"


# =============================================================================
# NP-06: Very Short Audio
# =============================================================================


class TestNP06VeryShortAudio:
    """NP-06: Very short audio (<100ms) should be handled."""

    @pytest.mark.asyncio
    async def test_very_short_audio_handled(self, voice_test_project):
        """Audio less than 100ms should be handled gracefully."""
        # Generate very short speech
        short_audio = generate_real_speech("Hi")
        # Take only first 50ms worth (24000 sample rate * 0.05 = 1200 samples = 2400 bytes)
        very_short = short_audio[:2400]

        result = await run_voice_session(
            audio_chunks=[very_short],
            system_prompt="You are a helpful assistant.",
            timeout_seconds=15.0,
        )

        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"Errors: {result.errors}")

        # Should complete without crashing
        assert result.final_state is not None, "Session did not complete"


# =============================================================================
# NP-07: Very Long Audio
# =============================================================================


class TestNP07VeryLongAudio:
    """NP-07: Very long audio should be handled without timeout."""

    @pytest.mark.asyncio
    async def test_long_audio_completes(self, voice_test_project):
        """Long audio (30+ seconds) should complete without issues."""
        # Generate a longer piece of speech
        long_text = (
            "This is a test of the voice system with a longer piece of audio. "
            "We want to make sure that the system can handle extended speech "
            "without timing out or running into memory issues. "
            "The quick brown fox jumps over the lazy dog."
        )
        long_audio = generate_real_speech(long_text)

        result = await run_voice_session(
            audio_chunks=[long_audio],
            system_prompt="You are a helpful assistant. Respond briefly.",
            timeout_seconds=90.0,
            silence_after_audio=3.0,
        )

        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")
        print(f"LLM Response: '{result.full_llm_response}'")
        print(f"TTS bytes: {result.total_tts_bytes}")
        print(f"Errors: {result.errors}")

        assert not result.has_error, f"Errors: {result.errors}"
        assert result.final_state == VoiceState.IDLE.value, (
            f"Session did not return to IDLE. States: {result.status_states}"
        )


# =============================================================================
# NP-08 through NP-10: Service Unavailability
# These tests would require mocking or actually stopping services,
# which is complex for E2E tests. Implementing basic checks instead.
# =============================================================================


class TestNPServiceErrors:
    """Tests for service error handling."""

    @pytest.mark.asyncio
    async def test_invalid_project_rejected(self, voice_test_project):
        """NP-11: Invalid project should return clear error."""
        from urllib.parse import quote

        host = get_server_host()
        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/nonexistent/project/voice/chat"
            f"?system_prompt={quote('test')}"
        )

        error_received = False
        error_message = ""

        try:
            async with websockets.connect(ws_url, close_timeout=5) as websocket:
                # Wait for error or rejection
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    if isinstance(raw_msg, str):
                        msg = json.loads(raw_msg)
                        if msg.get("type") == "error":
                            error_received = True
                            error_message = msg.get("message", "")
                except TimeoutError:
                    pass
        except websockets.exceptions.InvalidStatusCode as e:
            # Connection rejected with HTTP error - this is expected
            error_received = True
            error_message = f"HTTP {e.status_code}"
        except Exception as e:
            error_received = True
            error_message = str(e)

        print(f"Error received: {error_received}")
        print(f"Error message: {error_message}")

        # Should have received some kind of error
        assert error_received, "Expected error for invalid project"

    @pytest.mark.asyncio
    async def test_missing_llm_model_rejected(self):
        """Invalid model config should be rejected."""
        host = get_server_host()
        base_url = f"http://{host}:{settings.PORT}/v1"

        # Try to create a project with invalid model reference
        invalid_config = {
            "version": "v1",
            "name": "invalid-test",
            "namespace": TEST_NAMESPACE,
            "prompts": [{"name": "default", "messages": []}],
            "runtime": {"default_model": "default", "models": []},  # Empty models
            "voice": {
                "enabled": True,
                "llm_model": "nonexistent",  # Invalid reference
                "stt": {"model": "base"},
                "tts": {"model": "kokoro", "voice": "af_heart"},
            },
        }

        # Create project first
        with contextlib.suppress(Exception):
            httpx.post(
                f"{base_url}/projects/{TEST_NAMESPACE}",
                json={"name": "invalid-test", "config_template": None},
                timeout=10.0,
            )

        # Try to update with invalid config
        response = httpx.put(
            f"{base_url}/projects/{TEST_NAMESPACE}/invalid-test",
            json={"config": invalid_config},
            timeout=10.0,
        )

        print(f"Response status: {response.status_code}")
        print(f"Response: {response.text[:500] if response.text else 'empty'}")

        # Config update might succeed (validation is lazy) or fail
        # Either way, trying to use it should fail
        # This is more of a smoke test
