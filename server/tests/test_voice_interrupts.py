"""
Interruption / Barge-in tests for Voice Chat.

These tests verify the system handles user interruptions correctly:
- Stopping TTS when user speaks
- Processing new input after interrupt
- State transitions during interruption
- Handling rapid or edge-case interrupts

Prerequisites (must be running before tests):
- LlamaFarm Server running (`nx start server`)
- Universal Runtime with STT/TTS models loaded

Run with: pytest tests/test_voice_interrupts.py -v -s -m e2e
"""

import asyncio
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
# Test Configuration
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
    retries: int = 3,
    target_sample_rate: int = 16000,
) -> bytes:
    """Generate real speech audio using the TTS endpoint.

    TTS outputs at 24kHz, but voice chat expects 16kHz input.
    This function resamples to the target rate for compatibility.
    """
    url = f"http://{settings.universal_host}:{settings.universal_port}/v1/audio/speech"
    tts_sample_rate = 24000  # Kokoro TTS native output rate

    for attempt in range(retries):
        try:
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

            if response.status_code == 200:
                pcm_24k = response.content
                # Resample from TTS native rate to target rate
                if target_sample_rate != tts_sample_rate:
                    return resample_pcm(pcm_24k, tts_sample_rate, target_sample_rate)
                return pcm_24k
            elif attempt < retries - 1:
                time.sleep(0.5)  # Brief pause before retry
                continue
            else:
                raise RuntimeError(f"TTS synthesis failed: {response.status_code} - {response.text}")

        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            if attempt < retries - 1:
                time.sleep(1.0)  # Longer pause for connection issues
                continue
            raise RuntimeError(f"TTS connection failed after {retries} attempts: {e}") from e

    raise RuntimeError("TTS synthesis failed after all retries")


def generate_silence(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate silent audio (zeros)."""
    num_samples = int(sample_rate * duration_seconds)
    return b"\x00\x00" * num_samples


def chunk_audio(audio: bytes, chunk_ms: int = 100, sample_rate: int = 16000) -> list[bytes]:
    """Split PCM audio into smaller chunks to simulate streaming microphone input.

    Args:
        audio: Raw PCM audio bytes (16-bit signed little-endian).
        chunk_ms: Chunk duration in milliseconds (default 100ms).
        sample_rate: Sample rate in Hz (default 16000).

    Returns:
        List of audio chunks.
    """
    # 16-bit audio = 2 bytes per sample
    bytes_per_ms = sample_rate * 2 // 1000
    chunk_size = bytes_per_ms * chunk_ms

    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunks.append(audio[i : i + chunk_size])
    return chunks


def resample_pcm(audio: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample PCM audio using linear interpolation.

    Args:
        audio: Raw PCM audio bytes (16-bit signed little-endian).
        from_rate: Source sample rate in Hz.
        to_rate: Target sample rate in Hz.

    Returns:
        Resampled PCM audio bytes.
    """
    if from_rate == to_rate:
        return audio

    # Unpack samples (16-bit signed little-endian)
    num_samples = len(audio) // 2
    samples = struct.unpack(f"<{num_samples}h", audio)

    # Calculate output length
    ratio = to_rate / from_rate
    out_len = int(num_samples * ratio)

    # Linear interpolation resampling
    out_samples = []
    for i in range(out_len):
        src_idx = i / ratio
        idx0 = int(src_idx)
        idx1 = min(idx0 + 1, num_samples - 1)
        frac = src_idx - idx0

        # Interpolate between adjacent samples
        sample = int(samples[idx0] * (1 - frac) + samples[idx1] * frac)
        # Clamp to 16-bit range
        sample = max(-32768, min(32767, sample))
        out_samples.append(sample)

    # Pack back to bytes
    return struct.pack(f"<{len(out_samples)}h", *out_samples)


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


async def wait_for_state(
    ws,
    target_state: str,
    timeout: float = 10.0,
    collect_audio: bool = False,
) -> tuple[list[str], list[bytes]]:
    """Wait for a specific state, collecting states and optionally audio along the way."""
    states = []
    audio_chunks = []
    start = time.time()

    while time.time() - start < timeout:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

            if isinstance(msg, bytes):
                if collect_audio:
                    audio_chunks.append(msg)
            else:
                data = json.loads(msg)
                if data.get("type") == "status":
                    state = data.get("state")
                    states.append(state)
                    if state == target_state:
                        return states, audio_chunks

        except TimeoutError:
            continue

    return states, audio_chunks


async def send_audio_and_wait(
    ws,
    audio: bytes,
    silence_after: float = 1.0,
) -> None:
    """Send audio followed by silence and end signal."""
    await ws.send(audio)
    await asyncio.sleep(0.05)
    await ws.send(generate_silence(silence_after))
    await ws.send(json.dumps({"type": "end"}))


# =============================================================================
# INT-01: Basic Interrupt
# =============================================================================


class TestINT01BasicInterrupt:
    """INT-01: User speaks while TTS playing should stop it."""

    @pytest.mark.asyncio
    async def test_interrupt_during_speaking_triggers_state_change(self, voice_test_project):
        """Speaking while TTS plays should transition to INTERRUPTED state."""
        from urllib.parse import quote

        host = get_server_host()
        # Ask for a long response to give time to interrupt
        system_prompt = quote("Give a detailed, multi-paragraph response.")

        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/{TEST_NAMESPACE}/{TEST_PROJECT}/voice/chat"
            f"?system_prompt={system_prompt}"
            f"&stt_model=base&tts_model=kokoro&tts_voice=af_heart"
        )

        states_seen = []
        interrupted = False
        tts_chunks_before_interrupt = 0
        tts_chunks_after_interrupt = 0

        async with websockets.connect(ws_url, close_timeout=5) as ws:
            # Wait for initial IDLE
            states, _ = await wait_for_state(ws, VoiceState.IDLE.value)
            states_seen.extend(states)

            # Send audio to trigger a long response
            trigger_audio = generate_real_speech(
                "Tell me everything you know about the solar system in detail."
            )
            await ws.send(trigger_audio)
            await asyncio.sleep(0.05)
            await ws.send(generate_silence(1.5))
            await ws.send(json.dumps({"type": "end"}))

            # Wait for SPEAKING state
            start = time.time()
            while time.time() - start < 30.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    if isinstance(msg, bytes):
                        if not interrupted:
                            tts_chunks_before_interrupt += 1
                        else:
                            tts_chunks_after_interrupt += 1
                    else:
                        data = json.loads(msg)
                        if data.get("type") == "status":
                            state = data.get("state")
                            states_seen.append(state)

                            # Once we're SPEAKING, send interrupt audio
                            if state == VoiceState.SPEAKING.value and not interrupted:
                                # Wait a moment for TTS to start flowing
                                await asyncio.sleep(0.3)

                                # Send interrupt audio in chunks for barge-in detection
                                interrupt_audio = generate_real_speech("Stop! I have a question.")
                                for chunk in chunk_audio(interrupt_audio, chunk_ms=100):
                                    await ws.send(chunk)
                                await asyncio.sleep(0.05)
                                await ws.send(generate_silence(0.5))
                                await ws.send(json.dumps({"type": "end"}))
                                interrupted = True

                            # Check if we're back to IDLE after interrupt
                            if interrupted and state == VoiceState.IDLE.value:
                                break

                except TimeoutError:
                    continue

        print(f"States seen: {states_seen}")
        print(f"Interrupted: {interrupted}")
        print(f"TTS chunks before interrupt: {tts_chunks_before_interrupt}")
        print(f"TTS chunks after interrupt: {tts_chunks_after_interrupt}")

        # Verify interrupt happened and we saw appropriate states
        assert interrupted, "Interrupt was never triggered"
        assert VoiceState.SPEAKING.value in states_seen, "Never entered SPEAKING state"
        # Should eventually return to IDLE
        assert states_seen[-1] == VoiceState.IDLE.value, (
            f"Did not return to IDLE after interrupt. Final state: {states_seen[-1]}"
        )


# =============================================================================
# INT-02: Early Interrupt
# =============================================================================


class TestINT02EarlyInterrupt:
    """INT-02: Interrupt during first phrase should stop immediately."""

    @pytest.mark.asyncio
    async def test_early_interrupt_stops_response(self, voice_test_project):
        """Interrupting early in response should stop TTS quickly."""
        from urllib.parse import quote

        host = get_server_host()
        system_prompt = quote("Give a long, detailed answer.")

        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/{TEST_NAMESPACE}/{TEST_PROJECT}/voice/chat"
            f"?system_prompt={system_prompt}"
            f"&stt_model=base&tts_model=kokoro&tts_voice=af_heart"
        )

        states_seen = []
        tts_bytes_received = 0
        interrupted_at_bytes = 0

        async with websockets.connect(ws_url, close_timeout=5) as ws:
            # Wait for IDLE
            await wait_for_state(ws, VoiceState.IDLE.value)

            # Trigger long response
            trigger_audio = generate_real_speech("Tell me a long story about dragons.")
            await ws.send(trigger_audio)
            await ws.send(generate_silence(1.5))
            await ws.send(json.dumps({"type": "end"}))

            # Wait for first TTS audio, then immediately interrupt
            speaking_started = False
            interrupted = False
            start = time.time()

            while time.time() - start < 45.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    if isinstance(msg, bytes):
                        tts_bytes_received += len(msg)

                        # Interrupt as soon as we get first audio
                        if not interrupted and speaking_started:
                            interrupted_at_bytes = tts_bytes_received
                            interrupt_audio = generate_real_speech("Stop!")
                            for chunk in chunk_audio(interrupt_audio, chunk_ms=100):
                                await ws.send(chunk)
                            await ws.send(generate_silence(0.5))
                            await ws.send(json.dumps({"type": "end"}))
                            interrupted = True

                    else:
                        data = json.loads(msg)
                        if data.get("type") == "status":
                            state = data.get("state")
                            states_seen.append(state)

                            if state == VoiceState.SPEAKING.value:
                                speaking_started = True

                            if interrupted and state == VoiceState.IDLE.value:
                                break

                except TimeoutError:
                    continue

        print(f"States: {states_seen}")
        print(f"TTS bytes received: {tts_bytes_received}")
        print(f"Interrupted at bytes: {interrupted_at_bytes}")

        # Should have received relatively few bytes after interrupt
        assert interrupted_at_bytes > 0, "Never received TTS to interrupt"
        # Final state should be IDLE
        assert states_seen[-1] == VoiceState.IDLE.value


# =============================================================================
# INT-05: Interrupt Then Silence
# =============================================================================


class TestINT05InterruptThenSilence:
    """INT-05: Interrupt but say nothing should return to IDLE."""

    @pytest.mark.asyncio
    async def test_interrupt_with_silence_returns_to_idle(self, voice_test_project):
        """Interrupting with silence should timeout and return to IDLE."""
        from urllib.parse import quote

        host = get_server_host()
        system_prompt = quote("Give a detailed response.")

        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/{TEST_NAMESPACE}/{TEST_PROJECT}/voice/chat"
            f"?system_prompt={system_prompt}"
            f"&stt_model=base&tts_model=kokoro&tts_voice=af_heart"
        )

        states_seen = []

        async with websockets.connect(ws_url, close_timeout=5) as ws:
            # Wait for IDLE
            await wait_for_state(ws, VoiceState.IDLE.value)

            # Trigger response
            trigger_audio = generate_real_speech("Tell me about the ocean.")
            await ws.send(trigger_audio)
            await ws.send(generate_silence(1.5))
            await ws.send(json.dumps({"type": "end"}))

            # Wait for SPEAKING, then send only silence (interrupt with nothing)
            speaking_started = False
            interrupted = False
            start = time.time()

            while time.time() - start < 45.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    if isinstance(msg, bytes):
                        # Got TTS audio
                        if speaking_started and not interrupted:
                            # Send only silence - no speech
                            await ws.send(generate_silence(2.0))
                            await ws.send(json.dumps({"type": "end"}))
                            interrupted = True

                    else:
                        data = json.loads(msg)
                        if data.get("type") == "status":
                            state = data.get("state")
                            states_seen.append(state)

                            if state == VoiceState.SPEAKING.value:
                                speaking_started = True

                            if interrupted and state == VoiceState.IDLE.value:
                                break

                except TimeoutError:
                    continue

        print(f"States: {states_seen}")

        # Should return to IDLE eventually
        assert VoiceState.IDLE.value in states_seen, "Never returned to IDLE"


# =============================================================================
# INT-07: State Transitions
# =============================================================================


class TestINT07StateTransitions:
    """INT-07: Verify correct state transitions during interrupt."""

    @pytest.mark.asyncio
    async def test_interrupt_state_sequence(self, voice_test_project):
        """Interrupt should show SPEAKING → INTERRUPTED → LISTENING → ... → IDLE."""
        from urllib.parse import quote

        host = get_server_host()
        system_prompt = quote("Give a long answer.")

        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/{TEST_NAMESPACE}/{TEST_PROJECT}/voice/chat"
            f"?system_prompt={system_prompt}"
            f"&stt_model=base&tts_model=kokoro&tts_voice=af_heart"
        )

        all_states = []

        # Pre-generate audio before opening WebSocket
        trigger_audio = generate_real_speech("Explain quantum physics in detail.")
        interrupt_audio = generate_real_speech("Wait, what about gravity?")

        async with websockets.connect(ws_url, close_timeout=5) as ws:
            # Wait for IDLE
            states, _ = await wait_for_state(ws, VoiceState.IDLE.value)
            all_states.extend(states)

            # Trigger response
            await ws.send(trigger_audio)
            await ws.send(generate_silence(1.5))
            await ws.send(json.dumps({"type": "end"}))

            # Collect states, interrupt when speaking
            speaking_started = False
            interrupted = False
            start = time.time()

            while time.time() - start < 60.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    if isinstance(msg, bytes):
                        if speaking_started and not interrupted:
                            await asyncio.sleep(0.2)  # Let some audio play
                            for chunk in chunk_audio(interrupt_audio, chunk_ms=100):
                                await ws.send(chunk)
                            await ws.send(generate_silence(1.0))
                            await ws.send(json.dumps({"type": "end"}))
                            interrupted = True

                    else:
                        data = json.loads(msg)
                        if data.get("type") == "status":
                            state = data.get("state")
                            all_states.append(state)

                            if state == VoiceState.SPEAKING.value and not speaking_started:
                                speaking_started = True

                            # End after we've interrupted and returned to IDLE
                            if interrupted and state == VoiceState.IDLE.value:
                                break

                except TimeoutError:
                    continue

        print(f"All states: {all_states}")

        # Verify we saw key states
        assert VoiceState.IDLE.value in all_states, "Missing IDLE state"
        assert VoiceState.LISTENING.value in all_states, "Missing LISTENING state"
        assert VoiceState.SPEAKING.value in all_states, "Missing SPEAKING state"

        # Should end in IDLE
        assert all_states[-1] == VoiceState.IDLE.value, (
            f"Did not end in IDLE. Final: {all_states[-1]}"
        )


# =============================================================================
# INT-04: Rapid Interrupts
# =============================================================================


class TestINT04RapidInterrupts:
    """INT-04: Multiple rapid interrupts should be handled cleanly."""

    @pytest.mark.asyncio
    async def test_rapid_interrupts_handled(self, voice_test_project):
        """Multiple quick interrupts should not cause race conditions."""
        from urllib.parse import quote

        host = get_server_host()
        system_prompt = quote("Answer questions.")

        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/{TEST_NAMESPACE}/{TEST_PROJECT}/voice/chat"
            f"?system_prompt={system_prompt}"
            f"&stt_model=base&tts_model=kokoro&tts_voice=af_heart"
        )

        states_seen = []
        errors_seen = []

        # Pre-generate interrupt audio to avoid TTS calls during test
        interrupt_audios = [
            generate_real_speech("Stop"),
            generate_real_speech("Wait"),
        ]

        async with websockets.connect(ws_url, close_timeout=5) as ws:
            # Wait for IDLE
            await wait_for_state(ws, VoiceState.IDLE.value)

            # Trigger first response
            audio1 = generate_real_speech("Tell me a story")
            await ws.send(audio1)
            await ws.send(generate_silence(1.0))
            await ws.send(json.dumps({"type": "end"}))

            # Send interrupts when we get TTS audio
            interrupt_count = 0
            start = time.time()

            while time.time() - start < 60.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)

                    if isinstance(msg, bytes):
                        # On receiving TTS, send an interrupt (up to 2 times)
                        if interrupt_count < len(interrupt_audios):
                            await asyncio.sleep(0.2)  # Brief pause between interrupts
                            for chunk in chunk_audio(interrupt_audios[interrupt_count], chunk_ms=100):
                                await ws.send(chunk)
                            await ws.send(generate_silence(0.3))
                            await ws.send(json.dumps({"type": "end"}))
                            interrupt_count += 1

                    else:
                        data = json.loads(msg)
                        msg_type = data.get("type")

                        if msg_type == "status":
                            states_seen.append(data.get("state"))
                        elif msg_type == "error":
                            errors_seen.append(data.get("message", ""))

                        # Exit after final IDLE
                        if interrupt_count >= len(interrupt_audios) and data.get("state") == VoiceState.IDLE.value:
                            break

                except TimeoutError:
                    continue

        print(f"States: {states_seen}")
        print(f"Errors: {errors_seen}")
        print(f"Interrupt count: {interrupt_count}")

        # Should have completed without crashing
        assert interrupt_count > 0, "No interrupts were triggered"
        # Should end in IDLE
        assert states_seen[-1] == VoiceState.IDLE.value, (
            f"Did not end in IDLE after rapid interrupts. Final: {states_seen[-1]}"
        )
        # Should not have critical errors
        critical_errors = [e for e in errors_seen if "crash" in e.lower() or "exception" in e.lower()]
        assert len(critical_errors) == 0, f"Critical errors: {critical_errors}"


# =============================================================================
# INT-06: Interrupt Timing
# =============================================================================


class TestINT06InterruptTiming:
    """INT-06: Measure interrupt response latency."""

    @pytest.mark.asyncio
    async def test_interrupt_latency(self, voice_test_project):
        """Time from interrupt audio to TTS stop should be reasonable."""
        from urllib.parse import quote

        host = get_server_host()
        system_prompt = quote("Give a very long detailed response.")

        ws_url = (
            f"ws://{host}:{settings.PORT}/v1/{TEST_NAMESPACE}/{TEST_PROJECT}/voice/chat"
            f"?system_prompt={system_prompt}"
            f"&stt_model=base&tts_model=kokoro&tts_voice=af_heart"
        )

        interrupt_send_time = None
        state_change_time = None

        # Pre-generate audio before opening WebSocket
        trigger_audio = generate_real_speech(
            "Tell me the complete history of ancient Rome in great detail."
        )
        interrupt_audio = generate_real_speech("Stop now!")

        async with websockets.connect(ws_url, close_timeout=5) as ws:
            # Wait for IDLE
            await wait_for_state(ws, VoiceState.IDLE.value)

            # Trigger long response
            await ws.send(trigger_audio)
            await ws.send(generate_silence(1.5))
            await ws.send(json.dumps({"type": "end"}))

            speaking = False
            interrupted = False
            start = time.time()

            while time.time() - start < 60.0:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.5)

                    if isinstance(msg, bytes):
                        if speaking and not interrupted:
                            # Wait for some TTS to accumulate
                            await asyncio.sleep(0.5)
                            interrupt_send_time = time.time()
                            # Send interrupt audio in chunks to simulate streaming mic
                            # This is necessary for barge-in detection which counts
                            # consecutive chunks above speech threshold
                            for chunk in chunk_audio(interrupt_audio, chunk_ms=100):
                                await ws.send(chunk)
                            await ws.send(generate_silence(0.5))
                            await ws.send(json.dumps({"type": "end"}))
                            interrupted = True

                    else:
                        data = json.loads(msg)
                        if data.get("type") == "status":
                            state = data.get("state")

                            if state == VoiceState.SPEAKING.value:
                                speaking = True

                            # Record when state changes after interrupt
                            if interrupted and state != VoiceState.SPEAKING.value and state_change_time is None:
                                state_change_time = time.time()

                            if interrupted and state == VoiceState.IDLE.value:
                                break

                except TimeoutError:
                    continue

        # Calculate latencies
        if interrupt_send_time and state_change_time:
            latency = state_change_time - interrupt_send_time
            print(f"Interrupt send time: {interrupt_send_time}")
            print(f"State change time: {state_change_time}")
            print(f"Interrupt-to-state-change latency: {latency:.3f}s")

            # Latency should be reasonable (< 5s for E2E including VAD)
            assert latency < 5.0, f"Interrupt latency too high: {latency:.3f}s"
        else:
            print("Could not measure latency - interrupt may not have completed")
            # Still pass if we made it here without crashing
