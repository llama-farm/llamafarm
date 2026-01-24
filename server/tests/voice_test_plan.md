# Voice Feature Test Plan

Test plan for the LlamaFarm voice assistant pipeline (STT → LLM → TTS).

## Overview

The voice system processes audio through three stages:
1. **STT**: Transcribe user audio via Universal Runtime (Whisper)
2. **LLM**: Generate response with phrase boundary detection
3. **TTS**: Synthesize speech and stream back to client

This plan covers functional correctness, edge cases, and resilience testing.

---

## Test Architecture

### Test Files

| File | Type | Description |
|------|------|-------------|
| `test_voice_golden_path.py` | Unit | Mocked tests for pipeline orchestration logic |
| `test_voice_e2e.py` | E2E | Real services, real audio, full pipeline |
| `test_voice_negative.py` | E2E | Error handling and edge cases |
| `test_voice_interrupts.py` | E2E | Barge-in and interruption scenarios |
| `test_voice_performance.py` | E2E | Latency and throughput measurements |

### Audio Generation Strategy

**Key Learning**: Synthetic sine-wave audio does NOT produce reliable Whisper transcriptions. Use one of:

1. **`generate_real_speech(text)`** - Calls TTS endpoint to synthesize actual speech (recommended for E2E)
2. **Pre-recorded audio fixtures** - WAV/PCM files in `tests/fixtures/voice/` (for reproducible CI)
3. **`generate_speech_audio(text)`** - Synthetic sine waves (only for VAD/timing tests, not transcription)

### Test Project Configuration

E2E tests use a dedicated project config pushed to the server:

```python
TEST_PROJECT_CONFIG = {
    "version": "v1",
    "name": "voice-e2e",
    "namespace": "test",
    "prompts": [{"name": "default", "messages": [...]}],
    "runtime": {
        "default_model": "default",
        "models": [{
            "name": "default",
            "provider": "universal",
            "model": "unsloth/qwen3-1.7b-gguf:q5_k_m",  # Standard LLM
            "prompts": ["default"],
        }],
    },
    "voice": {
        "enabled": True,
        "llm_model": "default",  # Reference to runtime.models
        "stt": {"model": "base", "language": "en"},
        "tts": {"model": "kokoro", "voice": "af_heart", "speed": 0.95},
        "enable_thinking": False,
    },
}
```

For Omni model tests, use `ggml-org/qwen2.5-omni-3b:q5_k_m`.

---

## 1. Golden Path Tests

**Status**: ✅ Implemented in `test_voice_e2e.py` and `test_voice_golden_path.py`

Basic conversational flows that must work reliably.

| ID | Test Case | Description | Expected Behavior | Priority | Status |
|----|-----------|-------------|-------------------|----------|--------|
| GP-01 | Simple greeting | User says "Hello" | Model responds with greeting, TTS plays | P0 | ✅ E2E |
| GP-02 | Question and answer | User asks "What is the capital of France?" | Model answers correctly, TTS speaks | P0 | ✅ E2E |
| GP-03 | Multi-turn conversation | Conversation with context reference | Model maintains context across turns | P0 | ✅ E2E |
| GP-04 | Long response | Ask for detailed explanation | Response streams in phrases, TTS plays each | P0 | ✅ E2E |
| GP-05 | Short response | Ask yes/no question | Single phrase response, no truncation | P0 | ✅ E2E |
| GP-06 | Omni model (native audio) | Audio directly to Omni model | Model understands, responds with TTS | P0 | ✅ E2E |
| GP-07 | Session warm-up | Connect and verify pre-warming | TTS WebSocket and HTTP client initialized | P1 | ⬜ TODO |

### E2E Test Pattern (GP tests)

```python
@pytest.mark.asyncio
async def test_greeting_produces_transcription(self, voice_test_project):
    # Generate REAL speech using TTS
    audio = generate_real_speech("Hello")

    result = await run_voice_session(
        audio_chunks=[audio],
        system_prompt="You are a friendly assistant.",
        silence_after_audio=2.0,  # Give VAD time to detect end-of-speech
    )

    # Debug output (always include for troubleshooting)
    print(f"States: {result.status_states}")
    print(f"Transcription: '{result.full_transcription}'")
    print(f"LLM Response: '{result.full_llm_response}'")
    print(f"TTS bytes: {result.total_tts_bytes}")

    assert not result.has_error, f"Errors: {result.errors}"
    assert len(result.transcriptions) > 0
```

---

## 2. Negative Path / Error Handling

**Status**: ✅ Implemented in `test_voice_negative.py`

How the system handles unexpected or invalid inputs.

| ID | Test Case | Description | Expected Behavior | Priority | Status |
|----|-----------|-------------|-------------------|----------|--------|
| NP-01 | Empty audio | Send zero-length audio buffer | Gracefully skip turn, return to IDLE | P0 | ✅ E2E |
| NP-02 | Silent audio | Send audio with only silence | STT returns empty, skip LLM, return to IDLE | P0 | ✅ E2E |
| NP-03 | Malformed audio | Send corrupt/invalid audio data | Error message sent to client, session stable | P0 | ✅ E2E |
| NP-04 | Wrong audio format | Random noise instead of speech | Handle gracefully or return clear error | P1 | ✅ E2E |
| NP-05 | Truncated audio | Audio cut off mid-utterance | STT transcribes available audio | P1 | ✅ E2E |
| NP-06 | Very short audio | <100ms of speech | Transcribe if possible, otherwise skip | P1 | ✅ E2E |
| NP-07 | Very long audio | >30s of continuous speech | Handle without timeout, memory stable | P1 | ✅ E2E |
| NP-08 | LLM unavailable | Runtime not responding | Error message, session remains usable | P0 | ⬜ TODO |
| NP-09 | TTS unavailable | TTS endpoint fails | Error message, text response still delivered | P0 | ⬜ TODO |
| NP-10 | STT unavailable | STT endpoint fails | Clear error, suggest retry | P0 | ⬜ TODO |
| NP-11 | Invalid session config | Missing model or invalid voice | Reject with descriptive error | P1 | ✅ E2E |

### Implementation Pattern (NP tests)

```python
class TestNegativePathErrors:
    """Tests for error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_empty_audio_returns_to_idle(self, voice_test_project):
        """NP-01: Empty audio should not crash, just return to IDLE."""
        audio = b""  # Empty audio buffer

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="You are a helpful assistant.",
        )

        print(f"States: {result.status_states}")
        print(f"Errors: {result.errors}")

        # Should end in IDLE without crashing
        assert VoiceState.IDLE.value in result.status_states
        # May or may not have error depending on implementation

    @pytest.mark.asyncio
    async def test_silent_audio_skips_llm(self, voice_test_project):
        """NP-02: Pure silence should not trigger LLM response."""
        audio = generate_silence(duration_seconds=3.0)

        result = await run_voice_session(
            audio_chunks=[audio],
            system_prompt="You are a helpful assistant.",
            silence_after_audio=1.0,
        )

        print(f"States: {result.status_states}")
        print(f"Transcription: '{result.full_transcription}'")

        # Transcription should be empty or very short
        assert len(result.full_transcription.strip()) < 5
        # Should return to IDLE
        assert VoiceState.IDLE.value in result.status_states

    @pytest.mark.asyncio
    async def test_llm_unavailable_returns_error(self, voice_test_project):
        """NP-08: LLM failure should return error but keep session alive."""
        # This test requires a way to make the LLM unavailable
        # Option 1: Use a config pointing to non-existent model
        # Option 2: Stop the runtime during test (harder)
        audio = generate_real_speech("Hello")

        # Create config with invalid model
        invalid_config = TEST_PROJECT_CONFIG.copy()
        invalid_config["runtime"]["models"][0]["model"] = "nonexistent/model:q4"
        # ... setup invalid project ...

        result = await run_voice_session(
            audio_chunks=[audio],
            namespace="test",
            project="voice-e2e-invalid",
        )

        # Should have error message
        assert result.has_error or "error" in str(result.all_messages).lower()
```

---

## 3. Interruption / Barge-in Tests

**Status**: ✅ Implemented in `test_voice_interrupts.py`

User interrupting the assistant mid-response.

| ID | Test Case | Description | Expected Behavior | Priority | Status |
|----|-----------|-------------|-------------------|----------|--------|
| INT-01 | Basic interrupt | User speaks while TTS playing | TTS stops, listen to new input | P0 | ✅ E2E |
| INT-02 | Early interrupt | Interrupt during first phrase | Stop immediately, process new input | P0 | ✅ E2E |
| INT-03 | Late interrupt | Interrupt near end of response | Stop remaining audio, process new input | P1 | ⬜ TODO |
| INT-04 | Rapid interrupts | Multiple interrupts in quick succession | Handle cleanly without race conditions | P1 | ✅ E2E |
| INT-05 | Interrupt then silence | Interrupt but say nothing | Return to IDLE after VAD timeout | P1 | ✅ E2E |
| INT-06 | Interrupt timing | Measure time from speech to TTS stop | <5s E2E latency | P1 | ✅ E2E |
| INT-07 | State transitions | Verify SPEAKING → INTERRUPTED → LISTENING | Correct state messages sent to client | P0 | ✅ E2E |

### Implementation Pattern (INT tests)

```python
class TestInterruptions:
    """Tests for barge-in and interruption handling."""

    @pytest.mark.asyncio
    async def test_basic_interrupt_stops_tts(self, voice_test_project):
        """INT-01: Speaking while TTS plays should interrupt it."""
        # First, trigger a long response
        trigger_audio = generate_real_speech(
            "Tell me a very long story about a dragon."
        )

        # We need to send audio mid-response, so use raw WebSocket
        host = get_server_host()
        ws_url = f"ws://{host}:{settings.PORT}/v1/test/voice-e2e/voice/chat"

        states_seen = []
        interrupted = False

        async with websockets.connect(ws_url) as ws:
            # Wait for initial IDLE
            await wait_for_state(ws, "idle")

            # Send trigger audio
            await ws.send(trigger_audio)
            await ws.send(generate_silence(1.0))
            await ws.send(json.dumps({"type": "end"}))

            # Wait for SPEAKING state (response starting)
            await wait_for_state(ws, "speaking")

            # Now send interrupt audio while speaking
            interrupt_audio = generate_real_speech("Stop!")
            await ws.send(interrupt_audio)
            await ws.send(generate_silence(0.5))
            await ws.send(json.dumps({"type": "end"}))

            # Should see INTERRUPTED state
            state = await wait_for_state(ws, "interrupted", timeout=2.0)
            assert state == "interrupted"

            # Should then process new input
            # ... collect new transcription and response ...

    @pytest.mark.asyncio
    async def test_interrupt_state_transitions(self, voice_test_project):
        """INT-07: Verify correct state flow during interrupt."""
        # Similar to above but focus on state sequence
        # Expected: idle -> listening -> processing -> speaking -> interrupted -> listening -> ...
        pass
```

### Helper Functions for Interrupt Tests

```python
async def wait_for_state(ws, target_state: str, timeout: float = 10.0) -> str:
    """Wait for a specific state message on the WebSocket."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
            if isinstance(msg, str):
                data = json.loads(msg)
                if data.get("type") == "status":
                    if data.get("state") == target_state:
                        return target_state
        except asyncio.TimeoutError:
            continue
    raise TimeoutError(f"Did not see state '{target_state}' within {timeout}s")
```

---

## 4. Audio Quality / Input Conditions

**Status**: ⬜ Future - requires audio fixture files

Testing robustness to real-world audio conditions.

| ID | Test Case | Description | Expected Behavior | Priority |
|----|-----------|-------------|-------------------|----------|
| AQ-01 | Background noise | Speech with ambient noise | STT extracts speech reasonably | P1 |
| AQ-02 | Background music | Speech with music playing | STT focuses on speech | P2 |
| AQ-03 | Low volume | Quiet speaker | STT attempts transcription | P1 |
| AQ-04 | High volume / clipping | Audio with clipping distortion | STT handles gracefully | P2 |
| AQ-05 | Echo / reverb | Audio with room reverb | STT handles reasonably | P2 |

### Implementation Notes

These tests require pre-recorded audio fixtures with specific characteristics:

```
tests/fixtures/voice/audio_quality/
├── speech_with_cafe_noise.pcm
├── speech_with_music.pcm
├── speech_low_volume.pcm
├── speech_clipped.pcm
└── speech_reverb.pcm
```

Test pattern:
```python
@pytest.fixture
def audio_fixture_path():
    return Path(__file__).parent / "fixtures" / "voice" / "audio_quality"

@pytest.mark.asyncio
async def test_speech_with_background_noise(self, voice_test_project, audio_fixture_path):
    """AQ-01: Speech should be transcribed despite background noise."""
    audio = (audio_fixture_path / "speech_with_cafe_noise.pcm").read_bytes()

    result = await run_voice_session(audio_chunks=[audio], ...)

    # Should get some transcription (may not be perfect)
    assert len(result.full_transcription) > 0
    # Should not error
    assert not result.has_error
```

---

## 5. Turn-Taking and Timing

**Status**: ⬜ Partial - latency test implemented

| ID | Test Case | Description | Expected Behavior | Priority | Status |
|----|-----------|-------------|-------------------|----------|--------|
| TT-01 | End-of-utterance detection | User finishes speaking | VAD detects end promptly | P0 | ⬜ |
| TT-02 | Long pause mid-sentence | User pauses 2-3s then continues | Wait for actual end | P1 | ⬜ |
| TT-06 | Latency measurement | Time from end-of-speech to first TTS | Target: <1000ms | P0 | ✅ |
| TT-07 | Phrase streaming | LLM streams phrase by phrase | Phrases sent to TTS as detected | P0 | ⬜ |

---

## 6. Native Audio Mode (Omni Models)

**Status**: ✅ Implemented in `test_voice_e2e.py::TestE2EGP06OmniModel`

Testing direct audio input path for multimodal models like Qwen2.5-Omni.

| ID | Test Case | Description | Expected Behavior | Priority | Status |
|----|-----------|-------------|-------------------|----------|--------|
| NA-01 | Basic native audio | Send audio directly to Omni model | Model understands and responds | P0 | ✅ E2E |
| NA-02 | Factual question | Omni answers math question from audio | Correct answer produced | P0 | ✅ E2E |
| NA-03 | TTS synthesis | Omni response synthesized to speech | TTS audio returned | P0 | ✅ E2E |
| NA-04 | State transitions | Verify IDLE→LISTENING→PROCESSING→SPEAKING→IDLE | Correct state flow | P1 | ✅ E2E |
| NA-05 | Audio encoding | PCM → WAV conversion | Correct WAV header | P1 | ⬜ TODO |

### Omni Test Configuration

Native audio is **automatically detected** by the voice router when the model reports
`native_audio: true` capability. Tests use a separate project config:

```python
# Separate project config for Omni model (in test_voice_e2e.py)
OMNI_PROJECT_CONFIG = {
    "version": "v1",
    "name": "voice-e2e-omni",
    "namespace": "test",
    "runtime": {
        "default_model": "omni",
        "models": [{
            "name": "omni",
            "provider": "universal",
            "model": "ggml-org/qwen2.5-omni-3b:q5_k_m",
        }],
    },
    "voice": {
        "enabled": True,
        "llm_model": "omni",  # Omni model auto-detected for native audio
        "tts": {"model": "kokoro", "voice": "af_heart", "speed": 0.95},
        "enable_thinking": False,
    },
}
```

### Omni Test Pattern

```python
@pytest.mark.asyncio
async def test_omni_question_produces_answer(self, omni_test_project):
    """NA-02: Omni model answers factual question from audio."""
    audio = generate_real_speech("What is two plus two?")

    result = await run_voice_session(
        audio_chunks=[audio],
        system_prompt="Answer math questions with just the number.",
        namespace=omni_test_project["namespace"],
        project=omni_test_project["project"],
        timeout_seconds=60.0,
        silence_after_audio=2.0,
    )

    # Debug output
    print(f"States: {result.status_states}")
    print(f"Transcription (native audio): '{result.full_transcription}'")
    print(f"LLM Response: '{result.full_llm_response}'")

    assert not result.has_error, f"Errors: {result.errors}"
    assert len(result.llm_texts) > 0  # Native audio skips transcription
    assert "4" in result.full_llm_response.lower() or "four" in result.full_llm_response.lower()
```

### Key Differences from Standard Flow

| Aspect | Standard (STT → LLM) | Omni (Native Audio) |
|--------|---------------------|---------------------|
| Transcription | ✅ Whisper produces text | ❌ Skipped |
| Audio processing | PCM → STT → text → LLM | PCM → WAV → LLM |
| Transcription message | Sent to client | Not sent |
| LLM prompt | Text-based | Audio-based (multimodal) |

---

## Test Environment Requirements

### Running E2E Tests

Prerequisites:
1. LlamaFarm Server running: `nx start server`
2. Universal Runtime running: `nx start universal-runtime`
3. Models loaded:
   - STT: Whisper base (auto-loaded)
   - TTS: Kokoro (auto-loaded)
   - LLM: `unsloth/qwen3-1.7b-gguf:q5_k_m`

Run tests:
```bash
# All E2E tests
uv run pytest server/tests/test_voice_e2e.py -v -s

# Specific test class
uv run pytest server/tests/test_voice_e2e.py::TestE2EGP01SimpleGreeting -v -s

# Exclude E2E tests (unit tests only)
uv run pytest server/tests/ -v -m "not e2e"
```

### Audio Fixtures (Future)

For reproducible CI testing, create pre-recorded fixtures:

```
server/tests/fixtures/voice/
├── golden_path/
│   ├── hello.pcm                    # "Hello"
│   ├── capital_of_france.pcm        # "What is the capital of France?"
│   └── two_plus_two.pcm             # "What is two plus two?"
├── edge_cases/
│   ├── silence_5s.pcm               # Pure silence
│   ├── very_short_50ms.pcm          # Minimal speech
│   └── very_long_90s.pcm            # Extended speech
├── audio_quality/
│   ├── speech_with_noise.pcm
│   └── speech_low_volume.pcm
└── interrupts/
    ├── stop.pcm                     # "Stop!"
    └── wait.pcm                     # "Wait, actually..."
```

---

## Metrics to Track

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Time to first TTS audio | <1000ms | `TestE2ELatency.test_time_to_first_audio` |
| STT latency | <500ms for 3s audio | Instrumentation |
| LLM time to first token | <300ms | Instrumentation |
| Interrupt response time | <200ms | INT-06 test |
| Session memory growth | <10MB/hour | Memory profiling |
| Error rate | <1% of turns | Error logging |

---

## Priority Legend

- **P0**: Must pass before any release
- **P1**: Should pass for production readiness
- **P2**: Nice to have, improves robustness
- **P3**: Future consideration

---

## Implementation Status Summary

| Category | Tests | Implemented | Remaining |
|----------|-------|-------------|-----------|
| Golden Path (GP) | 7 | 6 | 1 |
| Negative Path (NP) | 11 | 8 | 3 |
| Interruptions (INT) | 7 | 6 | 1 |
| Audio Quality (AQ) | 5 | 0 | 5 |
| Turn-Taking (TT) | 8 | 1 | 7 |
| Native Audio (NA) | 5 | 4 | 1 |
| **Total** | **43** | **25** | **18** |
