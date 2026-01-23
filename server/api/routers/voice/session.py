"""
Voice session state management.

Manages stateful conversation sessions for the voice assistant pipeline.
"""

import asyncio
import logging
import subprocess
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .turn_detector import EndOfTurnDetector, TurnDetectorConfig
from .types import VoiceSessionConfig, VoiceState
from .vad import VoiceActivityDetector

logger = logging.getLogger(__name__)


# Security: Maximum buffer size to prevent memory exhaustion DoS
MAX_ENCODED_BUFFER_SIZE = 10 * 1024 * 1024  # 10MB max for streaming decode

# Allowed input formats for FFmpeg (whitelist for security)
ALLOWED_FFMPEG_FORMATS = frozenset({"webm", "ogg", "mp3", "flac", "aiff", "wav", "m4a", "mp4", "opus"})


class AudioFormat(str, Enum):
    """Detected audio format."""

    PCM = "pcm"  # Raw PCM 16-bit
    WEBM = "webm"  # WebM container (likely Opus)
    OGG = "ogg"  # Ogg container (likely Opus)
    UNKNOWN = "unknown"


def detect_audio_format(data: bytes) -> AudioFormat:
    """Detect audio format from initial bytes.

    Args:
        data: First chunk of audio data.

    Returns:
        Detected AudioFormat.

    Note:
        Returns UNKNOWN for unsupported formats (MP3, MP4, FLAC, etc.)
        which should be explicitly rejected by callers rather than
        defaulting to PCM processing.
    """
    if len(data) < 4:
        return AudioFormat.UNKNOWN

    # WebM starts with EBML header: 0x1A 0x45 0xDF 0xA3
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return AudioFormat.WEBM

    # Ogg starts with "OggS"
    if data[:4] == b"OggS":
        return AudioFormat.OGG

    # Explicitly detect and reject unsupported formats
    # MP3: ID3 tag or frame sync
    if data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
        logger.warning("MP3 format detected but not supported for streaming input")
        return AudioFormat.UNKNOWN

    # MP4/M4A: ftyp box
    if len(data) >= 8 and data[4:8] == b"ftyp":
        logger.warning("MP4/M4A format detected but not supported for streaming input")
        return AudioFormat.UNKNOWN

    # FLAC: fLaC magic
    if data[:4] == b"fLaC":
        logger.warning("FLAC format detected but not supported for streaming input")
        return AudioFormat.UNKNOWN

    # AIFF: FORM header
    if data[:4] == b"FORM":
        logger.warning("AIFF format detected but not supported for streaming input")
        return AudioFormat.UNKNOWN

    # WAV: RIFF header - actually supported, treat as PCM after stripping header
    if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WAVE":
        # WAV files contain PCM data, we can treat them as PCM
        return AudioFormat.PCM

    # If first bytes look like typical PCM values (not a magic number),
    # assume raw PCM. PCM doesn't have a header, so we heuristically check
    # that it doesn't match known container formats.
    return AudioFormat.PCM


class StreamingAudioDecoder:
    """Decodes WebM/Opus or other formats to PCM in real-time using ffmpeg.

    Uses ffmpeg subprocess with non-blocking I/O for streaming decode.
    Accumulates encoded audio and decodes periodically when enough data
    is available.

    Security features:
    - Input format whitelist to prevent command injection
    - Maximum buffer size to prevent memory exhaustion DoS
    - Async subprocess execution to avoid blocking the event loop
    """

    # Minimum bytes before attempting decode (need WebM header + some data)
    MIN_DECODE_BYTES = 2048

    # Decode interval in bytes (decode every N bytes of input)
    # WebM/Opus at 16kbps = 2KB/s, so 4KB = ~2 seconds
    DECODE_INTERVAL = 4096

    def __init__(self, input_format: str = "webm"):
        """Initialize decoder.

        Args:
            input_format: Input format hint for ffmpeg (webm, ogg, etc.)
                         Must be in ALLOWED_FFMPEG_FORMATS whitelist.

        Raises:
            ValueError: If input_format is not in the allowed formats whitelist.
        """
        # Validate format against whitelist to prevent command injection
        if input_format not in ALLOWED_FFMPEG_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {input_format}. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_FFMPEG_FORMATS))}"
            )
        self._input_format = input_format
        self._encoded_buffer = bytearray()
        self._last_decode_size = 0
        self._total_pcm_decoded = 0

    def _decode_buffer(self) -> bytes:
        """Decode the accumulated buffer to PCM (synchronous).

        Note: This method blocks while ffmpeg runs. For async contexts,
        use _decode_buffer_async() instead to avoid blocking the event loop.

        Returns:
            Decoded PCM data.
        """
        if len(self._encoded_buffer) < self.MIN_DECODE_BYTES:
            return b""

        # Run ffmpeg to decode the buffer
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-f", self._input_format,
            "-i", "pipe:0",
            "-ar", "16000",
            "-ac", "1",
            "-f", "s16le",
            "pipe:1",
        ]

        try:
            result = subprocess.run(
                cmd,
                input=bytes(self._encoded_buffer),
                capture_output=True,
                timeout=5.0,
            )

            if result.stderr:
                # Only log actual errors, not warnings about truncated streams
                stderr_str = result.stderr.decode(errors="ignore")
                if "error" in stderr_str.lower():
                    logger.warning(f"ffmpeg decode: {stderr_str[:200]}")

            return result.stdout

        except subprocess.TimeoutExpired:
            logger.warning("ffmpeg decode timeout")
            return b""
        except FileNotFoundError:
            logger.error("ffmpeg not found")
            return b""
        except Exception as e:
            logger.error(f"ffmpeg decode error: {e}")
            return b""

    async def _decode_buffer_async(self) -> bytes:
        """Decode the accumulated buffer to PCM (async, non-blocking).

        Uses asyncio.create_subprocess_exec to avoid blocking the event loop.

        Returns:
            Decoded PCM data.
        """
        if len(self._encoded_buffer) < self.MIN_DECODE_BYTES:
            return b""

        try:
            proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-f", self._input_format,
                "-i", "pipe:0",
                "-ar", "16000",
                "-ac", "1",
                "-f", "s16le",
                "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=bytes(self._encoded_buffer)),
                    timeout=5.0,
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                logger.warning("ffmpeg async decode timeout")
                return b""

            if stderr:
                stderr_str = stderr.decode(errors="ignore")
                if "error" in stderr_str.lower():
                    logger.warning(f"ffmpeg async decode: {stderr_str[:200]}")

            return stdout

        except FileNotFoundError:
            logger.error("ffmpeg not found")
            return b""
        except Exception as e:
            logger.error(f"ffmpeg async decode error: {e}")
            return b""

    def feed(self, data: bytes) -> bytes:
        """Feed encoded audio data and return newly decoded PCM.

        Decodes periodically when enough new data has accumulated.
        Returns only the NEW PCM (not previously returned).

        Args:
            data: Encoded audio chunk (WebM/Opus, etc.)

        Returns:
            Newly decoded PCM data (may be empty).

        Note:
            Buffer is trimmed when it exceeds MAX_ENCODED_BUFFER_SIZE to prevent
            memory exhaustion. This may cause audio discontinuities for very long
            streams, but prevents DoS attacks.
        """
        self._encoded_buffer.extend(data)

        # Security: Trim buffer if it exceeds max size to prevent memory exhaustion
        if len(self._encoded_buffer) > MAX_ENCODED_BUFFER_SIZE:
            # Keep only the most recent portion (needed for continuous decoding)
            trim_amount = len(self._encoded_buffer) - MAX_ENCODED_BUFFER_SIZE
            logger.warning(
                f"Encoded buffer exceeded max size ({MAX_ENCODED_BUFFER_SIZE} bytes), "
                f"trimming {trim_amount} bytes to prevent memory exhaustion"
            )
            # Decode what we have first
            all_pcm = self._decode_buffer()
            if len(all_pcm) > self._total_pcm_decoded:
                new_pcm = all_pcm[self._total_pcm_decoded:]
                self._total_pcm_decoded = len(all_pcm)
            else:
                new_pcm = b""
            # Reset buffer - unfortunately we lose continuity but prevent OOM
            self._encoded_buffer.clear()
            self._last_decode_size = 0
            self._total_pcm_decoded = 0
            return new_pcm

        # Check if we should decode
        bytes_since_last = len(self._encoded_buffer) - self._last_decode_size
        if bytes_since_last < self.DECODE_INTERVAL:
            return b""

        # Decode entire buffer
        all_pcm = self._decode_buffer()

        if len(all_pcm) > self._total_pcm_decoded:
            # Return only the new PCM
            new_pcm = all_pcm[self._total_pcm_decoded:]
            self._total_pcm_decoded = len(all_pcm)
            self._last_decode_size = len(self._encoded_buffer)
            return new_pcm

        self._last_decode_size = len(self._encoded_buffer)
        return b""

    def flush(self) -> bytes:
        """Decode any remaining data and return final PCM.

        Returns:
            Any remaining decoded PCM data.
        """
        if len(self._encoded_buffer) == 0:
            return b""

        all_pcm = self._decode_buffer()

        if len(all_pcm) > self._total_pcm_decoded:
            new_pcm = all_pcm[self._total_pcm_decoded:]
            self._total_pcm_decoded = len(all_pcm)
            return new_pcm

        return b""

    async def feed_async(self, data: bytes) -> bytes:
        """Feed encoded audio data and return newly decoded PCM (async version).

        Non-blocking version that uses asyncio subprocess to avoid freezing
        the event loop. Preferred for use in async contexts.

        Args:
            data: Encoded audio chunk (WebM/Opus, etc.)

        Returns:
            Newly decoded PCM data (may be empty).
        """
        self._encoded_buffer.extend(data)

        # Security: Trim buffer if it exceeds max size
        if len(self._encoded_buffer) > MAX_ENCODED_BUFFER_SIZE:
            trim_amount = len(self._encoded_buffer) - MAX_ENCODED_BUFFER_SIZE
            logger.warning(
                f"Encoded buffer exceeded max size, trimming {trim_amount} bytes"
            )
            all_pcm = await self._decode_buffer_async()
            if len(all_pcm) > self._total_pcm_decoded:
                new_pcm = all_pcm[self._total_pcm_decoded:]
                self._total_pcm_decoded = len(all_pcm)
            else:
                new_pcm = b""
            self._encoded_buffer.clear()
            self._last_decode_size = 0
            self._total_pcm_decoded = 0
            return new_pcm

        # Check if we should decode
        bytes_since_last = len(self._encoded_buffer) - self._last_decode_size
        if bytes_since_last < self.DECODE_INTERVAL:
            return b""

        # Decode entire buffer
        all_pcm = await self._decode_buffer_async()

        if len(all_pcm) > self._total_pcm_decoded:
            new_pcm = all_pcm[self._total_pcm_decoded:]
            self._total_pcm_decoded = len(all_pcm)
            self._last_decode_size = len(self._encoded_buffer)
            return new_pcm

        self._last_decode_size = len(self._encoded_buffer)
        return b""

    async def flush_async(self) -> bytes:
        """Decode any remaining data and return final PCM (async version).

        Returns:
            Any remaining decoded PCM data.
        """
        if len(self._encoded_buffer) == 0:
            return b""

        all_pcm = await self._decode_buffer_async()

        if len(all_pcm) > self._total_pcm_decoded:
            new_pcm = all_pcm[self._total_pcm_decoded:]
            self._total_pcm_decoded = len(all_pcm)
            return new_pcm

        return b""

    def reset(self) -> None:
        """Reset decoder state for new audio stream."""
        self._encoded_buffer.clear()
        self._last_decode_size = 0
        self._total_pcm_decoded = 0

    def close(self) -> None:
        """Clean up decoder resources."""
        self.reset()


@dataclass
class VoiceSession:
    """Stateful voice chat session.

    Holds conversation history, pipeline state, and configuration
    for a single voice assistant session.
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    config: VoiceSessionConfig = field(default_factory=VoiceSessionConfig)

    # Conversation history (OpenAI message format)
    messages: list[dict[str, Any]] = field(default_factory=list)

    # Pipeline state
    state: VoiceState = VoiceState.IDLE

    # Interrupt event for barge-in
    _interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Audio buffer for accumulating chunks
    _audio_buffer: bytearray = field(default_factory=bytearray)

    # Voice activity detector for automatic end-of-speech detection
    _vad: VoiceActivityDetector = field(default_factory=VoiceActivityDetector)

    # Detected audio format (detected after accumulating enough bytes)
    _audio_format: AudioFormat | None = None

    # Audio decoder for WebM/Opus (created when needed)
    _decoder: StreamingAudioDecoder | None = None

    # Buffer for format detection (accumulates until we have enough to detect)
    _format_detect_buffer: bytearray = field(default_factory=bytearray)

    # Current phrase index for TTS coordination
    _phrase_index: int = 0

    # Barge-in noise filter: consecutive speech chunk counter
    _barge_in_speech_chunks: int = 0

    # End-of-turn detector for linguistic analysis
    _turn_detector: EndOfTurnDetector | None = None

    # Partial transcription for turn detection (updated during silence window)
    _partial_transcript: str = ""

    def __post_init__(self):
        """Initialize session with system prompt if provided."""
        if self.config.system_prompt and not self.messages:
            self.messages.append({
                "role": "system",
                "content": self.config.system_prompt,
            })

        # Initialize turn detector with config
        self._turn_detector = EndOfTurnDetector(
            config=TurnDetectorConfig(
                base_silence_duration=self.config.base_silence_duration,
                thinking_silence_duration=self.config.thinking_silence_duration,
                max_silence_duration=self.config.max_silence_duration,
                enable_linguistic_analysis=self.config.turn_detection_enabled,
            )
        )

    def add_user_message(self, text: str) -> None:
        """Add a user message to conversation history."""
        self.messages.append({"role": "user", "content": text})

    def add_assistant_message(self, text: str) -> None:
        """Add an assistant message to conversation history."""
        self.messages.append({"role": "assistant", "content": text})

    def set_state(self, new_state: VoiceState) -> VoiceState:
        """Update session state and return the previous state."""
        old_state = self.state
        self.state = new_state
        return old_state

    def request_interrupt(self) -> None:
        """Signal barge-in interrupt."""
        self._interrupt_event.set()

    def clear_interrupt(self) -> None:
        """Clear interrupt flag."""
        self._interrupt_event.clear()

    def is_interrupted(self) -> bool:
        """Check if interrupt has been requested."""
        return self._interrupt_event.is_set()

    async def wait_for_interrupt(self, timeout: float | None = None) -> bool:
        """Wait for interrupt signal.

        Returns True if interrupted, False if timeout.
        """
        try:
            await asyncio.wait_for(self._interrupt_event.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    def append_audio(self, chunk: bytes) -> bool:
        """Append audio chunk to buffer and check for end-of-speech.

        Supports both raw PCM and encoded audio (WebM/Opus).
        For encoded audio, decodes to PCM before VAD analysis.

        Args:
            chunk: Audio chunk (PCM 16kHz 16-bit mono, or WebM/Opus).

        Returns:
            True if VAD detected end-of-speech and processing should begin.
        """
        # Always add raw chunk to buffer (for transcription)
        self._audio_buffer.extend(chunk)

        # Accumulate bytes for format detection if not yet detected
        if self._audio_format is None:
            self._format_detect_buffer.extend(chunk)

            # Need at least 4 bytes to detect format reliably
            if len(self._format_detect_buffer) < 4:
                return False

            # Now we have enough bytes to detect format
            self._audio_format = detect_audio_format(bytes(self._format_detect_buffer))
            logger.info(f"Detected audio format: {self._audio_format.value} (initial detection)")

            # Create decoder for encoded formats
            if self._audio_format == AudioFormat.WEBM:
                self._decoder = StreamingAudioDecoder(input_format="webm")
                # Feed accumulated bytes to decoder
                pcm_chunk = self._decoder.feed(bytes(self._format_detect_buffer))
                if pcm_chunk:
                    return self._vad.process_chunk(pcm_chunk)
                return False
            elif self._audio_format == AudioFormat.OGG:
                self._decoder = StreamingAudioDecoder(input_format="ogg")
                pcm_chunk = self._decoder.feed(bytes(self._format_detect_buffer))
                if pcm_chunk:
                    return self._vad.process_chunk(pcm_chunk)
                return False
            else:
                # PCM - process accumulated bytes through VAD
                return self._vad.process_chunk(bytes(self._format_detect_buffer))

        # Get PCM for VAD
        if self._decoder is not None:
            # Decode to PCM for VAD
            pcm_chunk = self._decoder.feed(chunk)
            if pcm_chunk:
                return self._vad.process_chunk(pcm_chunk)
            return False
        else:
            # Already PCM, pass directly to VAD
            return self._vad.process_chunk(chunk)

    def get_audio_buffer(self) -> bytes:
        """Get accumulated audio and clear buffer.

        Note: Audio format and decoder are preserved across utterances
        within the same WebSocket session. This is important for encoded
        formats like WebM where continuation chunks don't have headers.

        For continuous WebM streams, the decoder keeps accumulating data
        because subsequent chunks need the header context to decode.
        """
        audio = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        self._vad.reset()

        # Reset turn detector and partial transcript for new utterance
        if self._turn_detector is not None:
            self._turn_detector.reset()
        self._partial_transcript = ""

        # DO NOT reset decoder for continuous streams (WebM/Opus)
        # The decoder needs accumulated data including headers to decode
        # new chunks. It tracks _total_pcm_decoded internally to return
        # only newly decoded PCM.

        # Clear format detect buffer but keep _audio_format and _decoder
        # so we continue decoding the same format stream
        self._format_detect_buffer.clear()

        return audio

    def has_audio(self) -> bool:
        """Check if audio buffer has data."""
        return len(self._audio_buffer) > 0

    def discard_audio(self) -> None:
        """Discard accumulated audio without returning it.

        Used after barge-in to clear any buffered audio that arrived
        during TTS playback. This prevents stale/echo audio from being
        processed in the next utterance.
        """
        self._audio_buffer.clear()
        self._vad.reset()
        self._format_detect_buffer.clear()
        if self._turn_detector is not None:
            self._turn_detector.reset()
        self._partial_transcript = ""

    def flush_and_check_vad(self) -> bool:
        """Flush decoder and do final VAD check.

        For encoded audio, this decodes any remaining buffered data
        and checks if speech has ended.

        Returns:
            True if VAD detected end-of-speech.
        """
        if self._decoder is not None:
            # Flush decoder to get remaining PCM
            remaining_pcm = self._decoder.flush()
            if remaining_pcm:
                logger.debug(f"Flushed {len(remaining_pcm)} bytes of PCM from decoder")
                # Add flushed PCM to buffer so it's included in transcription
                self._audio_buffer.extend(remaining_pcm)
                return self._vad.process_chunk(remaining_pcm)
        return False

    def is_speech_active(self) -> bool:
        """Check if VAD detected active speech."""
        return self._vad.is_speech_active()

    def is_in_silence_window(self) -> bool:
        """Check if we're in the silence window after speech."""
        return self._vad.is_in_silence_window()

    def get_speech_duration(self) -> float:
        """Get duration of current speech in seconds."""
        return self._vad.get_speech_duration()

    def get_silence_duration(self) -> float:
        """Get duration of current silence in seconds."""
        return self._vad.get_silence_duration()

    def set_partial_transcript(self, text: str) -> None:
        """Set the partial transcription for turn detection analysis.

        Call this during the silence window with streaming STT results
        to enable linguistic analysis for end-of-turn detection.
        """
        self._partial_transcript = text

    def get_partial_transcript(self) -> str:
        """Get the current partial transcription."""
        return self._partial_transcript

    def check_end_of_turn_with_analysis(self) -> bool:
        """Check if end-of-turn should trigger using linguistic analysis.

        Uses the partial transcription to determine appropriate silence
        threshold, then checks if that threshold has been exceeded.

        Returns:
            True if turn should end and processing should begin.
        """
        if self._turn_detector is None:
            # Fall back to default VAD behavior
            return self._vad.check_end_of_turn(self._vad.config.silence_duration)

        speech_duration = self._vad.get_speech_duration()
        silence_duration = self._vad.get_silence_duration()

        return self._turn_detector.should_end_turn(
            silence_duration=silence_duration,
            speech_duration=speech_duration,
            partial_transcript=self._partial_transcript,
        )

    def get_required_silence_duration(self) -> float:
        """Get the required silence duration based on current context.

        Returns the dynamically calculated silence threshold based on
        speech duration and linguistic analysis of partial transcription.
        """
        if self._turn_detector is None:
            return self._vad.config.silence_duration

        return self._turn_detector.get_required_silence(
            partial_transcript=self._partial_transcript,
            speech_duration=self._vad.get_speech_duration(),
        )

    def detect_barge_in(self, chunk: bytes) -> bool:
        """Check if audio chunk contains speech (for barge-in detection).

        Used during SPEAKING state to detect if the user has started talking.
        Assumes client handles echo cancellation, so any detected speech
        is genuine user input.

        Note: Uses a temporary decoder to avoid corrupting the main decoder's
        state, which would cause issues when processing the actual utterance.

        Args:
            chunk: Audio chunk (PCM or encoded WebM/Opus).

        Returns:
            True if speech detected (should trigger interrupt).
        """
        # Check if barge-in is enabled
        if not self.config.barge_in_enabled:
            logger.info("Barge-in check: DISABLED")
            return False

        # Get PCM for energy analysis
        pcm_chunk: bytes
        if self._audio_format == AudioFormat.WEBM:
            # Use temporary decoder to avoid corrupting main decoder state
            temp_decoder = StreamingAudioDecoder(input_format="webm")
            pcm_chunk = temp_decoder.feed(chunk)
            temp_decoder.close()
            if not pcm_chunk:
                logger.info(f"Barge-in check: decoder returned empty (chunk={len(chunk)} bytes)")
                return False
        elif self._audio_format == AudioFormat.OGG:
            # Use temporary decoder to avoid corrupting main decoder state
            temp_decoder = StreamingAudioDecoder(input_format="ogg")
            pcm_chunk = temp_decoder.feed(chunk)
            temp_decoder.close()
            if not pcm_chunk:
                logger.info(f"Barge-in check: decoder returned empty (chunk={len(chunk)} bytes)")
                return False
        elif self._audio_format == AudioFormat.PCM or self._audio_format is None:
            # Raw PCM or format not yet detected (assume PCM)
            pcm_chunk = chunk
        else:
            # Unknown encoded format without decoder - can't analyze
            logger.info(f"Barge-in check: unknown format {self._audio_format}, no decoder")
            return False

        # Calculate energy and check against speech threshold
        energy = self._vad._calculate_energy(pcm_chunk)
        threshold = self._vad.config.speech_threshold
        is_speech = energy > threshold

        # Log every check for debugging
        logger.info(
            f"Barge-in check: energy={energy:.6f}, threshold={threshold:.6f}, "
            f"is_speech={is_speech}, chunk_size={len(pcm_chunk)}"
        )

        # Apply noise filter if enabled
        if self.config.barge_in_noise_filter:
            if is_speech:
                self._barge_in_speech_chunks += 1
                if self._barge_in_speech_chunks >= self.config.barge_in_min_chunks:
                    logger.info(
                        f"Barge-in TRIGGERED: {self._barge_in_speech_chunks} consecutive "
                        f"chunks above threshold (energy={energy:.4f})"
                    )
                    return True
                else:
                    logger.info(
                        f"Barge-in pending: {self._barge_in_speech_chunks}/{self.config.barge_in_min_chunks} "
                        f"chunks (energy={energy:.4f})"
                    )
                    return False
            else:
                # Reset counter on silence
                if self._barge_in_speech_chunks > 0:
                    logger.info(f"Barge-in reset: silence detected after {self._barge_in_speech_chunks} chunks")
                self._barge_in_speech_chunks = 0
                return False
        else:
            # No noise filter - trigger immediately on speech
            if is_speech:
                logger.info(f"Barge-in detected: energy={energy:.4f} > threshold={threshold}")
            return is_speech

    def reset_barge_in_state(self) -> None:
        """Reset barge-in detection state (call when transitioning out of SPEAKING)."""
        self._barge_in_speech_chunks = 0

    def next_phrase_index(self) -> int:
        """Get next phrase index and increment counter."""
        idx = self._phrase_index
        self._phrase_index += 1
        return idx

    def reset_phrase_counter(self) -> None:
        """Reset phrase counter for new response."""
        self._phrase_index = 0

    def update_config(
        self,
        stt_model: str | None = None,
        tts_model: str | None = None,
        tts_voice: str | None = None,
        llm_model: str | None = None,
        language: str | None = None,
        speed: float | None = None,
        sentence_boundary_only: bool | None = None,
        barge_in_enabled: bool | None = None,
        barge_in_noise_filter: bool | None = None,
        barge_in_min_chunks: int | None = None,
        turn_detection_enabled: bool | None = None,
        base_silence_duration: float | None = None,
        thinking_silence_duration: float | None = None,
        max_silence_duration: float | None = None,
    ) -> None:
        """Update session configuration."""
        if stt_model is not None:
            self.config.stt_model = stt_model
        if tts_model is not None:
            self.config.tts_model = tts_model
        if tts_voice is not None:
            self.config.tts_voice = tts_voice
        if llm_model is not None:
            self.config.llm_model = llm_model
        if language is not None:
            self.config.language = language
        if speed is not None:
            self.config.speed = speed
        if sentence_boundary_only is not None:
            self.config.sentence_boundary_only = sentence_boundary_only
        if barge_in_enabled is not None:
            self.config.barge_in_enabled = barge_in_enabled
        if barge_in_noise_filter is not None:
            self.config.barge_in_noise_filter = barge_in_noise_filter
        if barge_in_min_chunks is not None:
            self.config.barge_in_min_chunks = barge_in_min_chunks

        # Update turn detection config
        turn_detector_updated = False
        if turn_detection_enabled is not None:
            self.config.turn_detection_enabled = turn_detection_enabled
            turn_detector_updated = True
        if base_silence_duration is not None:
            self.config.base_silence_duration = base_silence_duration
            turn_detector_updated = True
        if thinking_silence_duration is not None:
            self.config.thinking_silence_duration = thinking_silence_duration
            turn_detector_updated = True
        if max_silence_duration is not None:
            self.config.max_silence_duration = max_silence_duration
            turn_detector_updated = True

        # Recreate turn detector if config changed
        if turn_detector_updated and self._turn_detector is not None:
            self._turn_detector = EndOfTurnDetector(
                config=TurnDetectorConfig(
                    base_silence_duration=self.config.base_silence_duration,
                    thinking_silence_duration=self.config.thinking_silence_duration,
                    max_silence_duration=self.config.max_silence_duration,
                    enable_linguistic_analysis=self.config.turn_detection_enabled,
                )
            )


class SessionManager:
    """Manages voice chat sessions.

    Thread-safe session storage with creation, retrieval, and cleanup.
    """

    def __init__(self, max_sessions: int = 100):
        self._sessions: dict[str, VoiceSession] = {}
        self._max_sessions = max_sessions
        self._lock = asyncio.Lock()

    async def create_session(
        self, config: VoiceSessionConfig | None = None
    ) -> VoiceSession:
        """Create a new voice session."""
        async with self._lock:
            # Enforce session limit
            if len(self._sessions) >= self._max_sessions:
                # Remove oldest session
                oldest_id = next(iter(self._sessions))
                del self._sessions[oldest_id]

            session = VoiceSession(config=config or VoiceSessionConfig())
            self._sessions[session.session_id] = session
            return session

    async def get_session(self, session_id: str) -> VoiceSession | None:
        """Get an existing session by ID."""
        return self._sessions.get(session_id)

    async def get_or_create_session(
        self, session_id: str | None, config: VoiceSessionConfig | None = None
    ) -> VoiceSession:
        """Get existing session or create new one.

        Uses lock to ensure atomicity of the check-and-create operation.
        """
        async with self._lock:
            if session_id:
                session = self._sessions.get(session_id)
                if session:
                    # Update config if provided
                    if config:
                        session.update_config(
                            stt_model=config.stt_model,
                            tts_model=config.tts_model,
                            tts_voice=config.tts_voice,
                            llm_model=config.llm_model,
                            language=config.language,
                            speed=config.speed,
                            sentence_boundary_only=config.sentence_boundary_only,
                            barge_in_enabled=config.barge_in_enabled,
                            barge_in_noise_filter=config.barge_in_noise_filter,
                            barge_in_min_chunks=config.barge_in_min_chunks,
                            turn_detection_enabled=config.turn_detection_enabled,
                            base_silence_duration=config.base_silence_duration,
                            thinking_silence_duration=config.thinking_silence_duration,
                            max_silence_duration=config.max_silence_duration,
                        )
                    return session

            # Create new session (inline to avoid releasing lock)
            if len(self._sessions) >= self._max_sessions:
                # Remove oldest session
                oldest_id = next(iter(self._sessions))
                del self._sessions[oldest_id]

            session = VoiceSession(config=config or VoiceSessionConfig())
            self._sessions[session.session_id] = session
            return session

    async def remove_session(self, session_id: str) -> bool:
        """Remove a session. Returns True if session existed."""
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    async def cleanup_idle_sessions(self, max_idle_seconds: float = 3600) -> int:
        """Remove sessions that have been idle too long.

        Returns number of sessions removed.
        """
        # For now, we don't track last activity time
        # This is a placeholder for future enhancement
        return 0

    @property
    def session_count(self) -> int:
        """Get current number of active sessions."""
        return len(self._sessions)


# Global session manager instance
_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager
