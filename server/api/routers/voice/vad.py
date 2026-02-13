"""
Voice Activity Detection (VAD) for automatic end-of-speech detection.

Uses energy-based silence detection to determine when a user has
stopped speaking, eliminating the need for explicit "end" signals.

Timing is based on audio sample count (not wall-clock time) so it works
correctly even when audio is sent faster than real-time.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class VADState(str, Enum):
    """Voice activity state."""

    IDLE = "idle"  # No speech detected yet
    SPEAKING = "speaking"  # Active speech detected
    SILENCE = "silence"  # Silence after speech, waiting for threshold


@dataclass
class VADConfig:
    """Configuration for voice activity detection."""

    # Energy threshold for speech detection (0.0-1.0)
    # Lower = more sensitive, higher = less sensitive
    speech_threshold: float = 0.015

    # How long silence must persist after speech to trigger end (seconds)
    # Lower = faster response, higher = fewer false positives
    silence_duration: float = 0.4

    # Minimum speech duration before we consider it valid (seconds)
    # Prevents triggering on brief noises
    min_speech_duration: float = 0.25

    # Sample rate of incoming audio (Hz)
    sample_rate: int = 16000

    # Expected audio format: 16-bit signed integer (2 bytes per sample)
    sample_width: int = 2


@dataclass
class VoiceActivityDetector:
    """Detects voice activity in streaming audio.

    Tracks when speech starts and ends using energy-based detection.
    After speech is detected and then followed by sustained silence,
    signals that the utterance is complete.

    Duration tracking is based on audio sample count, not wall-clock time,
    so VAD works correctly even when audio arrives faster than real-time.
    """

    config: VADConfig = field(default_factory=VADConfig)

    # Current state
    state: VADState = VADState.IDLE

    # Sample counts for duration tracking (based on audio data, not wall-clock)
    _speech_samples: int = 0  # Samples during speech
    _silence_samples: int = 0  # Samples during silence after speech

    # Running average for adaptive thresholding (optional)
    _energy_history: list[float] = field(default_factory=list)
    _max_history: int = 50

    def __post_init__(self):
        self._energy_history = []

    def reset(self) -> None:
        """Reset VAD state for a new utterance."""
        self.state = VADState.IDLE
        self._speech_samples = 0
        self._silence_samples = 0
        self._energy_history.clear()

    def _calculate_energy(self, audio_chunk: bytes) -> float:
        """Calculate RMS energy of audio chunk.

        Args:
            audio_chunk: Raw PCM 16kHz 16-bit mono audio.

        Returns:
            RMS energy normalized to 0.0-1.0 range.
        """
        if len(audio_chunk) < 2:
            return 0.0

        # Ensure chunk size is a multiple of 2 bytes (int16 = 2 bytes per sample)
        # Truncate any trailing byte that can't form a complete sample
        usable_bytes = len(audio_chunk) - (len(audio_chunk) % 2)
        if usable_bytes < 2:
            return 0.0

        # Convert bytes to numpy array
        samples = np.frombuffer(audio_chunk[:usable_bytes], dtype=np.int16).astype(np.float32)

        # Normalize to -1.0 to 1.0
        samples = samples / 32768.0

        # Calculate RMS energy
        rms = np.sqrt(np.mean(samples**2))

        return float(rms)

    def _get_num_samples(self, audio_chunk: bytes) -> int:
        """Get number of audio samples in a chunk."""
        return len(audio_chunk) // self.config.sample_width

    def _samples_to_seconds(self, samples: int) -> float:
        """Convert sample count to seconds."""
        return samples / self.config.sample_rate

    def process_chunk(self, audio_chunk: bytes) -> bool:
        """Process an audio chunk and update VAD state.

        Args:
            audio_chunk: Raw PCM 16kHz 16-bit mono audio.

        Returns:
            True if speech has ended and processing should begin,
            False otherwise.
        """
        energy = self._calculate_energy(audio_chunk)
        num_samples = self._get_num_samples(audio_chunk)

        # Track energy history for debugging
        self._energy_history.append(energy)
        if len(self._energy_history) > self._max_history:
            self._energy_history.pop(0)

        is_speech = energy > self.config.speech_threshold

        # Log when we see very low energy (potential silence)
        if energy < 0.001:
            logger.debug(f"VAD: Low energy chunk (energy={energy:.6f}, state={self.state.value})")

        if self.state == VADState.IDLE:
            if is_speech:
                # Speech started
                self.state = VADState.SPEAKING
                self._speech_samples = num_samples
                self._silence_samples = 0
                logger.info("VAD: State changed to \"speaking\"")
                logger.debug(f"VAD: Speech started (energy={energy:.4f}, threshold={self.config.speech_threshold})")

        elif self.state == VADState.SPEAKING:
            # Accumulate speech samples
            self._speech_samples += num_samples

            if not is_speech:
                # Speech might be ending, start silence counter
                self.state = VADState.SILENCE
                self._silence_samples = num_samples
                logger.info("VAD: State changed to \"silence\"")
                logger.debug(f"VAD: Silence detected after {self._samples_to_seconds(self._speech_samples):.2f}s speech (energy={energy:.4f})")

        elif self.state == VADState.SILENCE:
            if is_speech:
                # False alarm, speech resumed - add silence to speech duration
                self.state = VADState.SPEAKING
                self._speech_samples += self._silence_samples + num_samples
                self._silence_samples = 0
                logger.info("VAD: State changed to \"speaking\"")
                logger.debug(f"VAD: Speech resumed (energy={energy:.4f})")
            else:
                # Accumulate silence samples
                self._silence_samples += num_samples

                # Check if silence has lasted long enough
                silence_duration = self._samples_to_seconds(self._silence_samples)
                speech_duration = self._samples_to_seconds(self._speech_samples)

                if (
                    silence_duration >= self.config.silence_duration
                    and speech_duration >= self.config.min_speech_duration
                ):
                    # Speech has ended
                    logger.info("VAD: State changed to \"idle\" (end of speech)")
                    logger.debug(
                        f"VAD: Speech ended after {speech_duration:.2f}s speech, "
                        f"{silence_duration:.2f}s silence (thresholds: speech>{self.config.min_speech_duration}s, silence>{self.config.silence_duration}s)"
                    )
                    return True

        return False

    def is_speech_active(self) -> bool:
        """Check if speech is currently active."""
        return self.state in (VADState.SPEAKING, VADState.SILENCE)

    def get_speech_duration(self) -> float:
        """Get duration of current speech in seconds."""
        return self._samples_to_seconds(self._speech_samples)

    def get_silence_duration(self) -> float:
        """Get duration of current silence in seconds (after speech)."""
        return self._samples_to_seconds(self._silence_samples)

    def is_in_silence_window(self) -> bool:
        """Check if we're in the silence window after speech.

        This is the window where we might be waiting for the user to
        continue speaking (thinking pause) or for end-of-turn.
        """
        return self.state == VADState.SILENCE

    def check_end_of_turn(self, required_silence: float) -> bool:
        """Check if end-of-turn should trigger with dynamic threshold.

        Unlike process_chunk which uses the fixed config threshold,
        this method checks against a dynamically calculated threshold
        (e.g., based on linguistic analysis of partial transcription).

        Args:
            required_silence: Required silence duration in seconds.

        Returns:
            True if silence has exceeded the required duration and
            speech was long enough to be valid.
        """
        if self.state != VADState.SILENCE:
            return False

        silence_duration = self.get_silence_duration()
        speech_duration = self.get_speech_duration()

        if (
            silence_duration >= required_silence
            and speech_duration >= self.config.min_speech_duration
        ):
            logger.info("VAD: State changed to \"idle\" (dynamic end-of-turn)")
            logger.debug(
                f"VAD: Dynamic end-of-turn triggered "
                f"(silence={silence_duration:.2f}s >= {required_silence:.2f}s, "
                f"speech={speech_duration:.2f}s)"
            )
            return True

        return False

    def get_average_energy(self) -> float:
        """Get average energy from recent history."""
        if not self._energy_history:
            return 0.0
        return sum(self._energy_history) / len(self._energy_history)
