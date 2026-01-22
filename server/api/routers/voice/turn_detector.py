"""
End-of-turn detection with linguistic completeness analysis.

Distinguishes between "thinking pauses" (mid-thought hesitation) and
genuine end-of-utterance by analyzing partial transcriptions for
linguistic completeness markers.

This prevents the LLM from responding prematurely when a user pauses
to think but hasn't finished their thought.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class TurnCompleteness(str, Enum):
    """Assessment of whether an utterance appears complete."""

    COMPLETE = "complete"  # Grammatically complete, likely done
    INCOMPLETE = "incomplete"  # Trailing markers suggest more to come
    AMBIGUOUS = "ambiguous"  # Can't determine, use default threshold


@dataclass
class TurnDetectorConfig:
    """Configuration for end-of-turn detection."""

    # Base silence threshold for complete utterances (seconds)
    base_silence_duration: float = 0.4

    # Extended silence threshold for incomplete utterances (seconds)
    # User is likely thinking, wait longer before triggering
    thinking_silence_duration: float = 1.2

    # Maximum silence before forcing end-of-turn (seconds)
    # Even if utterance seems incomplete, don't wait forever
    max_silence_duration: float = 2.5

    # Minimum speech duration before analyzing completeness (seconds)
    # Very short utterances get longer silence threshold regardless
    min_speech_for_analysis: float = 0.5

    # Speech duration threshold for "short" utterances (seconds)
    # Short utterances get extended silence (user might be starting)
    short_utterance_threshold: float = 2.0

    # Multiplier for silence threshold on short utterances
    short_utterance_silence_multiplier: float = 1.5

    # Enable linguistic analysis (can be disabled for testing)
    enable_linguistic_analysis: bool = True


# Patterns that suggest the utterance is INCOMPLETE (user will continue)
INCOMPLETE_PATTERNS = [
    # Trailing conjunctions - user is connecting thoughts
    r"\b(and|but|or|so|because|since|although|though|while|if|when|"
    r"unless|until|after|before|as|than|that|which|who|whom|whose|where)\s*$",
    # Trailing prepositions - incomplete phrase
    r"\b(to|for|with|at|by|from|in|on|of|about|into|onto|upon|"
    r"through|during|without|within|between|among|under|over|after|before)\s*$",
    # Trailing articles/determiners - noun phrase incomplete
    r"\b(the|a|an|this|that|these|those|my|your|his|her|its|our|their|some|any|no)\s*$",
    # Trailing auxiliary verbs - predicate incomplete
    r"\b(is|are|was|were|be|been|being|have|has|had|do|does|did|"
    r"will|would|shall|should|can|could|may|might|must)\s*$",
    # Filler words/hesitation markers - user is thinking
    r"\b(um|uh|er|ah|like|you know|i mean|well|so|anyway|basically)\s*$",
    # Trailing "I" or "you" etc. without verb - incomplete subject
    r"\b(i|you|we|they|he|she|it)\s*$",
    # List indicators - more items coming
    r"\b(first|second|third|one|two|three|firstly|secondly|finally|also|another)\s*$",
    # Comparative without completion
    r"\b(more|less|better|worse|larger|smaller|faster|slower)\s+than\s*$",
    # Incomplete quotes
    r'["\'][^"\']*$',
    # Ellipsis or trailing dots (explicit continuation)
    r"\.{2,}\s*$",
    # Comma at end (list or clause continues)
    r",\s*$",
    # Colon at end (explanation coming)
    r":\s*$",
]

# Patterns that suggest the utterance is COMPLETE
COMPLETE_PATTERNS = [
    # Sentence-ending punctuation
    r"[.!?]\s*$",
    # Common complete phrases
    r"\b(yes|no|yeah|yep|nope|okay|ok|sure|thanks|thank you|please|"
    r"got it|i see|right|correct|exactly|absolutely|definitely)\s*[.!?]?\s*$",
    # Questions (even without ?)
    r"\b(what|where|when|why|how|who|which|is it|are you|can you|"
    r"do you|does it|will you|would you|could you)[^.!?]*[?]?\s*$",
    # Commands/requests ending naturally
    r"\b(stop|start|go|come|help|tell me|show me|give me|let me|"
    r"make it|do it|try it)\b[^,]*\s*$",
]

# Compiled patterns for efficiency
_incomplete_re = [re.compile(p, re.IGNORECASE) for p in INCOMPLETE_PATTERNS]
_complete_re = [re.compile(p, re.IGNORECASE) for p in COMPLETE_PATTERNS]


def analyze_completeness(text: str) -> TurnCompleteness:
    """Analyze text to determine if it appears linguistically complete.

    Args:
        text: Partial or complete transcription.

    Returns:
        TurnCompleteness assessment.
    """
    if not text or not text.strip():
        return TurnCompleteness.AMBIGUOUS

    text = text.strip()

    # Check for incompleteness markers FIRST - these take precedence
    # because trailing prepositions/conjunctions indicate more to come
    # even if the beginning of the sentence looks like a question/command
    for pattern in _incomplete_re:
        if pattern.search(text):
            logger.debug("Turn analysis: INCOMPLETE (matched incomplete pattern)")
            return TurnCompleteness.INCOMPLETE

    # Check for explicit completeness markers
    for pattern in _complete_re:
        if pattern.search(text):
            logger.debug("Turn analysis: COMPLETE (matched complete pattern)")
            return TurnCompleteness.COMPLETE

    # Heuristic: very short text without punctuation is likely incomplete
    words = text.split()
    if len(words) <= 2 and not re.search(r"[.!?]$", text):
        logger.debug("Turn analysis: INCOMPLETE (very short, no punctuation)")
        return TurnCompleteness.INCOMPLETE

    # Default: ambiguous
    logger.debug("Turn analysis: AMBIGUOUS (no patterns matched)")
    return TurnCompleteness.AMBIGUOUS


@dataclass
class EndOfTurnDetector:
    """Detects end-of-turn using silence duration and linguistic analysis.

    Integrates with VAD to provide smarter silence thresholds based on
    the content of what the user has said so far.
    """

    config: TurnDetectorConfig = field(default_factory=TurnDetectorConfig)

    # Last analyzed text (for caching/debugging)
    _last_text: str = ""
    _last_completeness: TurnCompleteness = TurnCompleteness.AMBIGUOUS

    def get_required_silence(
        self,
        partial_transcript: str,
        speech_duration: float,
    ) -> float:
        """Calculate required silence duration based on context.

        Args:
            partial_transcript: Current transcription of user speech.
            speech_duration: How long the user has been speaking (seconds).

        Returns:
            Required silence duration before triggering end-of-turn (seconds).
        """
        # Always respect maximum
        base = self.config.base_silence_duration

        # Very short utterances need longer silence (user might be starting)
        if speech_duration < self.config.short_utterance_threshold:
            base = base * self.config.short_utterance_silence_multiplier
            logger.debug(
                f"Short utterance ({speech_duration:.2f}s), "
                f"base silence extended to {base:.2f}s"
            )

        # Skip linguistic analysis if disabled or not enough speech
        if (
            not self.config.enable_linguistic_analysis
            or speech_duration < self.config.min_speech_for_analysis
        ):
            return min(base, self.config.max_silence_duration)

        # Analyze linguistic completeness
        completeness = analyze_completeness(partial_transcript)
        self._last_text = partial_transcript
        self._last_completeness = completeness

        if completeness == TurnCompleteness.COMPLETE:
            # User seems done, use base threshold
            required = base
        elif completeness == TurnCompleteness.INCOMPLETE:
            # User seems to be thinking, wait longer
            required = self.config.thinking_silence_duration
        else:
            # Ambiguous, use slightly extended threshold
            required = base * 1.25

        # Clamp to max
        required = min(required, self.config.max_silence_duration)

        logger.debug(
            f"Required silence: {required:.2f}s "
            f"(completeness={completeness.value}, speech={speech_duration:.2f}s)"
        )

        return required

    def should_end_turn(
        self,
        silence_duration: float,
        speech_duration: float,
        partial_transcript: str,
    ) -> bool:
        """Determine if the user's turn should end.

        Args:
            silence_duration: How long silence has lasted (seconds).
            speech_duration: How long user spoke before silence (seconds).
            partial_transcript: Current transcription of user speech.

        Returns:
            True if turn should end and LLM processing should begin.
        """
        # Hard maximum - always end turn
        if silence_duration >= self.config.max_silence_duration:
            logger.info(
                f"End of turn: max silence reached ({silence_duration:.2f}s >= "
                f"{self.config.max_silence_duration:.2f}s)"
            )
            return True

        # Get context-aware silence threshold
        required = self.get_required_silence(partial_transcript, speech_duration)

        if silence_duration >= required:
            logger.info(
                f"End of turn: silence threshold reached ({silence_duration:.2f}s >= "
                f"{required:.2f}s, completeness={self._last_completeness.value})"
            )
            return True

        return False

    def reset(self) -> None:
        """Reset detector state for new utterance."""
        self._last_text = ""
        self._last_completeness = TurnCompleteness.AMBIGUOUS
