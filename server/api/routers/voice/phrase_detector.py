"""
Phrase boundary detection for LLM streaming output.

Accumulates LLM tokens and yields complete phrases at natural boundaries,
enabling low-latency TTS synthesis without choppy playback.

Optimized for natural-sounding TTS by keeping phrases short (under 60 chars).
Neural TTS models sound more human with shorter utterances.
"""

import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

# Phrase boundary patterns (ordered by strength)
# Sentence endings - strongest boundary, always split
SENTENCE_ENDS = re.compile(r"[.!?](?:\s|$)")

# Clause endings - split if phrase is long enough
CLAUSE_ENDS = re.compile(r"[;:,](?:\s|$)")

# Dashes and parentheses - natural breath points
DASH_BREAKS = re.compile(r"(?:\s[-–—]\s|—)(?=\S)")  # " - ", " – ", "—"
PAREN_CLOSE = re.compile(r"\)(?:\s|$)")  # End of parenthetical

# Conjunctions - split before them if phrase is getting long
CONJUNCTIONS = re.compile(r"\s(?:and|or|but|so|yet)\s", re.IGNORECASE)

NEWLINE = re.compile(r"\n")


@dataclass
class PhraseBoundaryDetector:
    """Detects natural phrase boundaries in streaming LLM output.

    Accumulates tokens and yields complete phrases when:
    1. A sentence-ending punctuation is found (. ! ?)
    2. A clause-ending punctuation is found (; : ,) if text is long enough
    3. A dash or parenthetical boundary is found
    4. A conjunction is found (and/or/but) if phrase is long enough
    5. A newline is found
    6. Maximum phrase length is reached (force split)

    Modes:
    - sentence_boundary_only=True (default): Only split on sentences (. ! ?) and newlines.
      Best for natural-sounding speech - avoids weird mid-sentence breaks.
    - sentence_boundary_only=False: Aggressive chunking for lower latency but choppier speech.

    First phrase optimization: Uses a lower threshold for the first phrase
    to minimize time-to-first-audio (e.g., "Sure," triggers immediately).
    """

    # Balance: short enough for good TTS prosody, long enough to avoid choppy playback
    # Too short = gaps between phrases, too long = robotic prosody
    min_phrase_length: int = 12  # Minimum chars before yielding on weak boundary
    max_phrase_length: int = 500  # Force yield - high default to avoid mid-sentence breaks
    max_word_count: int = 80  # Alternative limit by word count
    first_phrase_min_length: int = 5  # First phrase threshold (e.g., "Sure," or "Hello!")
    conjunction_min_length: int = 40  # Only split on and/or/but if phrase is this long
    sentence_boundary_only: bool = True  # Only split on . ! ? and newlines (recommended)

    _buffer: str = field(default="", init=False)
    _first_phrase_emitted: bool = field(default=False, init=False)

    def reset(self) -> None:
        """Clear the buffer and reset first phrase flag."""
        self._buffer = ""
        self._first_phrase_emitted = False

    def _word_count(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def add_token(self, token: str) -> str | None:
        """Add a token and return a phrase if boundary detected.

        Uses a lower threshold for the first phrase to minimize
        time-to-first-audio. Prioritizes shorter phrases for better
        TTS prosody.

        Args:
            token: New token to add to buffer.

        Returns:
            Complete phrase if boundary detected, None otherwise.
        """
        self._buffer += token

        # Determine minimum length based on whether first phrase was emitted
        effective_min_length = (
            self.first_phrase_min_length
            if not self._first_phrase_emitted
            else self.min_phrase_length
        )

        buffer_len = len(self._buffer.strip())
        word_count = self._word_count(self._buffer)

        # Check for forced split at max length or word count
        if buffer_len >= self.max_phrase_length or word_count >= self.max_word_count:
            # Try to find a good split point
            phrase = self._find_best_split()
            if phrase:
                self._first_phrase_emitted = True
                return phrase

        # Check for sentence boundaries (strongest) - always yield
        match = SENTENCE_ENDS.search(self._buffer)
        if match:
            # Include the punctuation
            end_idx = match.end()
            phrase = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx:].lstrip()
            if phrase:
                self._first_phrase_emitted = True
                return phrase

        # Check for newlines
        match = NEWLINE.search(self._buffer)
        if match and len(self._buffer[:match.start()].strip()) >= effective_min_length:
            end_idx = match.end()
            phrase = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx:].lstrip()
            if phrase:
                self._first_phrase_emitted = True
                return phrase

        # The following boundaries are only checked when sentence_boundary_only is False
        # (aggressive chunking mode for lower latency but choppier speech)
        if not self.sentence_boundary_only:
            # Check for clause boundaries (semi-strong)
            match = CLAUSE_ENDS.search(self._buffer)
            if match and len(self._buffer[:match.end()].strip()) >= effective_min_length:
                end_idx = match.end()
                phrase = self._buffer[:end_idx].strip()
                self._buffer = self._buffer[end_idx:].lstrip()
                if phrase:
                    self._first_phrase_emitted = True
                    return phrase

            # Check for dash breaks (natural breath points)
            if buffer_len >= effective_min_length:
                match = DASH_BREAKS.search(self._buffer)
                if match:
                    # Split before the dash
                    end_idx = match.start()
                    phrase = self._buffer[:end_idx].strip()
                    self._buffer = self._buffer[end_idx:].lstrip()
                    if phrase:
                        self._first_phrase_emitted = True
                        return phrase

            # Check for end of parenthetical
            if buffer_len >= effective_min_length:
                match = PAREN_CLOSE.search(self._buffer)
                if match:
                    end_idx = match.end()
                    phrase = self._buffer[:end_idx].strip()
                    self._buffer = self._buffer[end_idx:].lstrip()
                    if phrase:
                        self._first_phrase_emitted = True
                        return phrase

            # Check for conjunctions (weakest - only if phrase is getting long)
            if buffer_len >= self.conjunction_min_length:
                match = CONJUNCTIONS.search(self._buffer)
                if match:
                    # Split before the conjunction
                    end_idx = match.start()
                    phrase = self._buffer[:end_idx].strip()
                    # Keep the conjunction for the next phrase
                    self._buffer = self._buffer[end_idx:].lstrip()
                    if phrase:
                        self._first_phrase_emitted = True
                        return phrase

        return None

    def _find_best_split(self) -> str | None:
        """Find the best split point when max length exceeded.

        Tries boundaries in order of naturalness:
        1. Sentence end (. ! ?)
        2. Newline
        3. Clause boundary (; : ,) - only if not sentence_boundary_only
        4. Dash / parenthetical - only if not sentence_boundary_only
        5. Conjunction (and/or/but) - only if not sentence_boundary_only
        6. Word boundary (last resort)
        """
        # Try sentence boundary first
        match = SENTENCE_ENDS.search(self._buffer)
        if match:
            end_idx = match.end()
            phrase = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx:].lstrip()
            return phrase

        # Try newline
        match = NEWLINE.search(self._buffer)
        if match:
            end_idx = match.end()
            phrase = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx:].lstrip()
            return phrase

        # The following are only tried if not in sentence_boundary_only mode
        if not self.sentence_boundary_only:
            # Try clause boundary
            match = CLAUSE_ENDS.search(self._buffer)
            if match:
                end_idx = match.end()
                phrase = self._buffer[:end_idx].strip()
                self._buffer = self._buffer[end_idx:].lstrip()
                return phrase

            # Try dash break
            match = DASH_BREAKS.search(self._buffer)
            if match:
                end_idx = match.start()
                phrase = self._buffer[:end_idx].strip()
                self._buffer = self._buffer[end_idx:].lstrip()
                if phrase:
                    return phrase

            # Try parenthetical close
            match = PAREN_CLOSE.search(self._buffer)
            if match:
                end_idx = match.end()
                phrase = self._buffer[:end_idx].strip()
                self._buffer = self._buffer[end_idx:].lstrip()
                if phrase:
                    return phrase

            # Try conjunction
            match = CONJUNCTIONS.search(self._buffer)
            if match:
                end_idx = match.start()
                phrase = self._buffer[:end_idx].strip()
                self._buffer = self._buffer[end_idx:].lstrip()
                if phrase:
                    return phrase

        # Last resort: split at word boundary
        words = self._buffer.split()
        if len(words) > 1:
            # For shorter, more natural phrases, take half the words
            split_idx = max(1, len(words) // 2)
            phrase_words = words[:split_idx]
            remaining_words = words[split_idx:]
            phrase = " ".join(phrase_words)
            self._buffer = " ".join(remaining_words)
            return phrase

        # Force split: single long token (e.g., URL) exceeding max length
        # Return the entire buffer to prevent unbounded growth
        if self._buffer.strip():
            phrase = self._buffer.strip()
            self._buffer = ""
            return phrase

        return None

    def flush(self) -> str | None:
        """Flush remaining buffer content.

        Call this when LLM stream ends to get any remaining text.

        Returns:
            Remaining phrase if buffer has content, None otherwise.
        """
        if self._buffer.strip():
            phrase = self._buffer.strip()
            self._buffer = ""
            return phrase
        return None


async def detect_phrases(
    token_stream: AsyncGenerator[str, None],
    min_phrase_length: int = 12,
    max_phrase_length: int = 500,
    max_word_count: int = 80,
    first_phrase_min_length: int = 5,
    sentence_boundary_only: bool = True,
) -> AsyncGenerator[str, None]:
    """Async generator that yields phrases from a token stream.

    Args:
        token_stream: Async generator yielding LLM tokens.
        min_phrase_length: Minimum chars before weak boundary triggers yield.
        max_phrase_length: Maximum chars before forced yield.
        max_word_count: Maximum words before forced yield.
        first_phrase_min_length: Lower threshold for first phrase (faster TTFA).
        sentence_boundary_only: If True (default), only split on sentence endings
            (. ! ?) for more natural speech. If False, also split on commas,
            semicolons, dashes, and conjunctions for lower latency but choppier speech.

    Yields:
        Complete phrases suitable for TTS synthesis.
    """
    detector = PhraseBoundaryDetector(
        min_phrase_length=min_phrase_length,
        max_phrase_length=max_phrase_length,
        max_word_count=max_word_count,
        first_phrase_min_length=first_phrase_min_length,
        sentence_boundary_only=sentence_boundary_only,
    )

    async for token in token_stream:
        phrase = detector.add_token(token)
        if phrase:
            yield phrase

    # Flush any remaining content
    final_phrase = detector.flush()
    if final_phrase:
        yield final_phrase
