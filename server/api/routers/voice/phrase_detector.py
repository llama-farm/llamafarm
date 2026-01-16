"""
Phrase boundary detection for LLM streaming output.

Accumulates LLM tokens and yields complete phrases at natural boundaries,
enabling low-latency TTS synthesis without choppy playback.
"""

import re
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field

# Phrase boundary patterns
SENTENCE_ENDS = re.compile(r"[.!?](?:\s|$)")
CLAUSE_ENDS = re.compile(r"[;:,](?:\s|$)")
NEWLINE = re.compile(r"\n")


@dataclass
class PhraseBoundaryDetector:
    """Detects natural phrase boundaries in streaming LLM output.

    Accumulates tokens and yields complete phrases when:
    1. A sentence-ending punctuation is found (. ! ?)
    2. A clause-ending punctuation is found (; : ,) if text is long enough
    3. A newline is found
    4. Maximum phrase length is reached (force split)

    This enables TTS to start speaking sooner while maintaining natural
    speech patterns.
    """

    min_phrase_length: int = 10  # Minimum chars before yielding on clause boundary (lower = faster first phrase)
    max_phrase_length: int = 150  # Force yield if accumulated text exceeds this (lower = more responsive)
    sentence_boundary_only: bool = False  # Only split on . ! ? (not ; : ,)

    _buffer: str = field(default="", init=False)

    def reset(self) -> None:
        """Clear the buffer."""
        self._buffer = ""

    def add_token(self, token: str) -> str | None:
        """Add a token and return a phrase if boundary detected.

        Args:
            token: New token to add to buffer.

        Returns:
            Complete phrase if boundary detected, None otherwise.
        """
        self._buffer += token

        # Check for forced split at max length
        if len(self._buffer) >= self.max_phrase_length:
            # Try to find a good split point
            phrase = self._find_best_split()
            if phrase:
                return phrase

        # Check for sentence boundaries (strongest)
        match = SENTENCE_ENDS.search(self._buffer)
        if match:
            # Include the punctuation
            end_idx = match.end()
            phrase = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx:].lstrip()
            if phrase:
                return phrase

        # Check for newlines
        match = NEWLINE.search(self._buffer)
        if match and len(self._buffer[:match.start()].strip()) >= self.min_phrase_length:
            end_idx = match.end()
            phrase = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx:].lstrip()
            if phrase:
                return phrase

        # Check for clause boundaries (weaker, needs min length)
        if not self.sentence_boundary_only:
            match = CLAUSE_ENDS.search(self._buffer)
            if match and len(self._buffer[:match.end()].strip()) >= self.min_phrase_length:
                end_idx = match.end()
                phrase = self._buffer[:end_idx].strip()
                self._buffer = self._buffer[end_idx:].lstrip()
                if phrase:
                    return phrase

        return None

    def _find_best_split(self) -> str | None:
        """Find the best split point when max length exceeded."""
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

        # Try clause boundary
        if not self.sentence_boundary_only:
            match = CLAUSE_ENDS.search(self._buffer)
            if match:
                end_idx = match.end()
                phrase = self._buffer[:end_idx].strip()
                self._buffer = self._buffer[end_idx:].lstrip()
                return phrase

        # Last resort: split at word boundary near max length
        words = self._buffer.split()
        if len(words) > 1:
            # Take roughly 3/4 of words
            split_idx = max(1, len(words) * 3 // 4)
            phrase_words = words[:split_idx]
            remaining_words = words[split_idx:]
            phrase = " ".join(phrase_words)
            self._buffer = " ".join(remaining_words)
            return phrase

        # Nothing to split
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
    min_phrase_length: int = 10,
    max_phrase_length: int = 150,
) -> AsyncGenerator[str, None]:
    """Async generator that yields phrases from a token stream.

    Args:
        token_stream: Async generator yielding LLM tokens.
        min_phrase_length: Minimum chars before clause boundary triggers yield.
        max_phrase_length: Maximum chars before forced yield.

    Yields:
        Complete phrases suitable for TTS synthesis.
    """
    detector = PhraseBoundaryDetector(
        min_phrase_length=min_phrase_length,
        max_phrase_length=max_phrase_length,
    )

    async for token in token_stream:
        phrase = detector.add_token(token)
        if phrase:
            yield phrase

    # Flush any remaining content
    final_phrase = detector.flush()
    if final_phrase:
        yield final_phrase
