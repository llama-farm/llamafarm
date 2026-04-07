"""History compression utilities.

Provides lossless and near-lossless compression techniques for
conversation history to reduce token usage before truncation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .token_counter import TokenCounter

logger = logging.getLogger(__name__)


class HistoryCompressor:
    """Compresses conversation history to reduce token usage.

    Applies multiple compression techniques that preserve meaning
    while reducing token count:
    - Whitespace normalization
    - Tool result truncation
    - Code block compression
    - Repetition removal
    """

    # Number of recent messages to leave untouched
    PRESERVE_RECENT = 4

    # Maximum tokens for old tool results
    MAX_TOOL_RESULT_TOKENS = 200

    # Maximum lines for code blocks in old messages
    MAX_CODE_BLOCK_LINES = 20

    def __init__(self, token_counter: TokenCounter | None = None):
        """Initialize history compressor.

        Args:
            token_counter: Optional TokenCounter for token-based compression.
                If not provided, some compression features are limited.
        """
        self._counter = token_counter

    def compress(
        self,
        messages: list[dict],
        preserve_recent: int | None = None,
    ) -> list[dict]:
        """Apply all compression techniques to messages.

        Compresses older messages while preserving the most recent
        messages untouched for immediate context.

        Args:
            messages: List of chat messages.
            preserve_recent: Number of recent messages to preserve (default: 4).

        Returns:
            Compressed messages (deep copy, original unchanged).
        """
        if not messages:
            return messages

        # Use explicit None check to allow preserve_recent=0
        if preserve_recent is None:
            preserve_recent = self.PRESERVE_RECENT

        # Deep copy to avoid modifying original
        # Use JSON for Pydantic-safe deep copy
        messages = json.loads(json.dumps(messages, default=str))

        # Split into old (to compress) and recent (to preserve)
        if len(messages) <= preserve_recent:
            # Not enough to compress, just normalize whitespace
            return self._normalize_all_whitespace(messages)

        old_msgs = messages[:-preserve_recent]
        recent_msgs = messages[-preserve_recent:]

        # Apply compression pipeline to old messages
        old_msgs = self._normalize_all_whitespace(old_msgs)
        old_msgs = self._compress_tool_results(old_msgs)
        old_msgs = self._compress_code_blocks(old_msgs)
        old_msgs = self._remove_repetitions(old_msgs)

        return old_msgs + recent_msgs

    def _normalize_all_whitespace(self, messages: list[dict]) -> list[dict]:
        """Normalize whitespace in all message contents.

        Args:
            messages: List of messages.

        Returns:
            Messages with normalized whitespace.
        """
        for msg in messages:
            content = msg.get("content")
            if content and isinstance(content, str):
                msg["content"] = self._normalize_whitespace(content)
        return messages

    def _normalize_whitespace(self, content: str) -> str:
        """Collapse excessive whitespace.

        Args:
            content: Text content to normalize.

        Returns:
            Normalized content.
        """
        # Collapse multiple newlines to max 2
        content = re.sub(r"\n{3,}", "\n\n", content)
        # Collapse multiple spaces to single (but preserve indentation at line start)
        content = re.sub(r"(?<=\S)  +", " ", content)
        return content.strip()

    def _compress_tool_results(self, messages: list[dict]) -> list[dict]:
        """Truncate verbose tool call results.

        Tool results (file contents, API responses) are often very long.
        After the assistant has processed them, the full content is
        less important for context.

        Args:
            messages: List of messages.

        Returns:
            Messages with compressed tool results.
        """
        for msg in messages:
            if msg.get("role") == "tool":
                content = msg.get("content", "")
                if not content:
                    continue

                # Check token count if counter available
                if self._counter:
                    token_count = self._counter.count_tokens(content)
                    if token_count > self.MAX_TOOL_RESULT_TOKENS:
                        truncated = self._counter.truncate_to_tokens(
                            content, self.MAX_TOOL_RESULT_TOKENS
                        )
                        msg["content"] = truncated + "\n\n[... result truncated ...]"
                else:
                    # Fallback: use character count (rough estimate: 4 chars per token)
                    max_chars = self.MAX_TOOL_RESULT_TOKENS * 4
                    if len(content) > max_chars:
                        msg["content"] = (
                            content[:max_chars] + "\n\n[... result truncated ...]"
                        )

        return messages

    def _compress_code_blocks(self, messages: list[dict]) -> list[dict]:
        """Compress large code blocks in messages.

        Large code blocks in assistant responses can be condensed
        after they've been seen.

        Args:
            messages: List of messages.

        Returns:
            Messages with compressed code blocks.
        """
        code_block_pattern = re.compile(
            r"```(\w*)\n(.*?)```",
            re.DOTALL,
        )

        for msg in messages:
            if msg.get("role") != "assistant":
                continue

            content = msg.get("content", "")
            if not content or "```" not in content:
                continue

            def compress_block(match: re.Match) -> str:
                language = match.group(1) or "code"
                code = match.group(2)
                lines = code.split("\n")

                if len(lines) <= self.MAX_CODE_BLOCK_LINES:
                    return match.group(0)  # Keep original

                # Create summary
                first_lines = "\n".join(lines[:5])
                summary = (
                    f"```{language}\n"
                    f"{first_lines}\n"
                    f"# ... ({len(lines)} lines total) ...\n"
                    f"```"
                )
                return summary

            msg["content"] = code_block_pattern.sub(compress_block, content)

        return messages

    def _remove_repetitions(self, messages: list[dict]) -> list[dict]:
        """Remove duplicate or near-duplicate content.

        If the same content appears multiple times in history,
        keep only the most recent occurrence.

        Args:
            messages: List of messages.

        Returns:
            Messages with repetitions removed.
        """
        seen_hashes: set[str] = set()
        result: list[dict] = []

        # Process in reverse to keep most recent
        for msg in reversed(messages):
            content = msg.get("content", "")

            # Skip empty or very short messages
            if not content or len(content) < 50:
                result.append(msg)
                continue

            # Create hash of normalized content
            normalized = self._normalize_for_hash(content)
            content_hash = hashlib.md5(normalized.encode()).hexdigest()

            if content_hash in seen_hashes:
                logger.debug(f"Removing duplicate message: {content[:50]}...")
                continue

            seen_hashes.add(content_hash)
            result.append(msg)

        # Restore original order
        return list(reversed(result))

    def _normalize_for_hash(self, content: str) -> str:
        """Normalize content for duplicate detection.

        Args:
            content: Text content.

        Returns:
            Normalized content for hashing.
        """
        # Lowercase, collapse whitespace, remove punctuation
        normalized = content.lower()
        normalized = re.sub(r"\s+", " ", normalized)
        normalized = re.sub(r"[^\w\s]", "", normalized)
        return normalized.strip()
