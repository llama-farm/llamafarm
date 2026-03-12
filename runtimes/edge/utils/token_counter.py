"""Token counting utilities for context management.

Provides token counting functionality using the model's tokenizer
for accurate context window management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llamafarm_llama import Llama

logger = logging.getLogger(__name__)


class TokenCounter:
    """Token counter using the model's tokenizer.

    Provides methods for counting tokens in text and messages,
    enabling accurate context window management.
    """

    # Estimated token overhead per message for role markers and formatting
    MESSAGE_OVERHEAD = 4

    # Chat template overhead factor (10% buffer for template markers)
    TEMPLATE_OVERHEAD_FACTOR = 1.10

    def __init__(self, llama: Llama):
        """Initialize token counter with a Llama model instance.

        Args:
            llama: A loaded Llama model instance with tokenize() method.
        """
        self._llama = llama

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: The text to tokenize.

        Returns:
            Number of tokens in the text.
        """
        if not text:
            return 0

        tokens = self._llama.tokenize(text, add_special=False, parse_special=True)
        return len(tokens)

    def count_message_tokens(self, message: dict) -> int:
        """Count tokens for a single message including role overhead.

        The overhead accounts for role markers (e.g., "<|user|>") and
        other formatting added by chat templates.

        Handles both text-only messages (content is string) and multimodal
        messages (content is list of content parts).

        Args:
            message: A message dict with 'role' and 'content' keys.

        Returns:
            Estimated token count for the message.
        """
        content = message.get("content") or ""

        # Handle multimodal messages (content is a list of parts)
        if isinstance(content, list):
            total_tokens = 0
            for part in content:
                if isinstance(part, dict):
                    part_type = part.get("type", "")
                    if part_type == "text":
                        # Count tokens in text parts
                        text = part.get("text", "")
                        total_tokens += self.count_tokens(text)
                    elif part_type == "input_audio":
                        # Audio parts don't contribute text tokens
                        # Use a small estimate for the audio marker/placeholder
                        total_tokens += 10
                    elif part_type == "image_url":
                        # Image parts - use a moderate estimate
                        total_tokens += 50
                    # Skip other unknown types
            return total_tokens + self.MESSAGE_OVERHEAD

        # Handle simple string content
        return self.count_tokens(content) + self.MESSAGE_OVERHEAD

    def count_messages_tokens(self, messages: list[dict]) -> int:
        """Count total tokens for a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            Total token count for all messages.
        """
        return sum(self.count_message_tokens(m) for m in messages)

    def estimate_prompt_tokens(
        self,
        messages: list[dict],
        include_template_overhead: bool = True,
    ) -> int:
        """Estimate total prompt tokens including chat template overhead.

        This is an estimate because the exact token count after template
        application depends on the specific model's chat template. The
        10% overhead is a conservative buffer that works for most templates.

        Args:
            messages: List of message dicts.
            include_template_overhead: Whether to add 10% overhead for chat
                template markers (BOS token, role tokens, etc.).

        Returns:
            Estimated token count for the prompt.
        """
        base_tokens = self.count_messages_tokens(messages)

        if include_template_overhead:
            return int(base_tokens * self.TEMPLATE_OVERHEAD_FACTOR)

        return base_tokens

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to a maximum number of tokens.

        Useful for truncating long tool results or code blocks.

        Args:
            text: The text to truncate.
            max_tokens: Maximum number of tokens to keep.

        Returns:
            Truncated text (may be the original if within limits).
        """
        if not text:
            return text

        tokens = self._llama.tokenize(text, add_special=False, parse_special=True)

        if len(tokens) <= max_tokens:
            return text

        # Truncate tokens and detokenize
        truncated_tokens = tokens[:max_tokens]
        return self._llama.detokenize(truncated_tokens)
