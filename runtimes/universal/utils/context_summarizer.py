"""Context summarization using an LLM.

Provides LLM-based summarization of conversation history to preserve
semantic meaning while dramatically reducing token count.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.gguf_language_model import GGUFLanguageModel

logger = logging.getLogger(__name__)


# Prompt for summarizing conversation history
SUMMARIZE_PROMPT = """Summarize the following conversation concisely, preserving:
- Key facts and decisions made
- Important context the assistant needs to remember
- Any commitments or action items
- Technical details that may be referenced later

Be concise but complete. Write in third person (e.g., "The user asked about X. The assistant explained Y.").

Conversation:
{conversation}

Summary:"""


class ContextSummarizer:
    """Summarizes conversation history using an LLM.

    When context exceeds budget, this class can compress older messages
    into a single summary message, preserving semantic meaning while
    dramatically reducing token count.

    Uses the server's model loading mechanism to benefit from caching
    and proper lifecycle management.
    """

    # Default model for summarization (small, fast, good at instruction following)
    # Qwen3 has better instruction following than Qwen2.5 for summarization tasks
    DEFAULT_MODEL = "Qwen/Qwen3-1.7B-GGUF"
    DEFAULT_QUANTIZATION = "Q4_K_M"

    # Default number of recent exchanges to preserve
    DEFAULT_KEEP_RECENT = 4

    def __init__(
        self,
        model_id: str | None = None,
        quantization: str | None = None,
        keep_recent: int | None = None,
        load_language: Callable | None = None,
    ):
        """Initialize context summarizer.

        Args:
            model_id: HuggingFace model ID for summarization (default: Qwen2.5-1.5B).
            quantization: GGUF quantization preference (default: Q4_K_M).
            keep_recent: Number of recent exchanges to preserve (default: 4).
            load_language: Model loader function (uses server's loader for caching).
        """
        self._model_id = model_id or self.DEFAULT_MODEL
        self._quantization = quantization or self.DEFAULT_QUANTIZATION
        # Use explicit None check to allow keep_recent=0
        self._keep_recent = (
            keep_recent if keep_recent is not None else self.DEFAULT_KEEP_RECENT
        )
        self._load_language = load_language
        self._model: GGUFLanguageModel | None = None

    async def ensure_model_loaded(self) -> None:
        """Load the summarization model using the server's caching mechanism."""
        if self._model is not None:
            return

        # Use the server's load_language function for proper caching
        if self._load_language is not None:
            logger.info(
                f"Loading summarization model via server cache: {self._model_id}"
            )
            self._model = await self._load_language(
                self._model_id,
                n_ctx=4096,
                preferred_quantization=self._quantization,
            )
            logger.info("Summarization model loaded (cached by server)")
        else:
            # Fallback: import server's loader directly
            try:
                from server import load_language

                logger.info(f"Loading summarization model: {self._model_id}")
                self._model = await load_language(
                    self._model_id,
                    n_ctx=4096,
                    preferred_quantization=self._quantization,
                )
                logger.info("Summarization model loaded successfully")
            except ImportError:
                # Last resort: create model directly (won't be cached)
                from models.gguf_language_model import GGUFLanguageModel

                logger.warning(
                    f"Loading summarization model directly (not cached): {self._model_id}"
                )
                self._model = GGUFLanguageModel(
                    model_id=self._model_id,
                    device="cpu",
                    n_ctx=4096,
                    preferred_quantization=self._quantization,
                )
                await self._model.load()

    async def summarize_messages(
        self,
        messages: list[dict],
        keep_recent: int | None = None,
    ) -> list[dict]:
        """Summarize older messages, keeping recent ones intact.

        Args:
            messages: List of chat messages.
            keep_recent: Number of recent exchanges to keep (default: 4).

        Returns:
            Messages with older content summarized into a single message.
        """
        # Use explicit None check to allow keep_recent=0
        if keep_recent is None:
            keep_recent = self._keep_recent

        # Separate system messages from conversation
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # Check if we have enough messages to summarize
        # keep_recent * 2 because each exchange is user + assistant
        min_messages = keep_recent * 2
        if len(other_msgs) <= min_messages:
            logger.debug("Not enough messages to summarize")
            return messages

        # Split into old (to summarize) and recent (to keep)
        # Handle min_messages=0 specially since [:-0] returns [] and [-0:] returns all
        if min_messages == 0:
            to_summarize = other_msgs
            to_keep = []
        else:
            to_summarize = other_msgs[:-min_messages]
            to_keep = other_msgs[-min_messages:]

        logger.info(
            f"Summarizing {len(to_summarize)} messages, keeping {len(to_keep)} recent"
        )

        # Ensure model is loaded
        await self.ensure_model_loaded()

        # Generate summary
        summary = await self._generate_summary(to_summarize)

        # Create summary message as a system-level context
        summary_msg = {
            "role": "system",
            "content": f"[Conversation Summary]\n{summary}",
        }

        # Return: original system + summary + recent messages
        return system_msgs + [summary_msg] + to_keep

    async def _generate_summary(self, messages: list[dict]) -> str:
        """Generate a summary of the given messages.

        Args:
            messages: List of messages to summarize.

        Returns:
            Summary text.
        """
        if self._model is None:
            raise RuntimeError("Summarization model not loaded")

        # Format messages for summarization
        conversation_text = self._format_for_summary(messages)

        # Build the summarization prompt
        prompt = SUMMARIZE_PROMPT.format(conversation=conversation_text)

        # Generate summary using the model
        summary = await self._model.generate(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,  # Limit summary length
            temperature=0.3,  # Lower temperature for more focused summary
        )

        return summary.strip()

    def _format_for_summary(self, messages: list[dict]) -> str:
        """Format messages for summarization prompt.

        Args:
            messages: List of messages.

        Returns:
            Formatted conversation text.
        """
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")

            if not content:
                continue

            # Capitalize role for readability
            role_label = role.capitalize()
            if role == "assistant":
                role_label = "Assistant"
            elif role == "user":
                role_label = "User"
            elif role == "tool":
                role_label = "Tool Result"

            # Truncate very long messages for the summary input
            if len(content) > 1000:
                content = content[:1000] + "..."

            parts.append(f"{role_label}: {content}")

        return "\n\n".join(parts)

    # Note: No explicit unload() method - the model is managed by the server's
    # cache and will be evicted based on the normal cache TTL policy.
