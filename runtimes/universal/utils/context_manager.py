"""Context management and truncation strategies.

Provides context window management for LLM conversations, including
validation, truncation, and multiple strategies for handling context overflow.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .token_counter import TokenCounter

logger = logging.getLogger(__name__)


class TruncationStrategy(Enum):
    """Available truncation strategies for context overflow."""

    # Remove oldest messages first (simple sliding window)
    SLIDING_WINDOW = "sliding_window"

    # Keep system messages, slide user/assistant messages
    KEEP_SYSTEM_SLIDING = "keep_system"

    # Keep system, first exchange, and recent messages; remove middle
    MIDDLE_OUT = "middle_out"

    # Summarize older messages using an LLM (requires summarizer)
    SUMMARIZE = "summarize"


@dataclass
class ContextBudget:
    """Token budget allocation for context window.

    Splits the context window into regions:
    - prompt: tokens for input messages
    - completion: tokens reserved for model output
    - safety_margin: buffer to avoid edge cases
    """

    total_context: int
    max_prompt_tokens: int
    reserved_completion: int
    safety_margin: int

    @classmethod
    def from_context_size(
        cls,
        n_ctx: int,
        max_completion_tokens: int = 512,
        safety_margin_pct: float = 0.05,
    ) -> ContextBudget:
        """Create a budget from model's context size.

        Args:
            n_ctx: Model's total context window size in tokens.
            max_completion_tokens: Tokens to reserve for output (default: 512).
            safety_margin_pct: Percentage of context as safety buffer (default: 5%).

        Returns:
            A ContextBudget instance with calculated allocations.
        """
        safety_margin = int(n_ctx * safety_margin_pct)
        max_prompt = n_ctx - max_completion_tokens - safety_margin

        return cls(
            total_context=n_ctx,
            max_prompt_tokens=max_prompt,
            reserved_completion=max_completion_tokens,
            safety_margin=safety_margin,
        )


@dataclass
class ContextUsage:
    """Context usage information for API responses.

    Provides visibility into how the context window is being used,
    including whether truncation was applied.
    """

    total_context: int
    prompt_tokens: int
    available_for_completion: int
    truncated: bool = False
    truncated_messages: int = 0
    strategy_used: str | None = None


class ContextManager:
    """Manages context window and applies truncation strategies.

    Validates that messages fit within the context budget and applies
    truncation strategies when needed to prevent overflow errors.
    """

    def __init__(
        self,
        token_counter: TokenCounter,
        budget: ContextBudget,
        default_strategy: TruncationStrategy = TruncationStrategy.SUMMARIZE,
    ):
        """Initialize context manager.

        Args:
            token_counter: TokenCounter instance for counting tokens.
            budget: ContextBudget defining token allocations.
            default_strategy: Default truncation strategy to use.
        """
        self._counter = token_counter
        self._budget = budget
        self._default_strategy = default_strategy

    @property
    def budget(self) -> ContextBudget:
        """Get the context budget."""
        return self._budget

    def validate_messages(self, messages: list[dict]) -> ContextUsage:
        """Validate messages fit within context budget.

        Returns usage info without modifying messages.

        Args:
            messages: List of chat messages to validate.

        Returns:
            ContextUsage with token counts and overflow status.
        """
        prompt_tokens = self._counter.estimate_prompt_tokens(messages)
        available = self._budget.total_context - prompt_tokens

        return ContextUsage(
            total_context=self._budget.total_context,
            prompt_tokens=prompt_tokens,
            available_for_completion=max(0, available - self._budget.safety_margin),
            truncated=False,
            truncated_messages=0,
            strategy_used=None,
        )

    def needs_truncation(self, messages: list[dict]) -> bool:
        """Check if messages exceed the context budget.

        Args:
            messages: List of chat messages.

        Returns:
            True if truncation is needed.
        """
        prompt_tokens = self._counter.estimate_prompt_tokens(messages)
        return prompt_tokens > self._budget.max_prompt_tokens

    def truncate_if_needed(
        self,
        messages: list[dict],
        strategy: TruncationStrategy | None = None,
    ) -> tuple[list[dict], ContextUsage]:
        """Truncate messages to fit context budget if needed.

        Args:
            messages: List of chat messages.
            strategy: Override default truncation strategy.

        Returns:
            Tuple of (possibly truncated messages, context usage info).
        """
        strategy = strategy or self._default_strategy
        prompt_tokens = self._counter.estimate_prompt_tokens(messages)

        if prompt_tokens <= self._budget.max_prompt_tokens:
            # No truncation needed
            # Calculate available tokens consistently with validate_messages (subtract safety_margin)
            available = self._budget.total_context - prompt_tokens
            return messages, ContextUsage(
                total_context=self._budget.total_context,
                prompt_tokens=prompt_tokens,
                available_for_completion=max(0, available - self._budget.safety_margin),
                truncated=False,
                truncated_messages=0,
                strategy_used=None,
            )

        # Deep copy to avoid modifying original
        # Use JSON for Pydantic-safe deep copy
        messages = json.loads(json.dumps(messages, default=str))
        original_count = len(messages)

        # Apply truncation strategy
        if strategy == TruncationStrategy.SLIDING_WINDOW:
            truncated = self._sliding_window(messages)
        elif strategy == TruncationStrategy.KEEP_SYSTEM_SLIDING:
            truncated = self._keep_system_sliding(messages)
        elif strategy == TruncationStrategy.MIDDLE_OUT:
            truncated = self._middle_out(messages)
        elif strategy == TruncationStrategy.SUMMARIZE:
            # Summarization is async and handled separately
            # Fall back to keep_system_sliding for sync truncation
            logger.warning(
                "Summarization strategy requires async handling, "
                "falling back to keep_system_sliding"
            )
            truncated = self._keep_system_sliding(messages)
        else:
            # Default fallback
            truncated = self._keep_system_sliding(messages)

        new_tokens = self._counter.estimate_prompt_tokens(truncated)
        messages_removed = original_count - len(truncated)

        logger.info(
            f"Context truncated: {original_count} -> {len(truncated)} messages "
            f"({prompt_tokens} -> {new_tokens} tokens), strategy={strategy.value}"
        )

        # Calculate available tokens consistently with validate_messages (subtract safety_margin)
        available = self._budget.total_context - new_tokens
        return truncated, ContextUsage(
            total_context=self._budget.total_context,
            prompt_tokens=new_tokens,
            available_for_completion=max(0, available - self._budget.safety_margin),
            truncated=True,
            truncated_messages=messages_removed,
            strategy_used=strategy.value,
        )

    def _sliding_window(self, messages: list[dict]) -> list[dict]:
        """Remove oldest messages until context fits.

        Simple strategy that removes messages from the beginning,
        regardless of role. Falls back to content truncation if
        needed.

        Args:
            messages: List of messages (will be modified).

        Returns:
            Truncated messages.
        """
        result = list(messages)

        while (
            len(result) > 1
            and self._counter.estimate_prompt_tokens(result)
            > self._budget.max_prompt_tokens
        ):
            result.pop(0)

        # If still over budget (single huge message), truncate content
        if (
            self._counter.estimate_prompt_tokens(result)
            > self._budget.max_prompt_tokens
        ):
            logger.warning(
                "Message removal insufficient in sliding_window, "
                "applying content truncation"
            )
            result = self._truncate_message_contents(result)

        return result

    def _keep_system_sliding(self, messages: list[dict]) -> list[dict]:
        """Keep system prompts, slide user/assistant messages.

        Preserves all system messages and removes oldest non-system
        messages until context fits. If still over budget after removing
        all but one message, truncates individual message content.

        Args:
            messages: List of messages (will be modified).

        Returns:
            Truncated messages.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # Calculate tokens for system messages
        system_tokens = self._counter.estimate_prompt_tokens(system_msgs)
        available_for_others = self._budget.max_prompt_tokens - system_tokens

        # Remove oldest non-system messages until fits
        while (
            len(other_msgs) > 1
            and self._counter.estimate_prompt_tokens(other_msgs) > available_for_others
        ):
            other_msgs.pop(0)

        result = system_msgs + other_msgs

        # If still over budget, apply aggressive content truncation
        if (
            self._counter.estimate_prompt_tokens(result)
            > self._budget.max_prompt_tokens
        ):
            logger.warning("Message removal insufficient, applying content truncation")
            result = self._truncate_message_contents(result)

        return result

    def _middle_out(self, messages: list[dict]) -> list[dict]:
        """Keep system, first exchange, and recent messages; remove middle.

        Useful for preserving initial context (task setup) and recent
        conversation while removing less relevant middle content.
        Falls back to content truncation if needed.

        Args:
            messages: List of messages (will be modified).

        Returns:
            Truncated messages.
        """
        if len(messages) <= 3:
            result = list(messages)
        else:
            system_msgs = [m for m in messages if m.get("role") == "system"]
            other_msgs = [m for m in messages if m.get("role") != "system"]

            if len(other_msgs) <= 2:
                result = list(messages)
            else:
                # Keep first non-system message and last N messages
                first_msg = [other_msgs[0]]
                remaining = other_msgs[1:]

                # Remove from the beginning of remaining (oldest after first)
                # until we fit within budget
                while (
                    len(remaining) > 1
                    and self._counter.estimate_prompt_tokens(
                        system_msgs + first_msg + remaining
                    )
                    > self._budget.max_prompt_tokens
                ):
                    remaining.pop(0)

                result = system_msgs + first_msg + remaining

        # If still over budget (huge messages), truncate content
        if (
            self._counter.estimate_prompt_tokens(result)
            > self._budget.max_prompt_tokens
        ):
            logger.warning(
                "Message removal insufficient in middle_out, "
                "applying content truncation"
            )
            result = self._truncate_message_contents(result)

        return result

    def _truncate_message_contents(self, messages: list[dict]) -> list[dict]:
        """Truncate individual message contents to fit context budget.

        This is a last resort when removing whole messages isn't enough
        (e.g., when a single message exceeds the entire context budget).

        Strategy:
        1. Calculate how much we're over budget
        2. Find the largest messages and truncate them proportionally
        3. Preserve the last user message as much as possible (most recent query)

        Args:
            messages: List of messages to truncate.

        Returns:
            Messages with truncated content.
        """
        # Use JSON for Pydantic-safe deep copy
        result = json.loads(json.dumps(messages, default=str))
        max_tokens = self._budget.max_prompt_tokens

        # Calculate current usage and how much we need to cut
        current_tokens = self._counter.estimate_prompt_tokens(result)
        if current_tokens <= max_tokens:
            return result

        tokens_to_cut = current_tokens - max_tokens + 100  # Extra buffer

        logger.info(
            f"Content truncation: need to cut ~{tokens_to_cut} tokens "
            f"from {current_tokens} total"
        )

        # Find messages with content, sorted by size (largest first)
        # Skip the last user message if possible (it's the current query)
        messages_with_size = []
        for i, msg in enumerate(result):
            content = msg.get("content") or ""
            if not content:
                continue
            tokens = self._counter.count_tokens(content)
            # Mark if this is the last user message
            is_last_user = msg.get("role") == "user" and all(
                m.get("role") != "user" for m in result[i + 1 :]
            )
            messages_with_size.append((i, tokens, is_last_user))

        # Sort by tokens descending, but keep last user message at end
        messages_with_size.sort(key=lambda x: (x[2], -x[1]))

        # Truncate largest messages first
        tokens_cut = 0
        for idx, msg_tokens, _is_last_user in messages_with_size:
            if tokens_cut >= tokens_to_cut:
                break

            content = result[idx].get("content", "")
            if not content or msg_tokens < 100:
                continue

            # Calculate how much to keep
            # For very large messages, be more aggressive
            if msg_tokens > 10000:
                # Keep at most 10% or 500 tokens
                keep_tokens = min(int(msg_tokens * 0.1), 500)
            elif msg_tokens > 1000:
                # Keep at most 30% or 300 tokens
                keep_tokens = min(int(msg_tokens * 0.3), 300)
            else:
                # Keep at most 50%
                keep_tokens = int(msg_tokens * 0.5)

            # Truncate the content
            truncated_content = self._counter.truncate_to_tokens(content, keep_tokens)
            cut_amount = msg_tokens - self._counter.count_tokens(truncated_content)

            result[idx]["content"] = (
                truncated_content + "\n\n[... content truncated ...]"
            )

            logger.debug(
                f"Truncated message {idx} (role={result[idx].get('role')}): "
                f"{msg_tokens} -> {keep_tokens} tokens"
            )

            tokens_cut += cut_amount

        # Verify we're now under budget, keep truncating if needed
        final_tokens = self._counter.estimate_prompt_tokens(result)
        emergency_iterations = 0
        max_emergency_iterations = len(result) * 2  # Safety limit

        while (
            final_tokens > max_tokens
            and emergency_iterations < max_emergency_iterations
        ):
            emergency_iterations += 1
            logger.warning(
                f"Content truncation incomplete: {final_tokens} > {max_tokens}. "
                f"Emergency truncation iteration {emergency_iterations}."
            )

            # Find the largest message and truncate it aggressively
            largest_idx = -1
            largest_tokens = 0
            for i, msg in enumerate(result):
                content = msg.get("content") or ""
                if content:
                    tokens = self._counter.count_tokens(content)
                    if tokens > largest_tokens:
                        largest_tokens = tokens
                        largest_idx = i

            if largest_idx < 0 or largest_tokens <= 50:
                # No more content to truncate
                logger.error(
                    f"Cannot reduce context further. Remaining: {final_tokens} tokens"
                )
                break

            # Truncate the largest message to 50 tokens
            content = result[largest_idx]["content"]
            result[largest_idx]["content"] = (
                self._counter.truncate_to_tokens(content, 50)
                + "\n[... heavily truncated ...]"
            )

            final_tokens = self._counter.estimate_prompt_tokens(result)
            logger.info(
                f"Emergency truncated message {largest_idx}: "
                f"{largest_tokens} -> ~50 tokens. New total: {final_tokens}"
            )

        return result
