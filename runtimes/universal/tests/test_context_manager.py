"""Tests for context management utilities."""

import pytest

from utils.context_manager import (
    ContextBudget,
    ContextManager,
    ContextUsage,
    TruncationStrategy,
)
from utils.history_compressor import HistoryCompressor
from utils.token_counter import TokenCounter


class MockLlama:
    """Mock Llama instance for testing."""

    def tokenize(
        self, text: str, add_special: bool = True, parse_special: bool = False
    ):
        """Mock tokenize that returns approximately 1 token per 4 characters."""
        if not text:
            return []
        # Simulate tokenization: roughly 4 chars per token
        return list(range(len(text) // 4 + 1))

    def detokenize(self, tokens: list[int]) -> str:
        """Mock detokenize that returns placeholder text."""
        return "x" * (len(tokens) * 4)


class TestTokenCounter:
    """Tests for TokenCounter class."""

    def test_count_tokens_empty_string(self):
        """Test counting tokens in empty string returns 0."""
        counter = TokenCounter(MockLlama())
        assert counter.count_tokens("") == 0

    def test_count_tokens_short_string(self):
        """Test counting tokens in short string."""
        counter = TokenCounter(MockLlama())
        # "Hello" is 5 chars, should be ~2 tokens
        tokens = counter.count_tokens("Hello")
        assert tokens > 0

    def test_count_message_tokens_includes_overhead(self):
        """Test that message token count includes role overhead."""
        counter = TokenCounter(MockLlama())
        message = {"role": "user", "content": "Hello"}
        content_tokens = counter.count_tokens("Hello")
        message_tokens = counter.count_message_tokens(message)
        # Message tokens should include MESSAGE_OVERHEAD
        assert message_tokens == content_tokens + TokenCounter.MESSAGE_OVERHEAD

    def test_count_message_tokens_handles_none_content(self):
        """Test counting tokens handles None content."""
        counter = TokenCounter(MockLlama())
        message = {"role": "assistant", "content": None}
        tokens = counter.count_message_tokens(message)
        assert tokens == TokenCounter.MESSAGE_OVERHEAD

    def test_estimate_prompt_tokens_includes_overhead(self):
        """Test that prompt estimation includes template overhead."""
        counter = TokenCounter(MockLlama())
        messages = [{"role": "user", "content": "Hello world"}]
        base_tokens = counter.count_messages_tokens(messages)
        estimated = counter.estimate_prompt_tokens(messages)
        # Should include 10% overhead
        assert estimated == int(base_tokens * TokenCounter.TEMPLATE_OVERHEAD_FACTOR)

    def test_truncate_to_tokens(self):
        """Test text truncation to token limit."""
        counter = TokenCounter(MockLlama())
        long_text = "a" * 100  # ~25 tokens
        truncated = counter.truncate_to_tokens(long_text, 5)
        # Truncated text should be shorter
        assert len(truncated) < len(long_text)


class TestContextBudget:
    """Tests for ContextBudget class."""

    def test_from_context_size_default_values(self):
        """Test budget creation with default values."""
        budget = ContextBudget.from_context_size(4096)
        assert budget.total_context == 4096
        assert budget.reserved_completion == 512
        assert budget.safety_margin > 0
        assert budget.max_prompt_tokens < 4096

    def test_from_context_size_custom_completion(self):
        """Test budget creation with custom completion tokens."""
        budget = ContextBudget.from_context_size(8192, max_completion_tokens=1024)
        assert budget.reserved_completion == 1024
        assert budget.max_prompt_tokens == 8192 - 1024 - budget.safety_margin

    def test_safety_margin_calculation(self):
        """Test safety margin is 5% by default."""
        budget = ContextBudget.from_context_size(10000)
        expected_margin = int(10000 * 0.05)
        assert budget.safety_margin == expected_margin


class TestContextManager:
    """Tests for ContextManager class."""

    @pytest.fixture
    def manager(self):
        """Create a context manager with mock token counter."""
        counter = TokenCounter(MockLlama())
        budget = ContextBudget.from_context_size(1000)
        return ContextManager(counter, budget)

    def test_validate_messages_returns_usage(self, manager):
        """Test that validate_messages returns ContextUsage."""
        messages = [{"role": "user", "content": "Hello"}]
        usage = manager.validate_messages(messages)
        assert isinstance(usage, ContextUsage)
        assert usage.total_context == 1000
        assert usage.prompt_tokens > 0
        assert usage.truncated is False

    def test_needs_truncation_false_for_small_messages(self, manager):
        """Test needs_truncation returns False for small messages."""
        messages = [{"role": "user", "content": "Hi"}]
        assert manager.needs_truncation(messages) is False

    def test_needs_truncation_true_for_large_messages(self, manager):
        """Test needs_truncation returns True for large messages."""
        # Create messages that exceed budget
        large_content = "x" * 10000  # Way more than 1000 token budget
        messages = [{"role": "user", "content": large_content}]
        assert manager.needs_truncation(messages) is True

    def test_truncate_if_needed_no_truncation_when_fits(self, manager):
        """Test that no truncation occurs when messages fit."""
        messages = [{"role": "user", "content": "Hello"}]
        result, usage = manager.truncate_if_needed(messages)
        assert len(result) == len(messages)
        assert usage.truncated is False
        assert usage.truncated_messages == 0

    def test_sliding_window_removes_oldest(self, manager):
        """Test sliding window strategy removes oldest messages."""
        # Create many messages that exceed budget (1000 tokens = ~4000 chars with mock)
        messages = [
            {"role": "user", "content": "x" * 2000},  # ~500 tokens, will be removed
            {"role": "user", "content": "y" * 2000},  # ~500 tokens, will be removed
            {"role": "user", "content": "z" * 400},  # ~100 tokens, should be kept
        ]
        result, usage = manager.truncate_if_needed(
            messages, TruncationStrategy.SLIDING_WINDOW
        )
        assert len(result) < len(messages)
        assert usage.truncated is True

    def test_keep_system_sliding_preserves_system(self, manager):
        """Test keep_system_sliding preserves system messages."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "x" * 500},
            {"role": "assistant", "content": "y" * 500},
            {"role": "user", "content": "z" * 100},
        ]
        result, usage = manager.truncate_if_needed(
            messages, TruncationStrategy.KEEP_SYSTEM_SLIDING
        )
        # System message should be preserved
        system_msgs = [m for m in result if m.get("role") == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful"

    def test_middle_out_keeps_first_and_recent(self, manager):
        """Test middle_out keeps first user message and recent messages."""
        messages = [
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "x" * 300},
            {"role": "user", "content": "y" * 300},
            {"role": "assistant", "content": "z" * 300},
            {"role": "user", "content": "Recent question"},
        ]
        result, usage = manager.truncate_if_needed(
            messages, TruncationStrategy.MIDDLE_OUT
        )
        # Should keep system, first user, and recent
        assert any(m.get("content") == "First question" for m in result)
        assert any(m.get("content") == "Recent question" for m in result)

    def test_truncate_does_not_modify_original(self, manager):
        """Test that truncation doesn't modify original messages."""
        messages = [
            {"role": "user", "content": "x" * 1000},
            {"role": "user", "content": "y" * 100},
        ]
        original_len = len(messages)
        original_first = messages[0]["content"]
        manager.truncate_if_needed(messages)
        assert len(messages) == original_len
        assert messages[0]["content"] == original_first

    def test_content_truncation_for_huge_message(self, manager):
        """Test that huge individual messages get content-truncated."""
        # Create a single message that exceeds entire context budget
        # With 1000 token budget and ~4 chars/token, 50000 chars = ~12500 tokens
        huge_content = "x" * 50000
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": huge_content},
        ]
        result, usage = manager.truncate_if_needed(
            messages, TruncationStrategy.KEEP_SYSTEM_SLIDING
        )

        # Result should fit within budget
        assert usage.truncated is True
        # The huge message should be truncated
        user_msg = [m for m in result if m.get("role") == "user"][0]
        assert len(user_msg["content"]) < len(huge_content)
        # Should contain truncation marker (either regular or emergency)
        assert "truncated" in user_msg["content"]

    def test_sliding_window_with_huge_message(self, manager):
        """Test sliding window handles single huge message."""
        # Single message exceeding budget
        huge_content = "x" * 50000
        messages = [{"role": "user", "content": huge_content}]
        result, usage = manager.truncate_if_needed(
            messages, TruncationStrategy.SLIDING_WINDOW
        )

        # Should have truncated the content
        assert usage.truncated is True
        assert len(result[0]["content"]) < len(huge_content)


class TestHistoryCompressor:
    """Tests for HistoryCompressor class."""

    def test_compress_empty_messages(self):
        """Test compression of empty message list."""
        compressor = HistoryCompressor()
        result = compressor.compress([])
        assert result == []

    def test_compress_preserves_recent_messages(self):
        """Test that recent messages are not compressed."""
        compressor = HistoryCompressor()
        messages = [
            {"role": "user", "content": "Old message"},
            {"role": "assistant", "content": "Old response"},
            {"role": "user", "content": "Recent 1"},
            {"role": "assistant", "content": "Recent 2"},
            {"role": "user", "content": "Recent 3"},
            {"role": "assistant", "content": "Recent 4"},
        ]
        result = compressor.compress(messages, preserve_recent=4)
        # Last 4 should be unchanged
        assert result[-4:] == messages[-4:]

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        compressor = HistoryCompressor()
        content = "Hello\n\n\n\nWorld"
        normalized = compressor._normalize_whitespace(content)
        assert "\n\n\n" not in normalized
        assert normalized == "Hello\n\nWorld"

    def test_compress_tool_results_with_counter(self):
        """Test tool result compression with token counter."""
        counter = TokenCounter(MockLlama())
        compressor = HistoryCompressor(counter)
        messages = [
            {"role": "tool", "content": "x" * 5000},  # Very long tool result
            {"role": "user", "content": "Short"},
            {"role": "assistant", "content": "Also short"},
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Recent too"},
        ]
        result = compressor.compress(messages, preserve_recent=4)
        # First message (old tool result) should be truncated
        first_tool = result[0]
        assert first_tool["role"] == "tool"
        assert "[... result truncated ...]" in first_tool["content"]

    def test_compress_code_blocks(self):
        """Test code block compression in assistant messages."""
        compressor = HistoryCompressor()
        long_code = "\n".join([f"line {i}" for i in range(50)])
        messages = [
            {
                "role": "assistant",
                "content": f"Here's the code:\n```python\n{long_code}\n```",
            },
            {"role": "user", "content": "Recent 1"},
            {"role": "assistant", "content": "Recent 2"},
            {"role": "user", "content": "Recent 3"},
            {"role": "assistant", "content": "Recent 4"},
        ]
        result = compressor.compress(messages, preserve_recent=4)
        # First message should have compressed code block
        first = result[0]
        # Check that lines were compressed (note: count includes trailing newline)
        assert "lines total" in first["content"]
        assert "line 0" in first["content"]  # First lines preserved

    def test_remove_repetitions(self):
        """Test removal of duplicate content."""
        compressor = HistoryCompressor()
        repeated_content = (
            "This is a long message that repeats multiple times in the conversation."
        )
        messages = [
            {"role": "user", "content": repeated_content},
            {"role": "assistant", "content": "Response 1"},
            {"role": "user", "content": repeated_content},  # Duplicate
            {"role": "assistant", "content": "Response 2"},
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Recent"},
            {"role": "user", "content": "Recent"},
            {"role": "assistant", "content": "Recent"},
        ]
        result = compressor.compress(messages, preserve_recent=4)
        # Should have removed one duplicate
        user_contents = [m["content"] for m in result if m.get("role") == "user"]
        # Count how many times the repeated content appears (excluding recent)
        repeat_count = sum(1 for c in user_contents if c == repeated_content)
        assert repeat_count <= 1


class TestTruncationStrategy:
    """Tests for TruncationStrategy enum."""

    def test_strategy_values(self):
        """Test that all strategies have correct values."""
        assert TruncationStrategy.SLIDING_WINDOW.value == "sliding_window"
        assert TruncationStrategy.KEEP_SYSTEM_SLIDING.value == "keep_system"
        assert TruncationStrategy.MIDDLE_OUT.value == "middle_out"
        assert TruncationStrategy.SUMMARIZE.value == "summarize"

    def test_strategy_from_string(self):
        """Test creating strategy from string value."""
        strategy = TruncationStrategy("keep_system")
        assert strategy == TruncationStrategy.KEEP_SYSTEM_SLIDING

    def test_invalid_strategy_raises(self):
        """Test that invalid strategy string raises ValueError."""
        with pytest.raises(ValueError):
            TruncationStrategy("invalid_strategy")
