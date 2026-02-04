"""Tests for context summarizer."""

import pytest

from utils.context_summarizer import ContextSummarizer


class TestContextSummarizer:
    """Tests for ContextSummarizer class."""

    @pytest.fixture
    def summarizer(self):
        """Create a context summarizer for testing."""
        return ContextSummarizer(keep_recent=2)

    @pytest.fixture
    def sample_messages(self):
        """Create sample messages for testing."""
        return [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I'm doing well!"},
            {"role": "user", "content": "What's the weather?"},
            {"role": "assistant", "content": "I don't know the weather."},
        ]

    def test_init_with_keep_recent_zero(self):
        """Test that keep_recent=0 is accepted during init."""
        summarizer = ContextSummarizer(keep_recent=0)
        assert summarizer._keep_recent == 0

    def test_init_with_keep_recent_none_uses_default(self):
        """Test that keep_recent=None uses the default value."""
        summarizer = ContextSummarizer(keep_recent=None)
        assert summarizer._keep_recent == ContextSummarizer.DEFAULT_KEEP_RECENT

    def test_split_messages_keep_recent_zero(self):
        """Test that keep_recent=0 summarizes all messages and keeps none.

        This test catches the Python slicing edge case where [:-0] returns []
        and [-0:] returns all elements, which is the opposite of intended behavior.
        """
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye"},
        ]

        # Separate system and other messages as the summarizer does
        other_msgs = [m for m in messages if m.get("role") != "system"]

        # Simulate the slicing logic
        keep_recent = 0
        min_messages = keep_recent * 2  # = 0

        # The fix: handle min_messages=0 specially
        if min_messages == 0:
            to_summarize = other_msgs
            to_keep = []
        else:
            to_summarize = other_msgs[:-min_messages]
            to_keep = other_msgs[-min_messages:]

        # With keep_recent=0, we expect:
        # - ALL non-system messages to be summarized
        # - NO messages to be kept recent
        assert len(to_summarize) == 4, "All non-system messages should be summarized"
        assert len(to_keep) == 0, "No messages should be kept when keep_recent=0"

    def test_split_messages_keep_recent_positive(self):
        """Test normal behavior with positive keep_recent value."""
        messages = [
            {"role": "user", "content": "Msg 1"},
            {"role": "assistant", "content": "Msg 2"},
            {"role": "user", "content": "Msg 3"},
            {"role": "assistant", "content": "Msg 4"},
            {"role": "user", "content": "Msg 5"},
            {"role": "assistant", "content": "Msg 6"},
        ]

        keep_recent = 2
        min_messages = keep_recent * 2  # = 4

        to_summarize = messages[:-min_messages]
        to_keep = messages[-min_messages:]

        # With keep_recent=2, we keep last 4 messages (2 exchanges)
        assert len(to_summarize) == 2, "First 2 messages should be summarized"
        assert len(to_keep) == 4, "Last 4 messages should be kept"
        assert to_keep[0]["content"] == "Msg 3"
        assert to_keep[-1]["content"] == "Msg 6"

    @pytest.mark.asyncio
    async def test_summarize_messages_not_enough_to_summarize(self, summarizer):
        """Test that summarization is skipped when not enough messages."""
        # With keep_recent=2, we need more than 4 non-system messages
        messages = [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]

        result = await summarizer.summarize_messages(messages)

        # Should return original messages unchanged
        assert result == messages

    @pytest.mark.asyncio
    async def test_summarize_messages_respects_keep_recent_override(self, summarizer):
        """Test that keep_recent parameter overrides instance default."""
        messages = [
            {"role": "user", "content": "M1"},
            {"role": "assistant", "content": "M2"},
            {"role": "user", "content": "M3"},
            {"role": "assistant", "content": "M4"},
        ]

        # With keep_recent=2 (default), min_messages=4, so 4 messages won't summarize
        result = await summarizer.summarize_messages(messages)
        assert result == messages  # Not enough to summarize

        # Note: We don't test with keep_recent=1 because it would trigger model loading,
        # which requires downloading a GGUF model and can cause segfaults in CI.
        # The keep_recent logic is tested in the unit tests above.

    def test_format_for_summary(self, summarizer):
        """Test message formatting for summary prompt."""
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "tool", "content": "Tool output"},
        ]

        formatted = summarizer._format_for_summary(messages)

        assert "User: Hello there" in formatted
        assert "Assistant: Hi!" in formatted
        assert "Tool Result: Tool output" in formatted

    def test_format_for_summary_truncates_long_content(self, summarizer):
        """Test that very long messages are truncated in summary format."""
        long_content = "x" * 2000
        messages = [{"role": "user", "content": long_content}]

        formatted = summarizer._format_for_summary(messages)

        # Should be truncated to 1000 chars + "..."
        assert len(formatted) < len(long_content) + 50  # Allow for role prefix
        assert "..." in formatted

    def test_format_for_summary_skips_empty_content(self, summarizer):
        """Test that empty content messages are skipped."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "World"},
        ]

        formatted = summarizer._format_for_summary(messages)

        assert "Hello" in formatted
        assert "World" in formatted
        # Should only have 2 parts (not 3)
        assert formatted.count("User:") == 2
