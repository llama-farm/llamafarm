"""Tests for GGUF model support in Universal Runtime.

Note: Tests for llamafarm_common functions (get_gguf_file_path, list_gguf_files,
select_gguf_file, etc.) are in common/tests/test_model_utils.py.

Tests for runtime-specific format detection (detect_model_format) are in
tests/test_model_format.py.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from models.gguf_language_model import GGUFLanguageModel


class TestGGUFLanguageModel:
    """Tests for GGUFLanguageModel class."""

    def test_model_initialization(self):
        """Test GGUF model initialization."""
        model = GGUFLanguageModel("test/model", "cpu")
        assert model.model_id == "test/model"
        assert model.device == "cpu"
        assert model.model_type == "language"
        assert model.supports_streaming is True
        assert model.llama is None  # Not loaded yet
        assert model.n_ctx is None  # Auto-computed on load
        assert model.actual_n_ctx is None  # Not loaded yet

    def test_model_initialization_custom_ctx(self):
        """Test GGUF model initialization with custom context size."""
        model = GGUFLanguageModel("test/model", "cpu", n_ctx=8192)
        assert model.n_ctx == 8192

    def test_format_messages_simple(self):
        """Test formatting of simple chat messages."""
        model = GGUFLanguageModel("test/model", "cpu")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]
        prompt = model.format_messages(messages)
        assert "System: You are helpful" in prompt
        assert "User: Hello" in prompt
        assert "Assistant:" in prompt

    def test_format_messages_conversation(self):
        """Test formatting of multi-turn conversation."""
        model = GGUFLanguageModel("test/model", "cpu")
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4"},
            {"role": "user", "content": "Thanks"},
        ]
        prompt = model.format_messages(messages)
        assert "System: You are helpful" in prompt
        assert "User: What is 2+2?" in prompt
        assert "Assistant: 2+2 equals 4" in prompt
        assert "User: Thanks" in prompt
        assert prompt.endswith("Assistant:")

    @pytest.mark.asyncio
    async def test_load_model_cpu(self, tmp_path):
        """Test loading GGUF model for CPU."""
        # Create mock GGUF file
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf content")

        model = GGUFLanguageModel("test/model", "cpu")

        # Mock the Llama class
        mock_llama = MagicMock()

        with (
            patch(
                "models.gguf_language_model.get_gguf_file_path",
                return_value=str(gguf_file),
            ),
            patch(
                "models.gguf_language_model.get_default_context_size",
                return_value=(2048, []),
            ),
            patch("llamafarm_llama.Llama", return_value=mock_llama),
        ):
            await model.load()
            assert model.llama is not None

    @pytest.mark.asyncio
    async def test_load_model_gpu(self, tmp_path):
        """Test loading GGUF model for GPU/MPS."""
        # Create mock GGUF file
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf content")

        model = GGUFLanguageModel("test/model", "mps")

        # Mock the Llama class
        mock_llama = MagicMock()

        with (
            patch(
                "models.gguf_language_model.get_gguf_file_path",
                return_value=str(gguf_file),
            ),
            patch(
                "models.gguf_language_model.get_default_context_size",
                return_value=(2048, []),
            ),
            patch("llamafarm_llama.Llama", return_value=mock_llama) as mock_llama_cls,
        ):
            await model.load()
            assert model.llama is not None
            # Verify that n_gpu_layers=-1 was passed for GPU
            call_kwargs = mock_llama_cls.call_args[1]
            assert call_kwargs["n_gpu_layers"] == -1

    @pytest.mark.asyncio
    async def test_generate_not_loaded(self):
        """Test generate raises error if model not loaded."""
        model = GGUFLanguageModel("test/model", "cpu")

        with pytest.raises(AssertionError, match="Model not loaded"):
            await model.generate([{"role": "user", "content": "Hello"}])

    @pytest.mark.asyncio
    async def test_generate_with_mock(self, tmp_path):
        """Test text generation with mocked llama-cpp."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf content")

        model = GGUFLanguageModel("test/model", "cpu")

        # Mock llama instance with create_chat_completion method
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.return_value = {
            "choices": [{"message": {"content": "Hello! How can I help?"}}]
        }

        with (
            patch(
                "models.gguf_language_model.get_gguf_file_path",
                return_value=str(gguf_file),
            ),
            patch(
                "models.gguf_language_model.get_default_context_size",
                return_value=(2048, []),
            ),
            patch("llamafarm_llama.Llama", return_value=mock_llama),
        ):
            await model.load()
            result = await model.generate(
                [{"role": "user", "content": "Hi"}], max_tokens=10
            )
            assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_generate_stream_not_loaded(self):
        """Test streaming generate raises error if model not loaded."""
        model = GGUFLanguageModel("test/model", "cpu")

        with pytest.raises(AssertionError, match="Model not loaded"):
            async for _ in model.generate_stream(
                [{"role": "user", "content": "Hello"}]
            ):
                pass

    @pytest.mark.asyncio
    async def test_generate_stream_exception_in_thread(self, caplog):
        """Test generate_stream handles exception from llama-cpp-python gracefully."""
        import logging

        model = GGUFLanguageModel("test/model", "cpu")

        # Mock the llama instance
        mock_llama = Mock()
        model.llama = mock_llama

        # Simulate llama-cpp-python raising an exception during streaming
        def raise_exception(*args, **kwargs):
            raise RuntimeError("Simulated llama-cpp-python error")

        # Mock create_chat_completion to return an iterable that raises
        mock_llama.create_chat_completion.side_effect = raise_exception

        with caplog.at_level(logging.ERROR):
            gen = model.generate_stream(
                [{"role": "user", "content": "Hi"}], max_tokens=10
            )
            with pytest.raises(RuntimeError, match="Simulated llama-cpp-python error"):
                # Exhaust the generator to trigger the exception
                async for _ in gen:
                    pass
            # Verify that an error was logged
            assert any(
                "Simulated llama-cpp-python error" in r.message for r in caplog.records
            )

    @pytest.mark.asyncio
    async def test_logits_processor_is_callable_not_list(self):
        """Test that logits_processor is passed as callable, not wrapped in a list.

        Regression test for issue #627: TypeError 'list' object is not callable.
        The llamafarm_llama library expects logits_processor to be a callable,
        not a list of processors.
        """
        model = GGUFLanguageModel("test/model", "cpu")

        # Mock the llama instance
        mock_llama = Mock()
        model.llama = mock_llama

        # Make create_chat_completion return an empty iterator
        mock_llama.create_chat_completion.return_value = iter([])

        # Call generate_stream with thinking_budget to trigger logits_processor setup
        gen = model.generate_stream(
            [{"role": "user", "content": "Hi"}],
            max_tokens=10,
            thinking_budget=100,
        )

        # Exhaust the generator
        async for _ in gen:
            pass

        # Verify create_chat_completion was called
        mock_llama.create_chat_completion.assert_called_once()

        # Get the logits_processor argument
        call_kwargs = mock_llama.create_chat_completion.call_args[1]
        logits_processor = call_kwargs.get("logits_processor")

        # Assert it's not a list (the bug was wrapping it in a list)
        assert logits_processor is not None, (
            "logits_processor should be set when thinking_budget is provided"
        )
        assert not isinstance(logits_processor, list), (
            "logits_processor must be a callable, not a list. "
            "See issue #627: TypeError 'list' object is not callable"
        )
        assert callable(logits_processor), "logits_processor must be callable"


@pytest.mark.integration
class TestGGUFIntegration:
    """Integration tests for GGUF support (requires actual model download)."""

    @pytest.mark.skip(reason="Requires downloading actual GGUF model - run manually")
    @pytest.mark.asyncio
    async def test_real_gguf_model_load(self):
        """Test loading a real GGUF model from HuggingFace."""
        # This test downloads a small GGUF model - skip by default
        model_id = "QuantFactory/Qwen-1.5-0.5B-GGUF"
        model = GGUFLanguageModel(model_id, "cpu")
        await model.load()
        assert model.llama is not None

    @pytest.mark.skip(reason="Requires downloading actual GGUF model - run manually")
    @pytest.mark.asyncio
    async def test_real_gguf_model_generate(self):
        """Test generation with a real GGUF model."""
        model_id = "QuantFactory/Qwen-1.5-0.5B-GGUF"
        model = GGUFLanguageModel(model_id, "cpu")
        await model.load()

        result = await model.generate(
            [{"role": "user", "content": "Hello"}], max_tokens=10
        )
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.skip(reason="Requires downloading actual GGUF model - run manually")
    @pytest.mark.asyncio
    async def test_real_gguf_model_stream(self):
        """Test streaming with a real GGUF model."""
        model_id = "QuantFactory/Qwen-1.5-0.5B-GGUF"
        model = GGUFLanguageModel(model_id, "cpu")
        await model.load()

        tokens = []
        async for token in model.generate_stream(
            [{"role": "user", "content": "Hello"}], max_tokens=10
        ):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
