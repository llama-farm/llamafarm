"""Tests for GGUF model support in Universal Runtime."""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from models.gguf_language_model import GGUFLanguageModel
from utils.model_format import (
    detect_model_format,
    get_gguf_file_path,
    clear_format_cache,
)


class TestModelFormatDetection:
    """Tests for model format detection utilities."""

    def test_detect_gguf_format_with_mock(self, tmp_path):
        """Test detection of GGUF format with mocked filesystem."""
        # Clear cache before test
        clear_format_cache()

        # Create temporary directory structure
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.gguf").touch()
        (model_dir / "config.json").touch()

        # Mock HfApi to return list of files without making API calls
        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = ["model.gguf", "config.json"]

        with patch("utils.model_format.HfApi", return_value=mock_api):
            with patch(
                "utils.model_format.snapshot_download", return_value=str(model_dir)
            ):
                format_type = detect_model_format("test/model")
                assert format_type == "gguf"

    def test_detect_transformers_format_with_mock(self, tmp_path):
        """Test detection of transformers format with mocked filesystem."""
        # Clear cache before test
        clear_format_cache()

        # Create temporary directory structure (no .gguf files)
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()
        (model_dir / "pytorch_model.bin").touch()

        # Mock HfApi to return list of files without .gguf
        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = ["config.json", "pytorch_model.bin"]

        with patch("utils.model_format.HfApi", return_value=mock_api):
            with patch(
                "utils.model_format.snapshot_download", return_value=str(model_dir)
            ):
                format_type = detect_model_format("test/model")
                assert format_type == "transformers"

    def test_format_detection_caching(self, tmp_path):
        """Test that format detection results are cached."""
        # Clear cache before test
        clear_format_cache()

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model.gguf").touch()

        # Mock HfApi to return list of files
        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = ["model.gguf"]

        with patch("utils.model_format.HfApi", return_value=mock_api) as mock_hf_api:
            with patch(
                "utils.model_format.snapshot_download", return_value=str(model_dir)
            ):
                # First call should trigger API call
                format1 = detect_model_format("test/model")
                assert format1 == "gguf"
                assert mock_hf_api.call_count == 1

                # Second call should use cache (no additional API call)
                format2 = detect_model_format("test/model")
                assert format2 == "gguf"
                assert mock_hf_api.call_count == 1  # Still just 1 call

    def test_get_gguf_file_path_with_mock(self, tmp_path):
        """Test getting GGUF file path with mocked filesystem."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model-q4_k_m.gguf"
        gguf_file.touch()

        # Mock HfApi to return list of GGUF files
        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = ["model-q4_k_m.gguf"]

        with patch("utils.model_format.HfApi", return_value=mock_api):
            with patch(
                "utils.model_format.snapshot_download", return_value=str(model_dir)
            ):
                gguf_path = get_gguf_file_path("test/model")
                assert gguf_path.endswith(".gguf")
                assert os.path.exists(gguf_path)

    def test_get_gguf_file_path_not_found(self, tmp_path):
        """Test error when no GGUF file found."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "config.json").touch()

        # Mock HfApi to return no GGUF files
        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = ["config.json"]

        with patch("utils.model_format.HfApi", return_value=mock_api):
            with patch(
                "utils.model_format.snapshot_download", return_value=str(model_dir)
            ):
                with pytest.raises(
                    FileNotFoundError, match="No GGUF files found in model repository"
                ):
                    get_gguf_file_path("test/model")

    def test_get_gguf_file_path_multiple_files(self, tmp_path):
        """Test handling of multiple GGUF files (should use first one)."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        (model_dir / "model-q4.gguf").touch()
        (model_dir / "model-q8.gguf").touch()

        # Mock HfApi to return multiple GGUF files
        mock_api = MagicMock()
        mock_api.list_repo_files.return_value = ["model-q4.gguf", "model-q8.gguf"]

        with patch("utils.model_format.HfApi", return_value=mock_api):
            with patch(
                "utils.model_format.snapshot_download", return_value=str(model_dir)
            ):
                gguf_path = get_gguf_file_path("test/model")
                assert gguf_path.endswith(".gguf")
                assert os.path.exists(gguf_path)


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

        with patch(
            "models.gguf_language_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch(
                "models.gguf_language_model.get_default_context_size",
                return_value=(2048, []),
            ):
                with patch("models.gguf_language_model.Llama", return_value=mock_llama):
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

        with patch(
            "models.gguf_language_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch(
                "models.gguf_language_model.get_default_context_size",
                return_value=(2048, []),
            ):
                with patch(
                    "models.gguf_language_model.Llama", return_value=mock_llama
                ) as mock_llama_cls:
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
            await model.generate("Hello")

    @pytest.mark.asyncio
    async def test_generate_with_mock(self, tmp_path):
        """Test text generation with mocked llama-cpp."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        gguf_file = model_dir / "model.gguf"
        gguf_file.write_text("mock gguf content")

        model = GGUFLanguageModel("test/model", "cpu")

        # Mock llama instance that returns generation result
        mock_llama = MagicMock()
        mock_llama.return_value = {"choices": [{"text": "Hello! How can I help?"}]}

        with patch(
            "models.gguf_language_model.get_gguf_file_path", return_value=str(gguf_file)
        ):
            with patch(
                "models.gguf_language_model.get_default_context_size",
                return_value=(2048, []),
            ):
                with patch("models.gguf_language_model.Llama", return_value=mock_llama):
                    await model.load()
                    result = await model.generate("Hi", max_tokens=10)
                    assert result == "Hello! How can I help?"

    @pytest.mark.asyncio
    async def test_generate_stream_not_loaded(self):
        """Test streaming generate raises error if model not loaded."""
        model = GGUFLanguageModel("test/model", "cpu")

        with pytest.raises(AssertionError, match="Model not loaded"):
            async for _ in model.generate_stream("Hello"):
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

        mock_llama.side_effect = raise_exception

        with caplog.at_level(logging.ERROR):
            gen = model.generate_stream("Hi", max_tokens=10)
            with pytest.raises(RuntimeError, match="Simulated llama-cpp-python error"):
                # Exhaust the generator to trigger the exception
                async for _ in gen:
                    pass
            # Verify that an error was logged
            assert any(
                "Simulated llama-cpp-python error" in r.message for r in caplog.records
            )


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

        result = await model.generate("Hello", max_tokens=10)
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
        async for token in model.generate_stream("Hello", max_tokens=10):
            tokens.append(token)

        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)
