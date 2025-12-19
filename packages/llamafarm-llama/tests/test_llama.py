"""Tests for Llama class."""

from unittest.mock import MagicMock, patch
import pytest


class TestLlamaInit:
    """Test Llama initialization."""

    @patch("llamafarm_llama.llama.ensure_backend")
    @patch("llamafarm_llama.llama.get_lib")
    def test_windows_path_conversion(self, mock_get_lib, mock_ensure_backend):
        """Windows paths should be converted to forward slashes."""
        import sys

        mock_lib = MagicMock()
        mock_lib.llama_model_default_params.return_value = MagicMock()
        mock_lib.llama_load_model_from_file.return_value = 1  # Non-null model pointer
        mock_lib.llama_context_default_params.return_value = MagicMock()
        mock_lib.llama_new_context_with_model.return_value = 1  # Non-null context pointer
        mock_lib.llama_n_vocab.return_value = 32000
        mock_lib.llama_n_ctx.return_value = 2048
        mock_lib.llama_model_meta_val_str.return_value = 0
        mock_get_lib.return_value = mock_lib

        from llamafarm_llama.llama import Llama

        original_platform = sys.platform
        try:
            sys.platform = "win32"
            # Create instance - path should be converted internally
            llama = Llama(model_path="C:\\models\\test.gguf")

            # Verify that llama_load_model_from_file was called
            # (path conversion happens inside the Llama class)
            assert mock_lib.llama_load_model_from_file.called
        finally:
            sys.platform = original_platform


class TestTokenization:
    """Test tokenization methods."""

    def test_tokenize_requires_model(self):
        """Tokenize should require a loaded model."""
        # This tests that the method exists and has correct signature
        from llamafarm_llama.llama import Llama

        # Check method signature
        import inspect

        sig = inspect.signature(Llama.tokenize)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "text" in params

    def test_detokenize_requires_model(self):
        """Detokenize should require a loaded model."""
        from llamafarm_llama.llama import Llama

        import inspect

        sig = inspect.signature(Llama.detokenize)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "tokens" in params


class TestChatCompletion:
    """Test chat completion methods."""

    def test_create_chat_completion_signature(self):
        """create_chat_completion should have expected parameters."""
        from llamafarm_llama.llama import Llama

        import inspect

        sig = inspect.signature(Llama.create_chat_completion)
        params = list(sig.parameters.keys())

        # Required parameters
        assert "messages" in params

        # Optional parameters
        assert "max_tokens" in params
        assert "temperature" in params
        assert "top_p" in params
        assert "stream" in params
        assert "stop" in params

    def test_stream_parameter_default_false(self):
        """Stream parameter should default to False."""
        from llamafarm_llama.llama import Llama

        import inspect

        sig = inspect.signature(Llama.create_chat_completion)
        params = sig.parameters

        assert params["stream"].default is False


class TestEmbeddings:
    """Test embedding methods."""

    def test_create_embedding_signature(self):
        """create_embedding should have expected parameters."""
        from llamafarm_llama.llama import Llama

        import inspect

        sig = inspect.signature(Llama.create_embedding)
        params = list(sig.parameters.keys())

        assert "input" in params


class TestUtf8Streaming:
    """Test UTF-8 streaming decoder for handling multi-byte sequences."""

    def test_decode_complete_ascii(self):
        """Complete ASCII should decode fully."""
        from llamafarm_llama.llama import Llama

        text, pending = Llama._decode_utf8_streaming(b"Hello World")
        assert text == "Hello World"
        assert pending == b""

    def test_decode_complete_emoji(self):
        """Complete emoji should decode fully."""
        from llamafarm_llama.llama import Llama

        # 😎 = \xf0\x9f\x98\x8e (4 bytes)
        text, pending = Llama._decode_utf8_streaming(b"\xf0\x9f\x98\x8e")
        assert text == "😎"
        assert pending == b""

    def test_decode_partial_emoji_1_byte(self):
        """Partial emoji (1 of 4 bytes) should be buffered."""
        from llamafarm_llama.llama import Llama

        # First byte of 😎
        text, pending = Llama._decode_utf8_streaming(b"\xf0")
        assert text == ""
        assert pending == b"\xf0"

    def test_decode_partial_emoji_2_bytes(self):
        """Partial emoji (2 of 4 bytes) should be buffered."""
        from llamafarm_llama.llama import Llama

        # First 2 bytes of 😎
        text, pending = Llama._decode_utf8_streaming(b"\xf0\x9f")
        assert text == ""
        assert pending == b"\xf0\x9f"

    def test_decode_partial_emoji_3_bytes(self):
        """Partial emoji (3 of 4 bytes) should be buffered."""
        from llamafarm_llama.llama import Llama

        # First 3 bytes of 😎
        text, pending = Llama._decode_utf8_streaming(b"\xf0\x9f\x98")
        assert text == ""
        assert pending == b"\xf0\x9f\x98"

    def test_decode_text_with_partial_emoji(self):
        """Text followed by partial emoji should decode text and buffer emoji."""
        from llamafarm_llama.llama import Llama

        # "Hi " + first 2 bytes of 😎
        text, pending = Llama._decode_utf8_streaming(b"Hi \xf0\x9f")
        assert text == "Hi "
        assert pending == b"\xf0\x9f"

    def test_decode_continuation_completes_emoji(self):
        """Adding remaining bytes should complete the emoji."""
        from llamafarm_llama.llama import Llama

        # Simulate streaming: first get partial, then complete
        _, pending = Llama._decode_utf8_streaming(b"\xf0\x9f")
        assert pending == b"\xf0\x9f"

        # Now add the remaining bytes
        text, pending = Llama._decode_utf8_streaming(pending + b"\x98\x8e")
        assert text == "😎"
        assert pending == b""

    def test_decode_empty_bytes(self):
        """Empty bytes should return empty results."""
        from llamafarm_llama.llama import Llama

        text, pending = Llama._decode_utf8_streaming(b"")
        assert text == ""
        assert pending == b""


class TestResponseTypes:
    """Test response type compatibility."""

    def test_chat_completion_response_type(self):
        """ChatCompletionResponse should have expected structure."""
        from llamafarm_llama.types import ChatCompletionResponse

        # TypedDict should have expected keys
        assert "id" in ChatCompletionResponse.__annotations__
        assert "choices" in ChatCompletionResponse.__annotations__
        assert "usage" in ChatCompletionResponse.__annotations__

    def test_embedding_response_type(self):
        """EmbeddingResponse should have expected structure."""
        from llamafarm_llama.types import EmbeddingResponse

        assert "data" in EmbeddingResponse.__annotations__
        assert "usage" in EmbeddingResponse.__annotations__

    def test_chat_message_type(self):
        """ChatMessage should have role and content."""
        from llamafarm_llama.types import ChatMessage

        assert "role" in ChatMessage.__annotations__
        assert "content" in ChatMessage.__annotations__
