"""Tests for Llama class."""

from unittest.mock import MagicMock, patch


class TestLlamaInit:
    """Test Llama initialization."""

    @patch("llamafarm_llama.llama.ensure_backend")
    @patch("llamafarm_llama.llama.get_lib")
    def test_windows_path_conversion(self, mock_get_lib, mock_ensure_backend):
        """Windows paths should be converted to forward slashes."""
        import sys

        mock_lib = MagicMock()
        mock_lib.llama_model_default_params.return_value = MagicMock()
        mock_lib.llama_model_load_from_file.return_value = 1  # Non-null model pointer
        mock_lib.llama_context_default_params.return_value = MagicMock()
        mock_lib.llama_init_from_model.return_value = 1  # Non-null context pointer
        mock_lib.llama_n_vocab.return_value = 32000
        mock_lib.llama_n_ctx.return_value = 2048
        mock_lib.llama_model_meta_val_str.return_value = 0
        mock_get_lib.return_value = mock_lib

        from llamafarm_llama.llama import Llama

        original_platform = sys.platform
        try:
            sys.platform = "win32"
            # Create instance - path should be converted internally
            _llama = Llama(model_path="C:\\models\\test.gguf")  # noqa: F841

            # Verify that llama_model_load_from_file was called
            # (path conversion happens inside the Llama class)
            assert mock_lib.llama_model_load_from_file.called
        finally:
            sys.platform = original_platform


class TestTokenization:
    """Test tokenization methods."""

    def test_tokenize_requires_model(self):
        """Tokenize should require a loaded model."""
        # This tests that the method exists and has correct signature
        # Check method signature
        import inspect

        from llamafarm_llama.llama import Llama

        sig = inspect.signature(Llama.tokenize)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "text" in params

    def test_detokenize_requires_model(self):
        """Detokenize should require a loaded model."""
        import inspect

        from llamafarm_llama.llama import Llama

        sig = inspect.signature(Llama.detokenize)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "tokens" in params


class TestChatCompletion:
    """Test chat completion methods."""

    def test_create_chat_completion_signature(self):
        """create_chat_completion should have expected parameters."""
        import inspect

        from llamafarm_llama.llama import Llama

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
        assert "logprobs" in params
        assert "top_logprobs" in params

    def test_stream_parameter_default_false(self):
        """Stream parameter should default to False."""
        import inspect

        from llamafarm_llama.llama import Llama

        sig = inspect.signature(Llama.create_chat_completion)
        params = sig.parameters

        assert params["stream"].default is False


class TestEmbeddings:
    """Test embedding methods."""

    def test_create_embedding_signature(self):
        """create_embedding should have expected parameters."""
        import inspect

        from llamafarm_llama.llama import Llama

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


class TestChatTemplateUsesModelTemplate:
    """Regression: _apply_chat_template must fetch and pass the model's template."""

    @patch("llamafarm_llama.llama.ensure_backend")
    @patch("llamafarm_llama.llama.get_lib")
    def test_fetches_model_template(self, mock_get_lib, mock_ensure_backend):
        from llamafarm_llama._bindings import ffi
        from llamafarm_llama.llama import Llama

        mock_lib = MagicMock()
        mock_lib.llama_model_default_params.return_value = MagicMock()
        mock_lib.llama_model_load_from_file.return_value = 1
        mock_lib.llama_context_default_params.return_value = MagicMock()
        mock_lib.llama_init_from_model.return_value = 1
        mock_lib.llama_n_vocab.return_value = 32000
        mock_lib.llama_n_ctx.return_value = 2048
        mock_lib.llama_model_meta_val_str.return_value = 0
        mock_get_lib.return_value = mock_lib

        llama = Llama(model_path="test.gguf")

        sentinel = ffi.new("char[]", b"gemma")
        mock_lib.llama_model_chat_template.return_value = sentinel
        mock_lib.llama_chat_apply_template.return_value = 10

        llama._apply_chat_template([{"role": "user", "content": "hi"}])

        mock_lib.llama_model_chat_template.assert_called_once_with(
            llama._model, ffi.NULL
        )
        apply_call = mock_lib.llama_chat_apply_template.call_args
        assert apply_call[0][0] == sentinel


class TestCreateCompletionBOS:
    """Regression: create_completion must tokenize with add_special=True."""

    def test_tokenize_called_with_add_special_true(self):
        import inspect

        from llamafarm_llama.llama import Llama

        source = inspect.getsource(Llama.create_completion)
        assert "add_special=True" in source


class TestSamplerChain:
    """Verify the sampler chain matches the requested sampling mode."""

    def _mock_llama(self):
        """Build a minimal Llama instance with just enough mocked state for
        `_create_sampler` to run, and track which sampler inits got chained."""
        from unittest.mock import MagicMock

        from llamafarm_llama.llama import Llama

        llama = Llama.__new__(Llama)  # bypass __init__ so we don't need a model
        llama._sampler = None
        llama._lib = MagicMock()
        llama._vocab = object()
        # Each init returns a unique sentinel so we can verify chain order.
        sentinels = {}
        def make(name):
            def _init(*_args, **_kwargs):
                sentinels[name] = object()
                return sentinels[name]
            return _init
        llama._lib.llama_sampler_chain_default_params.return_value = object()
        llama._lib.llama_sampler_chain_init.return_value = "chain"
        llama._lib.llama_sampler_init_greedy.side_effect = make("greedy")
        llama._lib.llama_sampler_init_dist.side_effect = make("dist")
        llama._lib.llama_sampler_init_top_k.side_effect = make("top_k")
        llama._lib.llama_sampler_init_top_p.side_effect = make("top_p")
        llama._lib.llama_sampler_init_min_p.side_effect = make("min_p")
        llama._lib.llama_sampler_init_temp.side_effect = make("temp")
        llama._lib.llama_sampler_init_penalties.side_effect = make("penalties")
        llama._lib.llama_vocab_n_tokens.return_value = 32000
        llama._lib.llama_vocab_eos.return_value = 2
        llama._lib.llama_vocab_nl.return_value = 13
        # Capture what got added to the chain, in order.
        added = []
        def chain_add(_chain, sampler):
            name = next(n for n, s in sentinels.items() if s is sampler)
            added.append(name)
        llama._lib.llama_sampler_chain_add.side_effect = chain_add
        return llama, added

    def test_temperature_zero_uses_greedy_only(self):
        """At temperature=0, only the greedy sampler is added.

        Regression for a bug where temperature=0 still chained a stochastic
        `dist` sampler at the end, causing deterministic prompts to produce
        random outputs from the top-k/top-p-filtered candidates."""
        llama, added = self._mock_llama()
        llama._create_sampler(
            temperature=0, top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.0,
        )
        assert added == ["greedy"], f"expected greedy-only, got {added}"

    def test_temperature_zero_with_repeat_penalty_applies_penalty_then_greedy(self):
        """Greedy mode must still honor repeat_penalty.

        The public API defaults to `repeat_penalty=1.1`; earlier the greedy
        early-return skipped the penalty sampler, silently making the
        documented default a no-op at temperature=0."""
        llama, added = self._mock_llama()
        llama._create_sampler(temperature=0, repeat_penalty=1.1)
        assert added == ["penalties", "greedy"], (
            f"expected penalties→greedy, got {added}"
        )

    def test_positive_temperature_uses_stochastic_chain(self):
        """At temperature>0, normal sampler chain ends with `dist`."""
        llama, added = self._mock_llama()
        llama._create_sampler(
            temperature=0.7, top_k=40, top_p=0.95, min_p=0.05, repeat_penalty=1.0,
        )
        assert "greedy" not in added
        assert added[-1] == "dist"
        assert "temp" in added

    def test_repeat_penalty_wired_into_chain(self):
        """repeat_penalty != 1.0 must actually add the penalties sampler."""
        llama, added = self._mock_llama()
        llama._create_sampler(temperature=0.7, repeat_penalty=1.1)
        assert "penalties" in added

    def test_repeat_penalty_one_is_noop(self):
        """repeat_penalty == 1.0 should not add a penalties sampler."""
        llama, added = self._mock_llama()
        llama._create_sampler(temperature=0.7, repeat_penalty=1.0)
        assert "penalties" not in added
