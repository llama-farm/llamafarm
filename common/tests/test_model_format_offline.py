"""Offline-mode tests for llamafarm_common.model_format.detect_model_format.

The function previously fell through to `huggingface_hub.HfApi.list_repo_files`
when no local cache or LLAMAFARM_MODEL_DIR hit was found, even with
LLAMAFARM_OFFLINE=1. That produced an OfflineModeIsEnabled traceback from
inside huggingface_hub instead of the structured FileNotFoundError that the
rest of llamafarm_common already raises (see list_gguf_files in model_utils).

These tests pin the new offline guard: when LLAMAFARM_OFFLINE=1, the function
must raise FileNotFoundError without constructing HfApi.
"""

from __future__ import annotations

import pytest

from llamafarm_common import model_format, offline_mode


@pytest.fixture(autouse=True)
def _reset_offline_state(monkeypatch):
    monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
    monkeypatch.delenv("LLAMAFARM_MODEL_DIR", raising=False)
    model_format.clear_format_cache()
    offline_mode.reset_for_tests()
    yield


class TestDetectModelFormatOffline:
    def test_offline_raises_filenotfounderror_without_calling_api(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        # Force the local-cache and LLAMAFARM_MODEL_DIR paths to miss so we
        # exercise the network-fallback branch — which must now short-circuit.
        monkeypatch.setattr(
            model_format,
            "_check_local_cache_for_model",
            lambda model_id: None,
        )
        monkeypatch.setattr(
            model_format,
            "resolve_from_model_dir",
            lambda alias: None,
        )

        def _fail(*a, **kw):
            raise AssertionError(
                "HfApi must not be constructed in offline mode — "
                "detect_model_format should short-circuit before reaching the network"
            )

        monkeypatch.setattr(model_format, "HfApi", _fail)

        with pytest.raises(model_format.OfflineModelNotCachedError, match="LLAMAFARM_OFFLINE"):
            model_format.detect_model_format("org/some-model")

    def test_offline_message_points_at_model_dir_fix(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setattr(
            model_format, "_check_local_cache_for_model", lambda model_id: None
        )
        monkeypatch.setattr(
            model_format, "resolve_from_model_dir", lambda alias: None
        )
        monkeypatch.setattr(
            model_format,
            "HfApi",
            lambda: pytest.fail("HfApi must not be constructed"),
        )

        with pytest.raises(FileNotFoundError) as excinfo:
            model_format.detect_model_format("org/some-model")

        msg = str(excinfo.value)
        assert "LLAMAFARM_MODEL_DIR" in msg
        assert "HuggingFace cache" in msg

    def test_offline_short_circuit_does_not_block_local_cache_hits(
        self, monkeypatch
    ):
        """Offline mode + local cache hit → returns without raising."""
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setattr(
            model_format,
            "_check_local_cache_for_model",
            lambda model_id: ["foo.Q4_K_M.gguf"],
        )

        # If the cache-hit path were skipped, this would fail.
        monkeypatch.setattr(
            model_format,
            "HfApi",
            lambda: pytest.fail("HfApi must not be constructed"),
        )

        result = model_format.detect_model_format("org/some-model")
        assert result == "gguf"

    def test_offline_short_circuit_does_not_block_model_dir_hits(self, monkeypatch):
        """Offline mode + LLAMAFARM_MODEL_DIR hit → returns without raising."""
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setattr(
            model_format, "_check_local_cache_for_model", lambda model_id: None
        )

        class FakeResolved:
            weights_path = "/fake/path/foo.gguf"

        monkeypatch.setattr(
            model_format, "resolve_from_model_dir", lambda alias: FakeResolved()
        )
        monkeypatch.setattr(
            model_format,
            "HfApi",
            lambda: pytest.fail("HfApi must not be constructed"),
        )

        result = model_format.detect_model_format("org/some-model")
        assert result == "gguf"

    def test_online_still_calls_api(self, monkeypatch):
        """When offline mode is OFF, the network path is reached as before."""
        monkeypatch.setattr(
            model_format, "_check_local_cache_for_model", lambda model_id: None
        )
        monkeypatch.setattr(
            model_format, "resolve_from_model_dir", lambda alias: None
        )

        class MockApi:
            def list_repo_files(self, repo_id, token=None):
                return ["weights.gguf", "tokenizer.json"]

        monkeypatch.setattr(model_format, "HfApi", MockApi)

        result = model_format.detect_model_format("org/some-model")
        assert result == "gguf"

    def test_gguf_extension_short_circuit_works_in_offline_mode(self, monkeypatch):
        """A bare .gguf filename never needs the network — and shouldn't."""
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setattr(
            model_format,
            "HfApi",
            lambda: pytest.fail("HfApi must not be constructed"),
        )

        assert model_format.detect_model_format("model.gguf") == "gguf"
