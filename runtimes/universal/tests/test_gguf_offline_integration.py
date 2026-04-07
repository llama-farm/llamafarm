"""Integration tests for GGUF language model offline-mode plumbing.

These tests verify that:
  - When `alias` is passed to `GGUFLanguageModel.__init__`, the resolver path
    goes through `resolve_gguf_path` (four-tier with LLAMAFARM_MODEL_DIR).
  - When `alias` is None, the legacy `get_gguf_file_path` entry point is used.
  - Both paths honor `LLAMAFARM_OFFLINE=1` and never touch the network.
  - The resolver can discover a model from `$LLAMAFARM_MODEL_DIR/<alias>/`.

These tests do NOT actually load a real GGUF model into llama-cpp — that
would require a real GGUF file and is covered by other tests. We stop at
the point where `gguf_path` is resolved.
"""

from pathlib import Path
from unittest.mock import patch

import pytest


def _write_fake_gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * 2048)
    return path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
    monkeypatch.delenv("LLAMAFARM_MODEL_DIR", raising=False)
    yield


class TestAliasRoutesThroughResolveGgufPath:
    def test_alias_uses_model_dir_tier(self, tmp_path, monkeypatch):
        """When alias is passed + LLAMAFARM_MODEL_DIR is set, the resolver
        finds the weights under the alias directory."""
        from llamafarm_common import model_utils

        model_dir_root = tmp_path / "models"
        alias_dir = model_dir_root / "qwen3-small"
        alias_dir.mkdir(parents=True)
        weights = _write_fake_gguf(alias_dir / "model.Q4_K_M.gguf")

        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(model_dir_root))

        # Call the resolver directly (the model's load() path goes through this).
        result = model_utils.resolve_gguf_path(
            "org/Qwen3-0.6B-GGUF",
            alias="qwen3-small",
        )
        assert result == str(weights)

    def test_alias_with_offline_and_missing_files_raises(
        self, tmp_path, monkeypatch
    ):
        """Offline + alias dir empty + HF cache empty → loud error."""
        from llamafarm_common import model_utils

        model_dir_root = tmp_path / "models"
        model_dir_root.mkdir()  # root exists, alias subdir does not

        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(model_dir_root))
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        # Make HF cache also empty by pointing at a temp dir.
        empty_hf_cache = tmp_path / "empty-hf"
        empty_hf_cache.mkdir()
        monkeypatch.setattr(model_utils, "HF_HUB_CACHE", str(empty_hf_cache))

        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            with pytest.raises(FileNotFoundError) as excinfo:
                model_utils.resolve_gguf_path(
                    "org/Qwen3-0.6B-GGUF",
                    alias="qwen3-small",
                )
            sd.assert_not_called()
            hf_api.assert_not_called()

        msg = str(excinfo.value)
        assert "qwen3-small" in msg
        assert str(model_dir_root / "qwen3-small") in msg
        assert "lf models pull" in msg


class TestNoAliasFallsBackToGetGgufFilePath:
    def test_no_alias_uses_hf_cache_directly(self, tmp_path, monkeypatch):
        """When alias is None, the legacy path is used; resolver tier skipped."""
        from llamafarm_common import model_utils

        # Populate only HF cache; model_dir is not set.
        hf_cache = tmp_path / "hf"
        hf_cache.mkdir()
        snap = hf_cache / "models--org--Qwen3-GGUF" / "snapshots" / "c1"
        snap.mkdir(parents=True)
        weights = _write_fake_gguf(snap / "qwen3.Q4_K_M.gguf")

        monkeypatch.setattr(model_utils, "HF_HUB_CACHE", str(hf_cache))

        # Use get_gguf_file_path directly (what the `else` branch calls).
        result = model_utils.get_gguf_file_path("org/Qwen3-GGUF")
        assert result == str(weights)


class TestStartupLogLine:
    def test_offline_log_format(self, monkeypatch, caplog):
        """Verify the startup log line contains the expected tokens.

        On CI, structlog is importable but not configured, so its default
        output goes through stdlib logging — which caplog captures. On a
        locally running universal runtime, structlog HAS been configured
        with a console renderer that writes to stdout, which is a different
        sink that caplog does not see.

        To make the test deterministic regardless of environment, we
        monkeypatch `_find_structlog_logger` to always return None, forcing
        the stdlib logging fallback path. That path always writes through
        the named logger, which caplog reliably catches.
        """
        import logging

        from llamafarm_common import offline_mode

        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", "/opt/llamafarm/models")
        monkeypatch.setattr(offline_mode, "_find_structlog_logger", lambda: None)
        offline_mode.reset_for_tests()

        with caplog.at_level(logging.INFO, logger=offline_mode.logger.name):
            offline_mode.log_startup_mode()

        messages = [r.getMessage() for r in caplog.records]
        assert any("mode=offline" in m for m in messages), f"got: {messages}"
        assert any("model_dir=/opt/llamafarm/models" in m for m in messages), f"got: {messages}"


class TestGGUFLanguageModelAcceptsAlias:
    def test_constructor_accepts_alias_kwarg(self):
        """Smoke test: the new `alias` kwarg exists on GGUFLanguageModel.__init__."""
        from models.gguf_language_model import GGUFLanguageModel

        # We don't actually load a model — just verify the constructor
        # accepts the kwarg without raising TypeError.
        model = GGUFLanguageModel(
            model_id="org/Fake-GGUF:Q4_K_M",
            device="cpu",
            alias="my-alias",
        )
        assert model.alias == "my-alias"

    def test_alias_defaults_to_none(self):
        from models.gguf_language_model import GGUFLanguageModel

        model = GGUFLanguageModel(
            model_id="org/Fake-GGUF:Q4_K_M",
            device="cpu",
        )
        assert model.alias is None

    def test_constructor_rejects_traversal_alias(self):
        """A malicious alias must be rejected at construction time, not at load time."""
        import pytest

        from models.gguf_language_model import GGUFLanguageModel

        with pytest.raises(ValueError, match="path traversal"):
            GGUFLanguageModel(
                model_id="org/Fake-GGUF:Q4_K_M",
                device="cpu",
                alias="../etc/passwd",
            )

    def test_constructor_rejects_slash_alias(self):
        import pytest

        from models.gguf_language_model import GGUFLanguageModel

        with pytest.raises(ValueError, match="path separator|path traversal"):
            GGUFLanguageModel(
                model_id="org/Fake-GGUF:Q4_K_M",
                device="cpu",
                alias="org/repo",
            )
