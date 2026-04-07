"""Tests for llamafarm_common.offline_mode bootstrap behaviors."""

import logging
import os

import pytest

from llamafarm_common import offline_mode


@pytest.fixture(autouse=True)
def _reset_startup_flag():
    """Ensure each test sees a fresh startup-log flag."""
    offline_mode.reset_for_tests()
    yield
    offline_mode.reset_for_tests()


# ---------------------------------------------------------------------------
# _as_bool / is_offline
# ---------------------------------------------------------------------------


class TestAsBoolTruthyDetection:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON"])
    def test_truthy_values(self, value):
        assert offline_mode._as_bool(value) is True

    @pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "maybe", " "])
    def test_falsy_values(self, value):
        assert offline_mode._as_bool(value) is False

    def test_whitespace_is_stripped(self):
        assert offline_mode._as_bool("  true  ") is True


class TestIsOffline:
    def test_unset(self, monkeypatch):
        monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
        assert offline_mode.is_offline() is False

    def test_set_truthy(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        assert offline_mode.is_offline() is True

    def test_set_false(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "0")
        assert offline_mode.is_offline() is False


class TestModelDir:
    def test_unset(self, monkeypatch):
        monkeypatch.delenv("LLAMAFARM_MODEL_DIR", raising=False)
        assert offline_mode.model_dir() is None

    def test_empty_string_is_none(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", "")
        assert offline_mode.model_dir() is None

    def test_whitespace_only_is_none(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", "   ")
        assert offline_mode.model_dir() is None

    def test_set_returns_value(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", "/opt/llamafarm/models")
        assert offline_mode.model_dir() == "/opt/llamafarm/models"


# ---------------------------------------------------------------------------
# propagate_hf_env
# ---------------------------------------------------------------------------


class TestPropagateHfEnv:
    def test_online_leaves_env_alone(self, monkeypatch):
        monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        offline_mode.propagate_hf_env()
        assert os.environ.get("HF_HUB_OFFLINE") is None
        assert os.environ.get("TRANSFORMERS_OFFLINE") is None

    def test_offline_sets_both(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
        offline_mode.propagate_hf_env()
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    def test_offline_overrides_falsy_existing(self, monkeypatch, caplog):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setenv("HF_HUB_OFFLINE", "0")
        monkeypatch.setenv("TRANSFORMERS_OFFLINE", "false")
        with caplog.at_level(logging.WARNING, logger=offline_mode.logger.name):
            offline_mode.propagate_hf_env()
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 2, "expected a warning for each overridden var"
        assert any("HF_HUB_OFFLINE" in r.message for r in warnings)
        assert any("TRANSFORMERS_OFFLINE" in r.message for r in warnings)

    def test_offline_leaves_truthy_existing_unchanged(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setenv("HF_HUB_OFFLINE", "1")
        offline_mode.propagate_hf_env()
        assert os.environ["HF_HUB_OFFLINE"] == "1"

    def test_idempotent_on_repeated_calls(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        offline_mode.propagate_hf_env()
        offline_mode.propagate_hf_env()
        offline_mode.propagate_hf_env()
        assert os.environ["HF_HUB_OFFLINE"] == "1"


# ---------------------------------------------------------------------------
# log_startup_mode
# ---------------------------------------------------------------------------


class TestLogStartupMode:
    def test_logs_once(self, monkeypatch, caplog):
        monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
        monkeypatch.delenv("LLAMAFARM_MODEL_DIR", raising=False)
        with caplog.at_level(logging.INFO, logger=offline_mode.logger.name):
            offline_mode.log_startup_mode()
            offline_mode.log_startup_mode()
            offline_mode.log_startup_mode()
        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        assert len(info_records) == 1, f"expected exactly 1 log, got {len(info_records)}"

    def test_online_message_contains_mode_online(self, monkeypatch, caplog):
        monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
        with caplog.at_level(logging.INFO, logger=offline_mode.logger.name):
            offline_mode.log_startup_mode()
        messages = [r.getMessage() for r in caplog.records]
        assert any("mode=online" in m for m in messages)

    def test_offline_message_contains_mode_offline_and_model_dir(self, monkeypatch, caplog):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", "/opt/llamafarm/models")
        with caplog.at_level(logging.INFO, logger=offline_mode.logger.name):
            offline_mode.log_startup_mode()
        messages = [r.getMessage() for r in caplog.records]
        assert any("mode=offline" in m for m in messages)
        assert any("model_dir=/opt/llamafarm/models" in m for m in messages)


# ---------------------------------------------------------------------------
# raise_offline_error / raise_offline_binary_error
# ---------------------------------------------------------------------------


class TestRaiseOfflineError:
    def test_raises_filenotfound_with_required_fields(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            offline_mode.raise_offline_error(
                alias="qwen3-1.7b",
                tried_paths=["/opt/llamafarm/models/qwen3-1.7b/", "/root/.cache/hf"],
                fix_command="run 'lf models pull qwen3-1.7b' on a host with internet",
            )
        msg = str(excinfo.value)
        assert "qwen3-1.7b" in msg
        assert "/opt/llamafarm/models/qwen3-1.7b/" in msg
        assert "/root/.cache/hf" in msg
        assert "lf models pull qwen3-1.7b" in msg
        assert "offline mode" in msg

    def test_includes_extra_note_when_provided(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            offline_mode.raise_offline_error(
                alias="x",
                tried_paths=["/p1"],
                fix_command="do thing",
                extra="consider also checking Y",
            )
        assert "consider also checking Y" in str(excinfo.value)


class TestRaiseOfflineBinaryError:
    def test_raises_filenotfound_for_binary(self):
        with pytest.raises(FileNotFoundError) as excinfo:
            offline_mode.raise_offline_binary_error(
                target="linux/arm64",
                tried_paths=["/cache/b7694/libllama.so"],
            )
        msg = str(excinfo.value)
        assert "linux/arm64" in msg
        assert "/cache/b7694/libllama.so" in msg
        assert "lf runtime binary pull" in msg
