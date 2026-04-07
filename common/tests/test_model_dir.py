"""Tests for llamafarm_common.model_dir alias-directory resolver."""

import logging
from pathlib import Path

import pytest

from llamafarm_common import model_dir as model_dir_mod
from llamafarm_common.model_dir import ModelDirResult, resolve_from_model_dir


def _write_gguf(path: Path, magic: bool = True) -> Path:
    """Write a fake GGUF file with (or without) valid magic bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if magic:
        path.write_bytes(b"GGUF" + b"\x00" * 12)
    else:
        path.write_bytes(b"NOTG" + b"\x00" * 12)
    return path


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(root))
    return root


class TestUnsetReturnsNone:
    def test_env_unset(self, monkeypatch):
        monkeypatch.delenv("LLAMAFARM_MODEL_DIR", raising=False)
        assert resolve_from_model_dir("qwen3") is None

    def test_env_empty_string(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", "")
        assert resolve_from_model_dir("qwen3") is None


class TestNonexistentRootWarns:
    def test_missing_root_warns_and_returns_none(self, monkeypatch, caplog, tmp_path):
        bogus = tmp_path / "does-not-exist"
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(bogus))
        with caplog.at_level(logging.WARNING, logger=model_dir_mod.logger.name):
            result = resolve_from_model_dir("anything")
        assert result is None
        assert any("does not exist" in r.message for r in caplog.records)


class TestAliasDirMissing:
    def test_alias_subdir_absent(self, fake_root):
        assert resolve_from_model_dir("nonexistent-alias") is None

    def test_alias_is_file_not_dir(self, fake_root):
        (fake_root / "weird-alias").write_bytes(b"oops")
        assert resolve_from_model_dir("weird-alias") is None


class TestEmptyAliasDir:
    def test_empty_dir_returns_none(self, fake_root):
        (fake_root / "qwen3").mkdir()
        assert resolve_from_model_dir("qwen3") is None

    def test_only_non_gguf_files(self, fake_root):
        alias = fake_root / "qwen3"
        alias.mkdir()
        (alias / "readme.md").write_text("hello")
        (alias / "manifest.json").write_text("{}")
        assert resolve_from_model_dir("qwen3") is None

    def test_only_mmproj_no_weights(self, fake_root):
        alias = fake_root / "qwen3"
        alias.mkdir()
        _write_gguf(alias / "mmproj-qwen-f16.gguf")
        assert resolve_from_model_dir("qwen3") is None


class TestGGUFMagicValidation:
    def test_bad_magic_rejected(self, fake_root, caplog):
        alias = fake_root / "qwen3"
        alias.mkdir()
        _write_gguf(alias / "fake.gguf", magic=False)
        with caplog.at_level(logging.WARNING, logger=model_dir_mod.logger.name):
            result = resolve_from_model_dir("qwen3")
        assert result is None
        assert any("missing GGUF magic" in r.message for r in caplog.records)

    def test_good_magic_accepted(self, fake_root):
        alias = fake_root / "qwen3"
        alias.mkdir()
        path = _write_gguf(alias / "qwen3.Q4_K_M.gguf")
        result = resolve_from_model_dir("qwen3")
        assert result is not None
        assert result.weights_path == str(path)


class TestCanonicalFilename:
    def test_canonical_name_resolves(self, fake_root):
        alias = fake_root / "qwen3-1.7b"
        alias.mkdir()
        path = _write_gguf(alias / "model.Q4_K_M.gguf")
        result = resolve_from_model_dir("qwen3-1.7b")
        assert result is not None
        assert result.weights_path == str(path)
        assert result.alias == "qwen3-1.7b"
        assert result.mmproj_path is None


class TestPreservedHFFilename:
    def test_hf_filename_resolves(self, fake_root):
        alias = fake_root / "qwen3-1.7b"
        alias.mkdir()
        path = _write_gguf(alias / "Qwen3-1.7B-Q4_K_M.gguf")
        result = resolve_from_model_dir("qwen3-1.7b")
        assert result is not None
        assert result.weights_path == str(path)


class TestMultipleQuantsPreferenceOrder:
    def test_q4_k_m_preferred_over_q8_0(self, fake_root):
        alias = fake_root / "qwen3"
        alias.mkdir()
        _write_gguf(alias / "qwen3.Q8_0.gguf")
        q4 = _write_gguf(alias / "qwen3.Q4_K_M.gguf")
        result = resolve_from_model_dir("qwen3")
        assert result is not None
        assert result.weights_path == str(q4)

    def test_full_preference_order(self, fake_root):
        alias = fake_root / "qwen3"
        alias.mkdir()
        # Add several in reverse preference order.
        _write_gguf(alias / "qwen3.F16.gguf")
        _write_gguf(alias / "qwen3.Q2_K.gguf")
        _write_gguf(alias / "qwen3.Q8_0.gguf")
        q5 = _write_gguf(alias / "qwen3.Q5_K_M.gguf")
        q4 = _write_gguf(alias / "qwen3.Q4_K_M.gguf")
        result = resolve_from_model_dir("qwen3")
        assert result is not None
        # Q4_K_M is first in the preference order.
        assert result.weights_path == str(q4)
        # Tiebreakers are deterministic — same input always returns q4.
        assert result.weights_path != str(q5)


class TestMmprojDetection:
    def test_mmproj_separated_from_weights(self, fake_root):
        alias = fake_root / "omni"
        alias.mkdir()
        weights = _write_gguf(alias / "omni.Q4_K_M.gguf")
        mmproj = _write_gguf(alias / "mmproj-omni-f16.gguf")
        result = resolve_from_model_dir("omni")
        assert result is not None
        assert result.weights_path == str(weights)
        assert result.mmproj_path == str(mmproj)

    def test_mmproj_only_none(self, fake_root):
        alias = fake_root / "omni"
        alias.mkdir()
        _write_gguf(alias / "mmproj-omni-f16.gguf")
        assert resolve_from_model_dir("omni") is None

    def test_canonical_mmproj_name(self, fake_root):
        alias = fake_root / "omni"
        alias.mkdir()
        _write_gguf(alias / "model.Q4_K_M.gguf")
        mmproj = _write_gguf(alias / "mmproj.f16.gguf")
        result = resolve_from_model_dir("omni")
        assert result is not None
        assert result.mmproj_path == str(mmproj)

    def test_mmproj_prefers_f16_over_f32(self, fake_root):
        alias = fake_root / "omni"
        alias.mkdir()
        _write_gguf(alias / "model.Q4_K_M.gguf")
        _write_gguf(alias / "mmproj-omni-f32.gguf")
        f16 = _write_gguf(alias / "mmproj-omni-f16.gguf")
        result = resolve_from_model_dir("omni")
        assert result is not None
        assert result.mmproj_path == str(f16)


class TestResultShape:
    def test_result_is_immutable_dataclass(self, fake_root):
        alias = fake_root / "qwen3"
        alias.mkdir()
        _write_gguf(alias / "qwen3.Q4_K_M.gguf")
        result = resolve_from_model_dir("qwen3")
        assert isinstance(result, ModelDirResult)
        assert result.alias == "qwen3"
        assert result.alias_dir == str(alias)
        # frozen=True: attempting to mutate raises
        with pytest.raises(Exception):
            result.weights_path = "/other/path"  # type: ignore[misc]
