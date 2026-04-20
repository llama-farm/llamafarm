"""Tests for llamafarm_common.model_format.detect_model_format."""

from pathlib import Path
from unittest.mock import patch

from llamafarm_common import model_format as mf


def _write_gguf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * 12)


class TestOrgPrefixStripping:
    """HuggingFace-style org/repo IDs must still hit LLAMAFARM_MODEL_DIR.

    Regression test: resolve_from_model_dir rejects aliases containing "/".
    detect_model_format has to strip the org prefix before the lookup or
    users with a correctly-populated LLAMAFARM_MODEL_DIR will silently
    fall through to a network HF call.
    """

    def test_org_prefix_stripped_before_model_dir_lookup(self, tmp_path, monkeypatch):
        mf.clear_format_cache()
        root = tmp_path / "models"
        root.mkdir()
        # Stored under basename only — matches how operators lay out files.
        _write_gguf(root / "Qwen3-1.7B-GGUF" / "model.Q8_0.gguf")
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(root))

        with patch.object(mf, "_check_local_cache_for_model", return_value=None):
            result = mf.detect_model_format("Qwen/Qwen3-1.7B-GGUF")

        assert result == "gguf"

    def test_bare_name_still_works(self, tmp_path, monkeypatch):
        mf.clear_format_cache()
        root = tmp_path / "models"
        root.mkdir()
        _write_gguf(root / "mission-router-v3" / "model.Q8_0.gguf")
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(root))

        with patch.object(mf, "_check_local_cache_for_model", return_value=None):
            result = mf.detect_model_format("mission-router-v3")

        assert result == "gguf"
