"""Offline-mode tests for llamafarm_common.model_utils.

These tests exercise the new `resolve_gguf_path` / `resolve_mmproj_path`
four-tier resolver and the offline guards added to `list_gguf_files`,
`get_gguf_file_path`, and `get_mmproj_file_path`.

Strategy:
  - Create a fake HuggingFace cache directory under a temp dir and point
    `HF_HUB_CACHE` at it by monkeypatching the module-level constant.
  - Create a fake `LLAMAFARM_MODEL_DIR` under another temp subdir.
  - Mock `huggingface_hub.HfApi.list_repo_files` and
    `huggingface_hub.snapshot_download` with mocks that raise if called
    when we expect zero network activity.
"""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from llamafarm_common import model_utils, offline_mode


def _write_gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x00" * 1024)
    return path


@pytest.fixture(autouse=True)
def _reset_offline_state(monkeypatch):
    """Ensure each test starts with a clean offline-mode environment."""
    monkeypatch.delenv("LLAMAFARM_OFFLINE", raising=False)
    monkeypatch.delenv("LLAMAFARM_MODEL_DIR", raising=False)
    offline_mode.reset_for_tests()
    yield


@pytest.fixture
def fake_hf_cache(tmp_path, monkeypatch):
    """Create a fake HF_HUB_CACHE directory and point model_utils at it."""
    cache = tmp_path / "hf-hub"
    cache.mkdir()
    monkeypatch.setattr(model_utils, "HF_HUB_CACHE", str(cache))
    return cache


def _layout_hf_repo(cache: Path, repo_id: str, commit: str, files: list[str]) -> dict:
    """Create a fake HF cache directory with the given files inside a snapshot.

    Returns a dict mapping filename → absolute path of the written file.
    """
    repo_slug = "models--" + repo_id.replace("/", "--")
    snap = cache / repo_slug / "snapshots" / commit
    snap.mkdir(parents=True)
    out = {}
    for name in files:
        out[name] = _write_gguf(snap / name)
    return out


# ---------------------------------------------------------------------------
# list_gguf_files offline guard
# ---------------------------------------------------------------------------


class TestListGgufFilesOffline:
    def test_online_calls_api(self, monkeypatch):
        mock_api = type("MockApi", (), {"list_repo_files": lambda self, repo_id, token=None: ["a.gguf", "b.gguf"]})
        monkeypatch.setattr(model_utils, "HfApi", mock_api)
        result = model_utils.list_gguf_files("org/repo")
        assert result == ["a.gguf", "b.gguf"]

    def test_offline_raises_without_calling_api(self, monkeypatch):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        def _fail(*a, **kw):
            raise AssertionError("HfApi must not be constructed in offline mode")

        monkeypatch.setattr(model_utils, "HfApi", _fail)
        with pytest.raises(FileNotFoundError, match="offline mode"):
            model_utils.list_gguf_files("org/repo")


# ---------------------------------------------------------------------------
# get_gguf_file_path offline + cache
# ---------------------------------------------------------------------------


class TestGetGgufFilePathOffline:
    def test_offline_cache_hit_succeeds_without_network(
        self, fake_hf_cache, monkeypatch
    ):
        _layout_hf_repo(
            fake_hf_cache,
            "org/Test-GGUF",
            "commit1",
            ["test.Q4_K_M.gguf", "test.Q8_0.gguf"],
        )
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        # Assert no network helpers are called.
        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            result = model_utils.get_gguf_file_path("org/Test-GGUF:Q4_K_M")
            sd.assert_not_called()
            hf_api.assert_not_called()

        assert "test.Q4_K_M.gguf" in result

    def test_offline_cache_miss_raises_with_alias_and_command(
        self, fake_hf_cache, monkeypatch
    ):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            with pytest.raises(FileNotFoundError) as excinfo:
                model_utils.get_gguf_file_path("org/Missing-GGUF")
            sd.assert_not_called()
            hf_api.assert_not_called()

        msg = str(excinfo.value)
        assert "offline mode" in msg
        assert "org/Missing-GGUF" in msg
        assert "lf models pull" in msg


# ---------------------------------------------------------------------------
# resolve_gguf_path four-tier resolution
# ---------------------------------------------------------------------------


class TestResolveGgufPathAbsolutePath:
    def test_absolute_path_passthrough(self, tmp_path, fake_hf_cache):
        path = tmp_path / "custom.gguf"
        _write_gguf(path)
        result = model_utils.resolve_gguf_path(str(path), alias="custom")
        assert result == str(path)

    def test_absolute_path_missing_raises(self, fake_hf_cache):
        with pytest.raises(FileNotFoundError, match="Absolute model path"):
            model_utils.resolve_gguf_path("/nonexistent/model.gguf", alias="x")


class TestResolveGgufPathModelDir:
    def test_alias_dir_wins_over_hf_cache(self, tmp_path, fake_hf_cache, monkeypatch):
        # Populate both tiers.
        _layout_hf_repo(
            fake_hf_cache,
            "org/Test-GGUF",
            "commit1",
            ["test.Q4_K_M.gguf"],
        )
        model_dir_root = tmp_path / "models"
        alias_dir = model_dir_root / "my-alias"
        alias_dir.mkdir(parents=True)
        md_weights = _write_gguf(alias_dir / "preferred.Q4_K_M.gguf")

        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(model_dir_root))

        result = model_utils.resolve_gguf_path("org/Test-GGUF", alias="my-alias")
        assert result == str(md_weights)

    def test_alias_dir_miss_falls_through_to_hf_cache(
        self, tmp_path, fake_hf_cache, monkeypatch
    ):
        files = _layout_hf_repo(
            fake_hf_cache,
            "org/Test-GGUF",
            "commit1",
            ["test.Q4_K_M.gguf"],
        )
        model_dir_root = tmp_path / "models"
        model_dir_root.mkdir()
        # Alias subdir is missing.
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(model_dir_root))

        result = model_utils.resolve_gguf_path("org/Test-GGUF", alias="missing-alias")
        assert result == str(files["test.Q4_K_M.gguf"])


class TestResolveGgufPathOfflineStrict:
    def test_offline_alias_dir_hit(self, tmp_path, fake_hf_cache, monkeypatch):
        model_dir_root = tmp_path / "models"
        alias_dir = model_dir_root / "qwen3-1.7b"
        alias_dir.mkdir(parents=True)
        weights = _write_gguf(alias_dir / "model.Q4_K_M.gguf")

        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(model_dir_root))
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            result = model_utils.resolve_gguf_path(
                "org/Qwen3-1.7B-GGUF", alias="qwen3-1.7b"
            )
            sd.assert_not_called()
            hf_api.assert_not_called()

        assert result == str(weights)

    def test_offline_complete_miss_lists_both_paths(
        self, tmp_path, fake_hf_cache, monkeypatch
    ):
        model_dir_root = tmp_path / "models"
        model_dir_root.mkdir()  # root exists but alias dir does not
        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(model_dir_root))
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            with pytest.raises(FileNotFoundError) as excinfo:
                model_utils.resolve_gguf_path(
                    "org/Qwen3-1.7B-GGUF", alias="qwen3-1.7b"
                )
            sd.assert_not_called()
            hf_api.assert_not_called()

        msg = str(excinfo.value)
        assert "qwen3-1.7b" in msg  # alias name
        assert str(model_dir_root / "qwen3-1.7b") in msg  # model dir path
        assert "models--org--Qwen3-1.7B-GGUF" in msg  # HF cache path
        assert "lf models pull" in msg

    def test_offline_hf_cache_hit(self, tmp_path, fake_hf_cache, monkeypatch):
        files = _layout_hf_repo(
            fake_hf_cache,
            "org/Test-GGUF",
            "c1",
            ["test.Q4_K_M.gguf"],
        )
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")

        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            result = model_utils.resolve_gguf_path(
                "org/Test-GGUF", alias="test"
            )
            sd.assert_not_called()
            hf_api.assert_not_called()

        assert result == str(files["test.Q4_K_M.gguf"])


# ---------------------------------------------------------------------------
# resolve_mmproj_path
# ---------------------------------------------------------------------------


class TestResolveMmprojPath:
    def test_alias_dir_mmproj(self, tmp_path, fake_hf_cache, monkeypatch):
        model_dir_root = tmp_path / "models"
        alias_dir = model_dir_root / "omni"
        alias_dir.mkdir(parents=True)
        _write_gguf(alias_dir / "model.Q4_K_M.gguf")
        mmproj = _write_gguf(alias_dir / "mmproj-f16.gguf")

        monkeypatch.setenv("LLAMAFARM_MODEL_DIR", str(model_dir_root))
        result = model_utils.resolve_mmproj_path("org/Omni-GGUF", alias="omni")
        assert result == str(mmproj)

    def test_offline_no_mmproj_returns_none(
        self, tmp_path, fake_hf_cache, monkeypatch
    ):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            result = model_utils.resolve_mmproj_path(
                "org/Text-Only-GGUF", alias="text-only"
            )
            sd.assert_not_called()
            hf_api.assert_not_called()

        assert result is None


# ---------------------------------------------------------------------------
# get_mmproj_file_path offline guard (legacy entry point)
# ---------------------------------------------------------------------------


class TestGetMmprojFilePathOffline:
    def test_offline_no_cache_returns_none_without_network(
        self, fake_hf_cache, monkeypatch
    ):
        monkeypatch.setenv("LLAMAFARM_OFFLINE", "1")
        with patch.object(model_utils, "snapshot_download") as sd, patch.object(
            model_utils, "HfApi"
        ) as hf_api:
            result = model_utils.get_mmproj_file_path("org/Text-Only-GGUF")
            sd.assert_not_called()
            hf_api.assert_not_called()

        assert result is None
