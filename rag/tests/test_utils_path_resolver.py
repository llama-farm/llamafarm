import os
from pathlib import Path
import tempfile
import pytest

from utils.path_resolver import PathResolver, resolve_paths_in_config


def test_resolve_path_absolute_and_relative(tmp_path):
    # Create a temp file inside tmp_path
    file_path = tmp_path / "file.txt"
    file_path.write_text("x")

    resolver = PathResolver(base_dir=tmp_path)

    # Relative resolution
    rel = resolver.resolve_path("file.txt")
    assert rel == file_path.resolve()

    # Absolute resolution
    abs_resolved = resolver.resolve_path(file_path)
    assert abs_resolved == file_path.resolve()


def test_resolve_path_missing_raises(tmp_path):
    resolver = PathResolver(base_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        resolver.resolve_path("missing.txt")


def test_resolve_config_path_search_locations(tmp_path, monkeypatch):
    # Prepare a config file only in base_dir
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("a: 1")
    resolver = PathResolver(base_dir=tmp_path)
    assert resolver.resolve_config_path("cfg.yaml") == cfg.resolve()

    # If absolute path not found -> raises
    with pytest.raises(FileNotFoundError):
        resolver.resolve_config_path(tmp_path / "nope.yaml")


def test_resolve_data_source_prefers_base_dir_then_cwd(tmp_path, monkeypatch):
    base = tmp_path / "base"
    cwd = tmp_path / "cwd"
    base.mkdir()
    cwd.mkdir()
    (base / "data.txt").write_text("base")
    (cwd / "data.txt").write_text("cwd")

    # Change CWD to cwd
    monkeypatch.chdir(cwd)

    resolver = PathResolver(base_dir=base)
    resolved = resolver.resolve_data_source("data.txt")
    # Should pick base/data.txt first
    assert resolved == (base / "data.txt").resolve()


def test_find_files_by_pattern(tmp_path):
    sub1 = tmp_path / "a"
    sub2 = tmp_path / "b"
    sub1.mkdir()
    sub2.mkdir()
    (sub1 / "x.csv").write_text("x")
    (sub1 / "y.csv").write_text("y")
    (sub2 / "y.csv").write_text("y2")

    resolver = PathResolver(base_dir=tmp_path)
    matches = resolver.find_files_by_pattern("*.csv", [sub1, sub2])
    paths = {p.name for p in matches}
    assert paths == {"x.csv", "y.csv"}


def test_validate_directory_writable(tmp_path):
    resolver = PathResolver(base_dir=tmp_path)
    resolved = resolver.validate_directory_writable(tmp_path)
    assert resolved == tmp_path.resolve()

    # Non-existent directory should raise
    with pytest.raises(FileNotFoundError):
        resolver.validate_directory_writable(tmp_path / "nope")


def test_resolve_paths_in_config_creates_directories(tmp_path):
    resolver = PathResolver(base_dir=tmp_path)
    cfg = {
        "persist_directory": "persist",
        "nested": {"output_directory": "out"},
        "file": "somefile.txt",  # not in path_keys -> unmodified unless exists
    }
    # Create file so it can be resolved
    (tmp_path / "somefile.txt").write_text("x")

    result = resolve_paths_in_config(cfg, resolver)
    assert Path(result["persist_directory"]).exists()
    assert Path(result["nested"]["output_directory"]).exists()
    assert result["file"].endswith("somefile.txt")

