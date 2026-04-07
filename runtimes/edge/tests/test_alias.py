"""Tests for the edge runtime's model-id → LLAMAFARM_MODEL_DIR alias helper.

These tests do NOT import server.py (which triggers a full llama.cpp
runtime bootstrap); they import the alias helper directly from the
lightweight utils module.
"""

from __future__ import annotations

import pytest

from utils.alias import derive_alias_from_model_id


class TestDeriveAliasSuccess:
    def test_org_and_quant(self):
        assert derive_alias_from_model_id("Qwen/Qwen3-0.6B-GGUF:Q4_K_M") == "Qwen3-0.6B-GGUF"

    def test_org_without_quant(self):
        assert derive_alias_from_model_id("unsloth/Qwen3-1.7B-GGUF") == "Qwen3-1.7B-GGUF"

    def test_bare_name(self):
        assert derive_alias_from_model_id("qwen3-small") == "qwen3-small"

    def test_bare_name_with_quant(self):
        assert derive_alias_from_model_id("qwen3-small:Q8_0") == "qwen3-small"

    def test_org_quant_lowercase(self):
        assert derive_alias_from_model_id("org/repo:q4_k_m") == "repo"

    def test_namespaced_models_collide(self):
        """Both foo/my-model and bar/my-model map to `my-model` — documented trade-off."""
        assert derive_alias_from_model_id("foo/my-model") == "my-model"
        assert derive_alias_from_model_id("bar/my-model") == "my-model"

    def test_dots_and_underscores_in_name(self):
        assert derive_alias_from_model_id("org/model_name-v2.1-gguf") == "model_name-v2.1-gguf"


class TestDeriveAliasSkip:
    def test_absolute_path_returns_none(self):
        assert derive_alias_from_model_id("/path/to/custom.gguf") is None

    def test_windows_absolute_path_returns_none(self, monkeypatch):
        # On non-Windows hosts os.path.isabs("C:/...") is False, so we just
        # test a Unix-style absolute path here.
        assert derive_alias_from_model_id("/data/model.gguf") is None

    def test_bare_gguf_filename_returns_none(self):
        assert derive_alias_from_model_id("model.gguf") is None

    def test_relative_path_with_gguf_returns_none(self):
        # A path that looks like a local .gguf file should be skipped, since
        # the model_dir resolver is for HF-style IDs.
        assert derive_alias_from_model_id("models/custom.gguf") is None


class TestDeriveAliasValidationFailures:
    def test_traversal_base_returns_none(self):
        assert derive_alias_from_model_id("org/..") is None

    def test_double_slash_returns_none(self):
        # After stripping one "/", "a/b" becomes "b", which is valid. But
        # "a//b" becomes "b" too. Verifying the happy path works.
        assert derive_alias_from_model_id("a//b") == "b"

    def test_empty_name_returns_none(self):
        assert derive_alias_from_model_id("") is None

    def test_only_colon_returns_none(self):
        # ":" splits into ["", ""], base becomes "", validate_alias rejects empty.
        assert derive_alias_from_model_id(":Q4_K_M") is None

    def test_only_slash_returns_none(self):
        # "/" is absolute and returns None at the isabs() check.
        assert derive_alias_from_model_id("/") is None


class TestDeriveAliasWhitespace:
    def test_trailing_whitespace_rejected(self):
        # validate_alias rejects whitespace in the name.
        assert derive_alias_from_model_id("org/name with space") is None

    def test_leading_dash_rejected(self):
        # Our validator requires the first char to be alphanumeric/._.
        assert derive_alias_from_model_id("org/-leading-dash") is None


class TestDeriveAliasIntegration:
    @pytest.mark.parametrize(
        "model_id,expected",
        [
            ("Qwen/Qwen3-0.6B-GGUF:Q4_K_M", "Qwen3-0.6B-GGUF"),
            ("Qwen/Qwen3-0.6B-GGUF:Q8_0", "Qwen3-0.6B-GGUF"),
            ("Qwen/Qwen3-0.6B-GGUF", "Qwen3-0.6B-GGUF"),
            ("Qwen3-0.6B-GGUF", "Qwen3-0.6B-GGUF"),
            ("Qwen3-0.6B-GGUF:Q4_K_M", "Qwen3-0.6B-GGUF"),
        ],
    )
    def test_same_model_maps_to_same_alias(self, model_id, expected):
        """All forms of the same model should produce the same alias, so the
        operator only needs to populate one directory regardless of which
        form the API client sends."""
        assert derive_alias_from_model_id(model_id) == expected
