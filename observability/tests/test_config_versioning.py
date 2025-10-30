"""
Unit tests for config versioning.

Tests hash determinism, deduplication, and storage.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from observability.config_versioning import (
    hash_config,
    save_config_snapshot,
    get_config_by_hash,
)


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary data directory for testing."""
    data_dir = tmp_path / "llamafarm_test"
    data_dir.mkdir()

    # Set LF_DATA_DIR for test
    old_env = os.environ.get("LF_DATA_DIR")
    os.environ["LF_DATA_DIR"] = str(data_dir)

    yield data_dir

    # Restore old env
    if old_env:
        os.environ["LF_DATA_DIR"] = old_env
    else:
        del os.environ["LF_DATA_DIR"]


@pytest.fixture
def mock_config():
    """Create a mock config object."""
    config = MagicMock()

    test_config_dict = {
        "version": "v1",
        "name": "test-project",
        "namespace": "default",
        "runtime": {
            "default_model": "fast",
            "models": {
                "fast": {
                    "provider": "ollama",
                    "model": "gemma3:1b"
                }
            }
        }
    }

    # Mock model_dump for hashing (returns dict)
    # Note: model_dump is called with mode='json', exclude_none=True, exclude={...}
    config.model_dump.return_value = test_config_dict

    # Mock model_dump_json for saving (returns JSON string)
    config.model_dump_json.return_value = json.dumps(test_config_dict, indent=2)

    return config


def test_hash_config_deterministic(mock_config):
    """Test that hash_config produces deterministic results."""
    hash1 = hash_config(mock_config)
    hash2 = hash_config(mock_config)

    assert hash1 == hash2
    assert hash1.startswith("sha256_")
    assert len(hash1) == 23  # "sha256_" + 16 chars


def test_hash_config_different_configs():
    """Test that different configs produce different hashes."""
    config1 = MagicMock()
    config1.model_dump.return_value = {"key": "value1"}

    config2 = MagicMock()
    config2.model_dump.return_value = {"key": "value2"}

    hash1 = hash_config(config1)
    hash2 = hash_config(config2)

    assert hash1 != hash2


def test_save_config_snapshot_new(temp_data_dir, mock_config):
    """Test saving a new config snapshot."""
    config_hash = hash_config(mock_config)

    # First save should return True (new)
    is_new = save_config_snapshot(mock_config, config_hash, "default", "test-project")
    assert is_new is True

    # Verify file exists
    config_file = (
        temp_data_dir / "projects" / "default" / "test-project" / "configs" / f"{config_hash}.json"
    )
    assert config_file.exists()

    # Verify content
    with open(config_file) as f:
        saved_config = json.load(f)

    assert saved_config["version"] == "v1"
    assert saved_config["name"] == "test-project"


def test_save_config_snapshot_deduplication(temp_data_dir, mock_config):
    """Test that duplicate configs are not saved again."""
    config_hash = hash_config(mock_config)

    # First save
    is_new1 = save_config_snapshot(mock_config, config_hash, "default", "test-project")
    assert is_new1 is True

    # Second save (same config) should return False (already exists)
    is_new2 = save_config_snapshot(mock_config, config_hash, "default", "test-project")
    assert is_new2 is False


def test_get_config_by_hash_found(temp_data_dir, mock_config):
    """Test retrieving an existing config by hash."""
    config_hash = hash_config(mock_config)

    # Save config
    save_config_snapshot(mock_config, config_hash, "default", "test-project")

    # Retrieve it
    retrieved_config = get_config_by_hash(config_hash, "default", "test-project")

    assert retrieved_config is not None
    assert retrieved_config["version"] == "v1"
    assert retrieved_config["name"] == "test-project"


def test_get_config_by_hash_not_found(temp_data_dir):
    """Test retrieving a non-existent config by hash."""
    result = get_config_by_hash("sha256_nonexistent", "default", "test-project")
    assert result is None


def test_config_versioning_multiple_versions(temp_data_dir):
    """Test saving multiple different config versions."""
    # Create different configs
    config1 = MagicMock()
    config1.model_dump.return_value = {"version": "v1"}
    config1.model_dump_json.return_value = json.dumps({"version": "v1"}, indent=2)

    config2 = MagicMock()
    config2.model_dump.return_value = {"version": "v2"}
    config2.model_dump_json.return_value = json.dumps({"version": "v2"}, indent=2)

    # Save both
    hash1 = hash_config(config1)
    hash2 = hash_config(config2)

    save_config_snapshot(config1, hash1, "default", "test-project")
    save_config_snapshot(config2, hash2, "default", "test-project")

    # Verify both exist
    configs_dir = temp_data_dir / "projects" / "default" / "test-project" / "configs"
    config_files = list(configs_dir.glob("sha256_*.json"))
    assert len(config_files) == 2

    # Verify can retrieve both
    retrieved1 = get_config_by_hash(hash1, "default", "test-project")
    retrieved2 = get_config_by_hash(hash2, "default", "test-project")

    assert retrieved1["version"] == "v1"
    assert retrieved2["version"] == "v2"


def test_config_snapshot_atomic_write(temp_data_dir, mock_config):
    """Test that config snapshot uses atomic writes."""
    config_hash = hash_config(mock_config)

    # Save config
    save_config_snapshot(mock_config, config_hash, "default", "test-project")

    # Verify no temporary files left behind
    configs_dir = temp_data_dir / "projects" / "default" / "test-project" / "configs"
    temp_files = list(configs_dir.glob("*.tmp"))
    assert not temp_files

    # Verify only the final file exists
    config_files = list(configs_dir.glob("sha256_*.json"))
    assert len(config_files) == 1
