"""
Tests for configuration writing functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

# Import the functions we want to test
from loader import load_config, save_config, update_config, ConfigError


class TestConfigWriter:
    """Test configuration writing functionality."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            "version": "v1",
            "name": "sample_config",
            "prompts": [
                {
                    "name": "test_prompt",
                    "prompt": "This is a test prompt for configuration testing.",
                    "description": "A sample prompt for testing purposes"
                }
            ],
            "rag": {
                "parsers": {
                    "csv": {
                        "type": "CustomerSupportCSVParser",
                        "config": {
                            "content_fields": ["question", "answer"],
                            "metadata_fields": ["category", "priority"],
                            "id_field": "id",
                            "combine_content": True
                        },
                        "file_extensions": [".csv"],
                        "mime_types": ["text/csv"]
                    }
                },
                "embedders": {
                    "default": {
                        "type": "OllamaEmbedder",
                        "config": {
                            "model": "nomic-embed-text",
                            "base_url": "http://localhost:11434",
                            "batch_size": 16,
                            "timeout": 30
                        }
                    }
                },
                "vector_stores": {
                    "default": {
                        "type": "ChromaStore",
                        "config": {
                            "collection_name": "test_collection",
                            "persist_directory": "./data/test"
                        }
                    }
                },
                "retrieval_strategies": {
                    "default": {
                        "type": "BasicSimilarityStrategy",
                        "config": {
                            "distance_metric": "cosine"
                        }
                    }
                },
                "defaults": {
                    "parser": "auto",
                    "embedder": "default",
                    "vector_store": "default",
                    "retrieval_strategy": "default"
                }
            },
            "datasets": [
                {
                    "name": "test_dataset",
                    "files": ["test_file.csv"],
                    "parser": "csv"
                }
            ],
            "models": [
                {
                    "provider": "local",
                    "model": "llama3.1:8b"
                }
            ]
        }

    def test_save_config_yaml(self, sample_config):
        """Test saving configuration to YAML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Save configuration
            saved_path = save_config(sample_config, config_path)

            # Verify file was created
            assert saved_path.exists()
            assert saved_path == config_path

            # Load and verify content
            loaded_config = load_config(config_path)
            assert loaded_config["version"] == "v1"
            assert loaded_config["rag"]["parsers"]["csv"]["type"] == "CustomerSupportCSVParser"

    def test_save_config_toml(self, sample_config):
        """Test saving configuration to TOML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.toml"

            try:
                # Save configuration
                saved_path = save_config(sample_config, config_path, "toml")

                # Verify file was created
                assert saved_path.exists()
                assert saved_path == config_path

                # Load and verify content
                loaded_config = load_config(config_path)
                assert loaded_config["version"] == "v1"
                assert loaded_config["rag"]["parsers"]["csv"]["type"] == "CustomerSupportCSVParser"
            except ConfigError as e:
                if "tomli-w is required" in str(e):
                    pytest.skip("tomli-w not installed, skipping TOML save test")
                else:
                    raise

    def test_save_config_json(self, sample_config):
        """Test saving configuration to JSON format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)

            # Save configuration
            saved_path = save_config(sample_config, config_path, "json")

            # Verify file was created with correct name and format
            assert saved_path.exists()
            assert saved_path.name == "llamafarm.json"
            assert saved_path.parent == config_path

            # Load and verify content
            loaded_config = load_config(config_path)
            assert loaded_config["version"] == "v1"
            assert loaded_config["rag"]["parsers"]["csv"]["type"] == "CustomerSupportCSVParser"

    def test_save_config_explicit_format(self, sample_config):
        """Test saving with explicit format specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)

            # Save as YAML with explicit format
            saved_path = save_config(sample_config, config_path, format="yaml")

            # Verify file was created
            assert saved_path.exists()

            # Verify it's actually YAML content
            content = saved_path.read_text()
            assert "version: v1" in content
            assert "type: CustomerSupportCSVParser" in content

    def test_save_config_backup(self, sample_config):
        """Test backup creation when saving over existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "llamafarm.yaml"

            # Create initial file
            config_path.write_text("initial: content\n")

            # Save configuration (should create backup)
            saved_path = save_config(sample_config, config_path, create_backup=True)

            # Verify original file was updated
            assert saved_path.exists()
            loaded_config = load_config(config_path)
            assert loaded_config["version"] == "v1"

            # Verify backup was created
            backup_files = list(Path(temp_dir).glob("llamafarm.*.yaml"))
            assert len(backup_files) == 1

            # Verify backup contains original content
            backup_content = backup_files[0].read_text()
            assert "initial: content" in backup_content

    def test_save_config_no_backup(self, sample_config):
        """Test saving without backup creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "llamafarm.yaml"

            # Create initial file
            config_path.write_text("initial: content\n")

            # Save configuration without backup
            saved_path = save_config(sample_config, config_path, create_backup=False)

            # Verify original file was updated
            assert saved_path.exists()

            # Verify no backup was created
            backup_files = list(Path(temp_dir).glob("llamafarm.*.yaml"))
            assert len(backup_files) == 0

    def test_update_config(self, sample_config):
        """Test updating an existing configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)

            # Save initial configuration
            save_config(sample_config, config_path)

            # Update configuration
            updates = {
                "rag": {
                    "embedders": {
                        "default": {
                            "config": {
                                "batch_size": 32  # Change from 16 to 32
                            }
                        }
                    }
                },
                "models": [
                    {
                        "provider": "local",
                        "model": "llama3.1:8b"
                    },
                    {
                        "provider": "openai",
                        "model": "gpt-4"
                    }
                ]
            }

            updated_path = update_config(config_path, updates)

            # Verify file was updated
            assert updated_path.exists()
            assert updated_path.parent == config_path

            # Load and verify changes
            loaded_config = load_config(config_path)
            assert loaded_config["rag"]["embedders"]["default"]["config"]["batch_size"] == 32
            assert len(loaded_config["models"]) == 2
            assert loaded_config["models"][1]["provider"] == "openai"

            # Verify other values remain unchanged
            assert loaded_config["version"] == "v1"
            assert loaded_config["rag"]["parsers"]["csv"]["type"] == "CustomerSupportCSVParser"

    def test_update_config_deep_merge(self, sample_config):
        """Test deep merging in update_config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)

            # Save initial configuration
            save_config(sample_config, config_path)

            # Update only nested values
            updates = {
                "rag": {
                    "embedders": {
                        "default": {
                            "config": {
                                "batch_size": 64,
                                "new_field": "test_value"
                            }
                        }
                    }
                }
            }

            update_config(config_path, updates)

            # Load and verify deep merge worked
            loaded_config = load_config(config_path)
            embedder_config = loaded_config["rag"]["embedders"]["default"]["config"]

            # Updated values
            assert embedder_config["batch_size"] == 64
            assert embedder_config["new_field"] == "test_value"

            # Preserved values
            assert embedder_config["model"] == "nomic-embed-text"
            assert embedder_config["base_url"] == "http://localhost:11434"
            assert embedder_config["timeout"] == 30

    def test_save_config_validation_error(self):
        """Test validation error when saving invalid configuration."""
        invalid_config = {
            "version": "invalid_version",  # Should be "v1"
            "name": "test_config",
            "prompts": [],
            "datasets": [],
            "models": [],
            "rag": {
                "parsers": {},
                "embedders": {},
                "vector_stores": {},
                "retrieval_strategies": {},
                "defaults": {
                    "parser": "auto",
                    "embedder": "default",
                    "vector_store": "default",
                    "retrieval_strategy": "default"
                }
            }
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid_config.yaml"

            with pytest.raises(ConfigError, match="validation error"):
                save_config(invalid_config, config_path, validate=True)

    def test_save_config_skip_validation(self):
        """Test saving invalid configuration with validation disabled."""
        invalid_config = {
            "version": "invalid_version",
            "incomplete": "config"
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid_config.yaml"

            # Should succeed when validation is disabled
            config_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path = save_config(invalid_config, config_path, validate=False)
            assert saved_path.exists()

    def test_update_nonexistent_file(self):
        """Test updating a file that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yaml"

            with pytest.raises(ConfigError, match="not found"):
                update_config(config_path, {"some": "update"})
