"""
Tests for configuration writing functionality.
"""

import tempfile
from pathlib import Path
from typing import Any

import pytest

# Import the functions we want to test
from config import ConfigError, load_config_dict, save_config, update_config
from config.datamodel import LlamaFarmConfig, Version


class TestConfigWriter:
    """Test configuration writing functionality."""

    @pytest.fixture
    def sample_config(self) -> LlamaFarmConfig:
        """Sample configuration for testing."""
        config: dict[str, Any] = {
            "version": Version.v1,
            "name": "sample_config",
            "namespace": "test",
            "prompts": [
                {
                    "role": "system",
                    "content": "This is a test prompt for configuration testing.",
                }
            ],
            "rag": {
                "databases": [
                    {
                        "name": "test_db",
                        "type": "ChromaStore",
                        "config": {},
                        "embedding_strategies": [
                            {
                                "name": "test_embedding",
                                "type": "OllamaEmbedder",
                                "config": {
                                    "model": "nomic-embed-text",
                                    "base_url": "http://localhost:11434",
                                    "batch_size": 16,
                                    "timeout": 30,
                                    "auto_pull": True,
                                },
                            }
                        ],
                        "retrieval_strategies": [
                            {
                                "name": "test_retrieval",
                                "type": "BasicSimilarityStrategy",
                                "config": {},
                                "default": True,
                            }
                        ],
                    }
                ],
                "data_processing_strategies": [
                    {
                        "name": "default",
                        "description": "Default strategy",
                        "parsers": [
                            {
                                "type": "CSVParser_Pandas",
                                "config": {
                                    "content_fields": ["question"],
                                    "metadata_fields": ["category"],
                                    "id_field": "id",
                                    "combine_content": True,
                                },
                                "file_extensions": [".csv"],
                            }
                        ],
                        "extractors": [],
                    }
                ],
            },
            "datasets": [
                {
                    "name": "test_dataset",
                    "files": ["test_file.csv"],
                    "data_processing_strategy": "auto",
                    "database": "test_db",
                }
            ],
            "runtime": {
                "provider": "openai",
                "model": "llama3.1:8b",
                "api_key": "ollama",
                "base_url": "http://localhost:11434/v1",
                "model_api_parameters": {"temperature": 0.5},
            },
        }
        return LlamaFarmConfig(**config)

    def test_save_config_yaml(self, sample_config):
        """Test saving configuration to YAML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Save configuration
            saved_path, _ = save_config(sample_config, config_path)

            # Verify file was created
            assert saved_path.exists()
            assert saved_path == config_path

            # Load and verify content
            loaded_config = load_config_dict(config_path)
            assert loaded_config["version"] == "v1"

    def test_save_config_toml(self, sample_config):
        """Test saving configuration to TOML format."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.toml"

            try:
                # Save configuration
                saved_path, _ = save_config(sample_config, config_path, "toml")

                # Verify file was created
                assert saved_path.exists()
                assert saved_path == config_path

                # Load and verify content
                loaded_config = load_config_dict(config_path)
                assert loaded_config["version"] == "v1"
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
            saved_path, _ = save_config(sample_config, config_path, "json")

            # Verify file was created with correct name and format
            assert saved_path.exists()
            assert saved_path.name == "llamafarm.json"
            assert saved_path.parent == config_path

            # Load and verify content
            loaded_config = load_config_dict(config_path)
            assert loaded_config["version"] == "v1"

    def test_save_config_explicit_format(self, sample_config):
        """Test saving with explicit format specification."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)

            # Save as YAML with explicit format
            saved_path, _ = save_config(sample_config, config_path, format="yaml")

            # Verify file was created
            assert saved_path.exists()

            # Verify it's actually YAML content
            content = saved_path.read_text()
            assert "version: v1" in content

    def test_save_config_backup(self, sample_config):
        """Test backup creation when saving over existing file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "llamafarm.yaml"

            # Create initial file
            config_path.write_text("initial: content\n")

            # Save configuration (should create backup)
            saved_path, _ = save_config(sample_config, config_path, create_backup=True)

            # Verify original file was updated
            assert saved_path.exists()
            loaded_config = load_config_dict(config_path)
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
            saved_path, _ = save_config(sample_config, config_path, create_backup=False)

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
                    "databases": [
                        {
                            "name": "test_db",
                            "type": "ChromaStore",
                            "config": {},
                            "embedding_strategies": [
                                {
                                    "name": "test_embedding",
                                    "type": "OllamaEmbedder",
                                    "config": {"batch_size": 32},
                                }
                            ],
                            "retrieval_strategies": [
                                {
                                    "name": "test_retrieval",
                                    "type": "BasicSimilarityStrategy",
                                    "config": {},
                                    "default": True,
                                }
                            ],
                        }
                    ],
                    "data_processing_strategies": [
                        {
                            "name": "default",
                            "description": "Default strategy",
                            "parsers": [
                                {
                                    "type": "CSVParser_LlamaIndex",
                                    "config": {
                                        "content_fields": ["question"],
                                        "combine_content": True,
                                        "table_format": "markdown",
                                    },
                                    "file_extensions": [".csv"],
                                }
                            ],
                            "extractors": [],
                        }
                    ],
                },
                "runtime": {
                    "models": [{"name": "test_model", "model": "gpt-4", "provider": "openai"}]
                },
            }

            updated_path, _ = update_config(config_path, updates)

            # Verify file was updated
            assert updated_path.exists()
            assert updated_path.parent == config_path

            # Load and verify changes
            loaded_config = load_config_dict(config_path)
            assert (
                loaded_config["rag"]["databases"][0]["embedding_strategies"][0]["config"][
                    "batch_size"
                ]
                == 32
            )
            assert loaded_config["runtime"]["models"][0]["model"] == "gpt-4"

            # Verify other values remain unchanged
            assert loaded_config["version"] == "v1"

    def test_update_config_deep_merge(self, sample_config):
        """Test deep merging in update_config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir)

            # Save initial configuration
            save_config(sample_config, config_path)

            # Update only nested values
            updates = {
                "rag": {
                    "databases": [
                        {
                            "name": "test_db",
                            "type": "ChromaStore",
                            "config": {},
                            "embedding_strategies": [
                                {
                                    "name": "test_embedding",
                                    "type": "OllamaEmbedder",
                                    "config": {
                                        "model": "nomic-embed-text",
                                        "base_url": "http://localhost:11434",
                                        "batch_size": 64,
                                        "timeout": 45,
                                    },
                                }
                            ],
                            "retrieval_strategies": [
                                {
                                    "name": "test_retrieval",
                                    "type": "BasicSimilarityStrategy",
                                    "config": {},
                                    "default": True,
                                }
                            ],
                        }
                    ],
                    "data_processing_strategies": [
                        {
                            "name": "default",
                            "description": "Default strategy",
                            "parsers": [
                                {
                                    "type": "CSVParser_Pandas",
                                    "config": {},
                                    "file_extensions": [".csv"],
                                }
                            ],
                            "extractors": [],
                        }
                    ],
                },
                "runtime": {
                    "models": [
                        {
                            "provider": "ollama",
                            "name": "test_model",
                            "model": "test_model",
                            "model_api_parameters": {"top_p": 0.9},
                        }
                    ],
                },
            }

            update_config(config_path, updates)

            # Load and verify deep merge worked
            loaded_config = load_config_dict(config_path)
            embedder_config = loaded_config["rag"]["databases"][0]["embedding_strategies"][0][
                "config"
            ]

            # Updated values
            assert embedder_config["batch_size"] == 64
            assert embedder_config["timeout"] == 45

            # Preserved values
            assert embedder_config["model"] == "nomic-embed-text"
            assert embedder_config["base_url"] == "http://localhost:11434"
            # Runtime deep merge
            assert loaded_config["runtime"]["models"][0]["model_api_parameters"]["top_p"] == 0.9

    def test_update_nonexistent_file(self):
        """Test updating a file that doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "nonexistent.yaml"

            with pytest.raises(ConfigError, match="not found"):
                update_config(config_path, {"some": "update"})
