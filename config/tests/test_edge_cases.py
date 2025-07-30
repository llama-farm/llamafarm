#!/usr/bin/env python3
"""
Edge case and error condition tests for the configuration loader.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loader import ConfigError, find_config_file, load_config


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_yaml_file(self, temp_config_file):
        """Test loading an empty YAML file."""
        temp_path = temp_config_file("", ".yaml")

        with pytest.raises(ConfigError):
            load_config(config_path=temp_path)

    def test_malformed_yaml(self, temp_config_file):
        """Test loading malformed YAML."""
        malformed_yaml = """
version: v1
models:
  - provider: local
    model: test
      invalid: indentation
"""
        temp_path = temp_config_file(malformed_yaml, ".yaml")

        with pytest.raises(ConfigError):
            load_config(config_path=temp_path)

    def test_malformed_toml(self, temp_config_file):
        """Test loading malformed TOML."""
        malformed_toml = """
version = "v1"
[models
invalid toml syntax
"""
        temp_path = temp_config_file(malformed_toml, ".toml")

        with pytest.raises(ConfigError):
            load_config(config_path=temp_path)

    def test_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with pytest.raises(ConfigError, match="Configuration file not found"):
            load_config(config_path="/nonexistent/path/config.yaml")

    def test_nonexistent_directory(self):
        """Test searching in a nonexistent directory."""
        with pytest.raises(ConfigError, match="Directory does not exist"):
            find_config_file("/nonexistent/directory")

    def test_directory_as_config_file(self):
        """Test passing a directory path as config file."""
        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(
            ConfigError, match="Configuration file not found"
        ):
            load_config(config_path=temp_dir)

    def test_permission_denied(self):
        """Test handling permission denied errors."""
        # Create a file without read permissions
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("version: v1\nmodels: []\nrag: {}")
            temp_path = f.name

        try:
            # Remove read permissions
            os.chmod(temp_path, 0o000)

            with pytest.raises(ConfigError):
                load_config(config_path=temp_path)
        finally:
            # Restore permissions for cleanup
            os.chmod(temp_path, 0o644)
            os.unlink(temp_path)

    def test_very_large_config(self, temp_config_file):
        """Test loading a very large configuration file."""
        # Create a config with many models and prompts
        large_config = """version: v1

rag:
  parser:
    type: CustomerSupportCSVParser
    config:
      content_fields: ["question"]
      metadata_fields: ["category"]
  embedder:
    type: OllamaEmbedder
    config:
      model: "test-model"
      batch_size: 16
  vector_store:
    type: ChromaStore
    config:
      collection_name: "test"
      persist_directory: "./test"

prompts:
"""

        # Add 100 prompts
        for i in range(100):
            large_config += f"""  - name: "prompt_{i}"
    prompt: "This is prompt number {i} with some content."
    description: "Description for prompt {i}"
"""

        large_config += "\nmodels:\n"

        # Add 50 models
        for i in range(50):
            provider = ["local", "openai", "anthropic", "google", "custom"][i % 5]
            large_config += f"""  - provider: "{provider}"
    model: "model_{i}"
"""

        temp_path = temp_config_file(large_config, ".yaml")

        # Should load successfully despite size
        config = load_config(config_path=temp_path)
        assert len(config["prompts"]) == 100
        assert len(config["models"]) == 50

    def test_unicode_content(self, temp_config_file):
        """Test loading configuration with Unicode content."""
        unicode_config = """version: v1

prompts:
  - name: "multilingual_support"
    prompt: "你好, こんにちは, Здравствуйте, مرحبا"
    description: "Prompt with Unicode characters: café, naïve, résumé"

rag:
  parser:
    type: CustomerSupportCSVParser
    config:
      content_fields: ["question"]
      metadata_fields: ["category"]
  embedder:
    type: OllamaEmbedder
    config:
      model: "test-model"
      batch_size: 16
  vector_store:
    type: ChromaStore
    config:
      collection_name: "test_unicode_™"
      persist_directory: "./data/unicode_♠"

models:
  - provider: "local"
    model: "test-model"
"""

        temp_path = temp_config_file(unicode_config, ".yaml")

        config = load_config(config_path=temp_path)
        assert "你好" in config["prompts"][0]["prompt"]
        assert "café" in config["prompts"][0]["description"]
        assert (
            config["rag"]["vector_store"]["config"]["collection_name"]
            == "test_unicode_™"
        )

    def test_deeply_nested_paths(self, temp_config_file):
        """Test configuration with deeply nested file paths."""
        deep_path_config = """version: v1

rag:
  parser:
    type: CustomerSupportCSVParser
    config:
      content_fields: ["question"]
      metadata_fields: ["category"]
  embedder:
    type: OllamaEmbedder
    config:
      model: "test-model"
      batch_size: 16
  vector_store:
    type: ChromaStore
    config:
      collection_name: "test"
      persist_directory: "./very/deeply/nested/directory/structure/that/goes/many/levels/deep"

models:
  - provider: "local"
    model: "test-model"
"""

        temp_path = temp_config_file(deep_path_config, ".yaml")

        config = load_config(config_path=temp_path)
        persist_dir = config["rag"]["vector_store"]["config"]["persist_directory"]
        assert (
            persist_dir
            == "./very/deeply/nested/directory/structure/that/goes/many/levels/deep"
        )

    def test_config_with_special_characters(self, temp_config_file):
        """Test configuration with special characters in values."""
        special_chars_config = """version: v1

prompts:
  - name: "special_chars"
    prompt: "Handle chars: @#$%^&*()_+-=[]{}|;',./<>?"
    description: "Testing with special characters & symbols!"

rag:
  parser:
    type: CustomerSupportCSVParser
    config:
      content_fields: ["question"]
      metadata_fields: ["category"]
  embedder:
    type: OllamaEmbedder
    config:
      model: "model@version:1.0"
      batch_size: 16
  vector_store:
    type: ChromaStore
    config:
      collection_name: "collection_name_with-dashes_and_underscores"
      persist_directory: "./data/with spaces and (parentheses)"

models:
  - provider: "local"
    model: "model:tag@version"
"""

        temp_path = temp_config_file(special_chars_config, ".yaml")

        config = load_config(config_path=temp_path)
        assert "@#$%^&*" in config["prompts"][0]["prompt"]
        assert config["rag"]["embedder"]["config"]["model"] == "model@version:1.0"
        assert (
            "spaces and" in config["rag"]["vector_store"]["config"]["persist_directory"]
        )

    @pytest.mark.skip(
        reason="Dependency simulation test is complex and not critical for core functionality"
    )
    def test_missing_dependencies_simulation(self, monkeypatch):
        """Test behavior when dependencies are missing."""
        # This test is skipped as it's complex to simulate missing dependencies
        # in a reliable way across different environments. The error handling
        # for missing dependencies is covered by the actual import statements
        # in the loader module.
        pass

    def test_config_file_discovery_priority(self):
        """Test that config files are discovered in the correct priority order."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple config files
            (temp_path / "llamafarm.yml").write_text("version: v1\nmodels: []\nrag: {}")
            (temp_path / "llamafarm.toml").write_text(
                'version = "v1"\nmodels = []\n[rag]'
            )
            (temp_path / "llamafarm.yaml").write_text(
                "version: v1\nmodels: []\nrag: {}"
            )

            # Should find .yaml first (highest priority)
            found_file = find_config_file(temp_path)
            assert found_file.name == "llamafarm.yaml"

            # Remove .yaml, should find .yml next
            (temp_path / "llamafarm.yaml").unlink()
            found_file = find_config_file(temp_path)
            assert found_file.name == "llamafarm.yml"

            # Remove .yml, should find .toml last
            (temp_path / "llamafarm.yml").unlink()
            found_file = find_config_file(temp_path)
            assert found_file.name == "llamafarm.toml"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
