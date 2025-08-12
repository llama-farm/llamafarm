#!/usr/bin/env python3
"""
Comprehensive test suite for the LlamaFarm configuration loader.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigError, LlamaFarmConfig, find_config_file, load_config_dict


class TestConfigLoader:
    """Test class for configuration loader functionality."""

    @pytest.fixture
    def test_data_dir(self):
        """Return the path to test data directory."""
        return Path(__file__).parent

    def test_load_yaml_sample_config(self, test_data_dir: Path):
        """Test loading the comprehensive YAML sample configuration."""
        config_path = test_data_dir / "sample_config.yaml"
        config = load_config_dict(config_path=config_path)

        # Verify basic structure
        assert config["version"] == "v1"
        assert "rag" in config
        assert "models" in config
        assert "prompts" in config

        # Verify prompts
        assert len(config["prompts"]) == 3
        prompt_names = [p["name"] for p in config["prompts"]]
        assert "customer_support" in prompt_names
        assert "technical_documentation" in prompt_names
        assert "code_review" in prompt_names

        # Verify RAG configuration
        rag = config["rag"]
        # New strict schema: strategies array
        assert isinstance(rag["strategies"], list) and len(rag["strategies"]) >= 1
        strat = rag["strategies"][0]
        assert strat["components"]["parser"]["type"] == "CSVParser"
        assert strat["components"]["embedder"]["type"] == "OllamaEmbedder"
        assert strat["components"]["vector_store"]["type"] == "ChromaStore"

        # Verify parser config
        parser_config = strat["components"]["parser"]["config"]
        assert "question" in parser_config["content_fields"]
        assert "answer" in parser_config["content_fields"]
        assert "category" in parser_config["metadata_fields"]
        assert "timestamp" in parser_config["metadata_fields"]

        # Verify embedder config
        embedder_config = strat["components"]["embedder"]["config"]
        assert embedder_config["model"] == "mxbai-embed-large"
        assert embedder_config["batch_size"] == 32

        # Verify vector store config
        vector_config = strat["components"]["vector_store"]["config"]
        assert vector_config["collection_name"] == "customer_support_knowledge_base"
        assert vector_config["persist_directory"] == "./data/vector_store/chroma"

        # Verify defaults section
        # Defaults removed in strict schema version; strategies govern components

        # Verify models
        assert len(config["models"]) == 8
        providers = [m["provider"] for m in config["models"]]
        assert "local" in providers
        assert "openai" in providers
        assert "anthropic" in providers
        assert "google" in providers
        assert "custom" in providers

        # Verify specific models
        model_names = [m["model"] for m in config["models"]]
        assert "llama3.1:8b" in model_names
        assert "gpt-4" in model_names
        assert "claude-3-sonnet-20240229" in model_names
        assert "gemini-pro" in model_names

    def test_load_toml_sample_config(self, test_data_dir):
        """Test loading the comprehensive TOML sample configuration."""
        config_path = test_data_dir / "sample_config.toml"
        config = load_config_dict(config_path=config_path)

        # Should have same structure as YAML version
        assert config["version"] == "v1"
        assert len(config["prompts"]) == 3
        assert len(config["models"]) == 8

        # Verify TOML-specific parsing worked correctly
        strat = config["rag"]["strategies"][0]
        assert strat["components"]["embedder"]["config"]["batch_size"] == 32
        assert isinstance(strat["components"]["parser"]["config"]["content_fields"], list)
        assert isinstance(config["models"], list)

    def test_load_minimal_config(self, test_data_dir):
        """Test loading minimal valid configuration."""
        config_path = test_data_dir / "minimal_config.yaml"
        config = load_config_dict(config_path=config_path)

        assert config["version"] == "v1"
        assert len(config["models"]) == 1
        assert config["models"][0]["provider"] == "local"
        assert config["models"][0]["model"] == "llama3.1:8b"

        # Prompts should be present since it's a required field
        assert "prompts" in config
        assert isinstance(config["prompts"], list)
        assert len(config["prompts"]) == 1
        assert config["prompts"][0]["name"] == "minimal_prompt"

        # RAG should be properly configured
        strat = config["rag"]["strategies"][0]
        assert strat["components"]["parser"]["config"]["content_fields"] == ["question"]
        assert strat["components"]["embedder"]["config"]["model"] == "nomic-embed-text"

    def test_validation_with_invalid_config(self, test_data_dir):
        """Test that validation catches invalid configurations."""
        config_path = test_data_dir / "invalid_config.yaml"

        with pytest.raises(ConfigError):
            load_config_dict(config_path=config_path, validate=True)

    def test_load_without_validation(self, test_data_dir):
        """Test loading invalid config without validation."""
        config_path = test_data_dir / "invalid_config.yaml"

        # Should load without error when validation is disabled
        config = load_config_dict(config_path=config_path, validate=False)
        assert config["version"] == "v2"  # Invalid version but loaded anyway

    def test_find_config_file(self, test_data_dir):
        """Test configuration file discovery."""
        # Create a temporary directory with proper llamafarm config files
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy our sample config to the expected filename
            sample_config = test_data_dir / "sample_config.yaml"
            if sample_config.exists():
                shutil.copy(sample_config, temp_path / "llamafarm.yaml")

            # Should find llamafarm.yaml
            found_file = find_config_file(temp_path)
            assert found_file is not None
            assert found_file.name == "llamafarm.yaml"

    def test_missing_config_file(self):
        """Test behavior when no config file is found."""
        with tempfile.TemporaryDirectory() as temp_dir, pytest.raises(
            ConfigError, match="No configuration file found"
        ):
            load_config_dict(directory=temp_dir)

    def test_unsupported_file_format(self):
        """Test error handling for unsupported file formats."""
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write('<?xml version="1.0"?><config><version>v1</version></config>')
            temp_path = f.name

        try:
            with pytest.raises(ConfigError, match="Unsupported file format"):
                load_config_dict(config_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_missing_required_fields(self):
        """Test validation of missing required fields."""
        incomplete_config = """version: v1
models:
  - provider: local
    model: test
# Missing rag section
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(incomplete_config)
            temp_path = f.name

        try:
            with pytest.raises(ConfigError):
                load_config_dict(config_path=temp_path)
        finally:
            os.unlink(temp_path)

    def test_all_provider_types(self, test_data_dir):
        """Test that all provider types from schema are covered in sample config."""
        config_path = test_data_dir / "sample_config.yaml"
        config = load_config_dict(config_path=config_path)

        providers = {m["provider"] for m in config["models"]}
        expected_providers = {"openai", "anthropic", "google", "local", "custom"}

        assert providers == expected_providers, (
            f"Missing providers: {expected_providers - providers}"
        )

    def test_type_safety(self, test_data_dir):
        """Test that loaded config matches expected types."""
        config_path = test_data_dir / "sample_config.yaml"
        config: LlamaFarmConfig = load_config_dict(config_path=config_path)

        # These should pass type checking
        version: str = config["version"]
        models: list = config["models"]
        rag: dict = config["rag"]

        assert isinstance(version, str)
        assert isinstance(models, list)
        assert isinstance(rag, dict)

        # Test nested structure types
        for model in models:
            assert isinstance(model["provider"], str)
            assert isinstance(model["model"], str)

        # Test RAG structure (strict schema)
        strat = rag["strategies"][0]
        assert isinstance(strat["components"]["parser"]["config"]["content_fields"], list)
        assert isinstance(strat["components"]["embedder"]["config"]["batch_size"], int)

    def test_config_with_no_prompts(self, test_data_dir):
        """Test configuration loading with minimal prompts section."""
        config_path = test_data_dir / "minimal_config.yaml"
        config = load_config_dict(config_path=config_path)

        # Should load successfully with minimal prompts (required field)
        assert "prompts" in config
        assert isinstance(config["prompts"], list)
        # The minimal config has at least one prompt
        assert len(config["prompts"]) >= 1

    def test_directory_vs_file_loading(self, test_data_dir):
        """Test loading by directory vs explicit file path."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Copy sample config to proper filename for directory discovery
            sample_config = test_data_dir / "sample_config.yaml"
            if sample_config.exists():
                shutil.copy(sample_config, temp_path / "llamafarm.yaml")

            # Load by directory (should find llamafarm.yaml)
            config1 = load_config_dict(directory=temp_path)

            # Load by explicit file path
            config2 = load_config_dict(config_path=test_data_dir / "sample_config.yaml")

            # Both should succeed and have same version
            assert config1["version"] == "v1"
            assert config2["version"] == "v1"


def test_integration_usage():
    """Test how the config module would be used by other modules in the project."""
    # This simulates how other modules would import and use the config
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Test package-style import
    from config import load_config
    from config.datamodel import LlamaFarmConfig

    test_dir = Path(__file__).parent
    config_path = test_dir / "sample_config.yaml"

    # Load config as other modules would
    config: LlamaFarmConfig = load_config_dict(config_path=config_path)

    # Verify typical usage patterns
    assert config["version"] == "v1"

    # Test accessing RAG configuration (common use case)
    strat = config["rag"]["strategies"][0]
    parser_type = strat["components"]["parser"]["type"]
    embedder_model = strat["components"]["embedder"]["config"]["model"]
    collection_name = strat["components"]["vector_store"]["config"]["collection_name"]

    assert parser_type == "CSVParser"
    assert embedder_model == "mxbai-embed-large"
    assert collection_name == "customer_support_knowledge_base"

    # Test accessing models (common use case)
    local_models = [m for m in config["models"] if m["provider"] == "local"]
    openai_models = [m for m in config["models"] if m["provider"] == "openai"]

    assert len(local_models) >= 1
    assert len(openai_models) >= 1

    # Test accessing prompts (common use case)
    if config.get("prompts"):
        customer_support_prompt = next(
            (p for p in config["prompts"] if p["name"] == "customer_support"), None
        )
        assert customer_support_prompt is not None
        assert "assistant" in customer_support_prompt["prompt"].lower()


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
