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
        assert "prompts" in config

        # Verify prompts match strict schema (list of prompt objects with name and messages)
        if "prompts" in config:
            assert isinstance(config["prompts"], list)
            if config["prompts"]:
                first_prompt = config["prompts"][0]
                assert "name" in first_prompt
                assert "messages" in first_prompt

        # Verify RAG configuration
        rag = config["rag"]
        # New schema: databases and data_processing_strategies arrays
        assert isinstance(rag["databases"], list) and len(rag["databases"]) >= 1
        assert (
            isinstance(rag["data_processing_strategies"], list)
            and len(rag["data_processing_strategies"]) >= 1
        )

        # Verify database configuration
        db = rag["databases"][0]
        assert db["type"] == "ChromaStore"
        assert isinstance(db["embedding_strategies"], list) and len(db["embedding_strategies"]) >= 1
        assert db["embedding_strategies"][0]["type"] == "OllamaEmbedder"

        # Verify data processing strategy
        strat = rag["data_processing_strategies"][0]
        assert isinstance(strat["parsers"], list) and len(strat["parsers"]) >= 1
        assert strat["parsers"][0]["type"] in [
            "CSVParser_LlamaIndex",
            "CSVParser_Pandas",
            "CSVParser_Python",
        ]

        # Verify parser config
        parser_config = strat["parsers"][0]["config"]
        assert "question" in parser_config["content_fields"]
        assert "answer" in parser_config["content_fields"]
        assert "category" in parser_config["metadata_fields"]
        assert "timestamp" in parser_config["metadata_fields"]

        # Verify embedder config
        embedder_config = db["embedding_strategies"][0]["config"]
        assert embedder_config["model"] == "mxbai-embed-large"
        assert embedder_config["batch_size"] == 32

        # Verify vector store config
        vector_config = db["config"]
        assert vector_config["collection_name"] == "customer_support_knowledge_base"
        assert vector_config["persist_directory"] == "./data/vector_store/chroma"

        # Verify defaults section
        # Defaults removed in strict schema version; strategies govern components

        # Models list is optional under the current schema; skip strict checks

    def test_load_toml_sample_config(self, test_data_dir):
        """Test loading the comprehensive TOML sample configuration."""
        config_path = test_data_dir / "sample_config.toml"
        # Relax validation for TOML sample since prompts may use legacy fields
        config = load_config_dict(config_path=config_path, validate=False)

        # Should load and have valid version
        assert config["version"] == "v1"

        # Verify TOML-specific parsing worked correctly
        db = config["rag"]["databases"][0]
        assert db["embedding_strategies"][0]["config"]["batch_size"] == 32
        strat = config["rag"]["data_processing_strategies"][0]
        assert isinstance(strat["parsers"][0]["config"]["content_fields"], list)

    def test_load_minimal_config(self, test_data_dir):
        """Test loading minimal valid configuration."""
        config_path = test_data_dir / "minimal_config.yaml"
        config = load_config_dict(config_path=config_path)

        assert config["version"] == "v1"

        # Prompts optional list; when present, objects have name and messages
        assert "prompts" in config
        assert isinstance(config["prompts"], list)
        if config["prompts"]:
            assert "name" in config["prompts"][0]
            assert "messages" in config["prompts"][0]

        # RAG should be properly configured
        strat = config["rag"]["data_processing_strategies"][0]
        assert strat["parsers"][0]["config"]["content_fields"] == ["question"]
        db = config["rag"]["databases"][0]
        assert db["embedding_strategies"][0]["config"]["model"] == "nomic-embed-text"

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
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            pytest.raises(ConfigError, match="No configuration file found"),
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

    def test_runtime_provider_type(self, test_data_dir):
        """Test that runtime provider matches the schema's allowed types."""
        config_path = test_data_dir / "sample_config.yaml"
        config = load_config_dict(config_path=config_path)

        assert "runtime" in config
        assert config["runtime"]["provider"] == "openai"

    def test_type_safety(self, test_data_dir):
        """Test that loaded config matches expected types."""
        config_path = test_data_dir / "sample_config.yaml"
        config: LlamaFarmConfig = load_config_dict(config_path=config_path)

        # These should pass type checking
        version: str = config["version"]
        rag: dict = config["rag"]

        assert isinstance(version, str)
        assert isinstance(rag, dict)

        # Test RAG structure (strict schema)
        strat = rag["data_processing_strategies"][0]
        assert isinstance(strat["parsers"][0]["config"]["content_fields"], list)
        db = rag["databases"][0]
        assert isinstance(db["embedding_strategies"][0]["config"]["batch_size"], int)

    def test_config_with_no_prompts(self, test_data_dir):
        """Test configuration loading with minimal prompts section."""
        config_path = test_data_dir / "minimal_config.yaml"
        config = load_config_dict(config_path=config_path)

        # Should load successfully with minimal prompts (required field)
        assert "prompts" in config
        assert isinstance(config["prompts"], list)
        # Minimal config has at least one prompt
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
    strat = config["rag"]["data_processing_strategies"][0]
    parser_type = strat["parsers"][0]["type"]
    db = config["rag"]["databases"][0]
    embedder_model = db["embedding_strategies"][0]["config"]["model"]
    collection_name = db["config"]["collection_name"]

    assert parser_type in ["CSVParser_LlamaIndex", "CSVParser_Pandas", "CSVParser_Python"]
    assert embedder_model == "mxbai-embed-large"
    assert collection_name == "customer_support_knowledge_base"

    # Models list is optional in the current schema; rely on runtime instead
    assert config["runtime"]["provider"] == "openai"

    # Test accessing prompts (common use case)
    if config.get("prompts"):
        first_prompt = config["prompts"][0]
        assert "name" in first_prompt
        assert "messages" in first_prompt
        if first_prompt["messages"]:
            first_message = first_prompt["messages"][0]
            assert "content" in first_message


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
