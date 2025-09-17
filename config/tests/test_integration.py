#!/usr/bin/env python3
"""
Integration tests demonstrating how other modules in the project would use the config module.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import LlamaFarmConfig, load_config_dict


class TestModuleIntegration:
    """Test how the config module would be used by other modules in the project."""

    @pytest.fixture
    def sample_config_dir(self):
        """Return the path to sample configurations."""
        return Path(__file__).parent

    def test_rag_module_usage(self, sample_config_dir):
        """Test how a RAG module would use the configuration."""
        config_path = sample_config_dir / "sample_config.yaml"
        config = load_config_dict(config_path=config_path)

        # Simulate RAG module extracting its configuration
        rag_config = config["rag"]
        strat = rag_config["data_processing_strategies"][0]

        # Parser configuration extraction
        parser_type = strat["parsers"][0]["type"]
        parser_config = strat["parsers"][0]["config"]
        content_fields = parser_config["content_fields"]
        metadata_fields = parser_config["metadata_fields"]

        assert parser_type in ["CSVParser_LlamaIndex", "CSVParser_Pandas", "CSVParser_Python"]
        assert isinstance(content_fields, list)
        assert isinstance(metadata_fields, list)
        assert len(content_fields) >= 1
        assert len(metadata_fields) >= 1

        # Embedder configuration extraction
        db = rag_config["databases"][0]
        embedder_type = db["embedding_strategies"][0]["type"]
        embedder_config = db["embedding_strategies"][0]["config"]
        embedding_model = embedder_config["model"]
        batch_size = embedder_config["batch_size"]

        assert embedder_type == "OllamaEmbedder"
        assert isinstance(embedding_model, str)
        assert isinstance(batch_size, int)
        assert batch_size > 0

        # Vector store configuration extraction
        vector_store_type = db["type"]
        vector_store_config = db["config"]
        collection_name = vector_store_config["collection_name"]
        persist_directory = vector_store_config["persist_directory"]

        assert vector_store_type == "ChromaStore"
        assert isinstance(collection_name, str)
        assert isinstance(persist_directory, str)
        assert len(collection_name) > 0
        assert len(persist_directory) > 0

    def test_prompt_manager_usage(self, sample_config_dir):
        """Test how a prompt manager module would use the configuration."""
        config_path = sample_config_dir / "sample_config.yaml"
        config = load_config_dict(config_path=config_path)

        # Handle prompts: list of role/content objects
        prompts = config.get("prompts", [])
        if prompts:
            first = prompts[0]
            assert "content" in first

    def test_configuration_validation_service(self, sample_config_dir):
        """Test how a configuration validation service would use the module."""
        # Test loading different configurations and validating them
        configs_to_test = [
            "sample_config.yaml",
            "sample_config.toml",
            "minimal_config.yaml",
        ]

        for config_file in configs_to_test:
            config_path = sample_config_dir / config_file
            if config_path.exists():
                # Validate configuration loads successfully
                config = load_config_dict(config_path=config_path)

                # Perform validation checks that a service might do
                assert config["version"] == "v1", f"Invalid version in {config_file}"

                # Check required sections exist
                assert "rag" in config, f"Missing RAG section in {config_file}"

                # Check RAG structure
                rag = config["rag"]
                # Strict schema uses databases and data_processing_strategies arrays
                assert "databases" in rag
                assert "data_processing_strategies" in rag

                # Runtime is the canonical place for execution provider
                runtime = config["runtime"]
                assert runtime["provider"] == "openai"

    def test_runtime_config_reload(self, sample_config_dir):
        """Test configuration reloading during runtime (common pattern)."""
        config_path = sample_config_dir / "sample_config.yaml"

        # Initial load
        config1 = load_config_dict(config_path=config_path)

        # Reload (simulating runtime configuration reload)
        config2 = load_config_dict(config_path=config_path)

        # Should be identical
        assert config1["version"] == config2["version"]
        assert (
            config1["rag"]["data_processing_strategies"][0]["parsers"][0]["type"]
            == config2["rag"]["data_processing_strategies"][0]["parsers"][0]["type"]
        )

    def test_environment_specific_configs(self, temp_config_file):
        """Test loading different configurations for different environments."""
        # Development config
        dev_config = """version: v1

name: dev_config
namespace: test

rag:
  databases:
    - name: "dev_db"
      type: "ChromaStore"
      config:
        collection_name: "dev_collection"
        persist_directory: "./data/dev"
      embedding_strategies:
        - name: "dev_embedding"
          type: "OllamaEmbedder"
          config:
            model: "nomic-embed-text"
            base_url: "http://localhost:11434"
            batch_size: 8
            timeout: 30
      retrieval_strategies:
        - name: "dev_retrieval"
          type: "BasicSimilarityStrategy"
          config:
            distance_metric: "cosine"
          default: true
  data_processing_strategies:
    - name: "default"
      description: "Dev strategy"
      parsers:
        - type: "CSVParser_LlamaIndex"
          config:
            content_fields: ["question"]
            metadata_fields: ["category"]
            id_field: "id"
            combine_content: true
          file_extensions: [".csv"]
      extractors: []

runtime:
  provider: "openai"
  model: "llama3.1:8b"
  api_key: "ollama"
  base_url: "http://localhost:11434/v1"
  model_api_parameters:
    temperature: 0.5

datasets:
  - name: "dev_dataset"
    files: ["test_file.csv"]
    data_processing_strategy: "default"
    database: "dev_db"

prompts:
  - role: "system"
    content: "This is a dev prompt."
"""

        # Production config
        prod_config = """version: v1

name: prod_config
namespace: test

rag:
  databases:
    - name: "prod_db"
      type: "ChromaStore"
      config:
        collection_name: "production_collection"
        persist_directory: "./data/production"
      embedding_strategies:
        - name: "prod_embedding"
          type: "OllamaEmbedder"
          config:
            model: "mxbai-embed-large"
            base_url: "http://localhost:11434"
            batch_size: 64
            timeout: 60
      retrieval_strategies:
        - name: "prod_retrieval"
          type: "BasicSimilarityStrategy"
          config:
            distance_metric: "cosine"
          default: true
  data_processing_strategies:
    - name: "default"
      description: "Prod strategy"
      parsers:
        - type: "CSVParser_LlamaIndex"
          config:
            content_fields: ["question", "answer", "solution"]
            metadata_fields: ["category", "priority", "timestamp"]
            id_field: "id"
            combine_content: true
          file_extensions: [".csv"]
      extractors: []

runtime:
  provider: "openai"
  model: "llama3.1:8b"
  api_key: "ollama"
  base_url: "http://localhost:11434/v1"
  model_api_parameters:
    temperature: 0.5

datasets:
  - name: "prod_dataset"
    files: ["test_file.csv"]
    data_processing_strategy: "default"
    database: "prod_db"

prompts:
  - role: "system"
    content: "This is a prod prompt."
"""

        dev_path = temp_config_file(dev_config, ".yaml")
        prod_path = temp_config_file(prod_config, ".yaml")

        # Load development config
        dev_cfg = load_config_dict(config_path=dev_path)
        dev_db = dev_cfg["rag"]["databases"][0]
        assert dev_db["embedding_strategies"][0]["config"]["batch_size"] == 8
        assert dev_db["config"]["collection_name"] == "dev_collection"

        # Load production config
        prod_cfg = load_config_dict(config_path=prod_path)
        prod_db = prod_cfg["rag"]["databases"][0]
        assert prod_db["embedding_strategies"][0]["config"]["batch_size"] == 64
        assert prod_db["config"]["collection_name"] == "production_collection"
        # Models list is optional; rely on runtime instead
        assert prod_cfg["runtime"]["provider"] == "openai"

    def test_config_driven_component_initialization(self, sample_config_dir):
        """Test how components would be initialized based on configuration."""
        config_path = sample_config_dir / "sample_config.yaml"
        config = load_config_dict(config_path=config_path)

        # Simulate component factory pattern based on config
        def create_parser_from_config(rag_config):
            """Simulate parser factory."""
            strat = rag_config["data_processing_strategies"][0]
            parser_type = strat["parsers"][0]["type"]
            parser_config = strat["parsers"][0]["config"]

            if parser_type == "CSVParser_LlamaIndex":
                return {
                    "type": parser_type,
                    "content_fields": parser_config["content_fields"],
                    "metadata_fields": parser_config["metadata_fields"],
                }
            else:
                raise ValueError(f"Unknown parser type: {parser_type}")

        def create_embedder_from_config(rag_config):
            """Simulate embedder factory."""
            db = rag_config["databases"][0]
            embedder_type = db["embedding_strategies"][0]["type"]
            embedder_config = db["embedding_strategies"][0]["config"]

            if embedder_type == "OllamaEmbedder":
                return {
                    "type": embedder_type,
                    "model": embedder_config["model"],
                    "batch_size": embedder_config["batch_size"],
                }
            else:
                raise ValueError(f"Unknown embedder type: {embedder_type}")

        # Test component creation
        rag_config = config["rag"]

        parser = create_parser_from_config(rag_config)
        assert len(parser["content_fields"]) > 0
        assert parser["type"] in [
            "CSVParser_LlamaIndex",
            "CSVParser_Pandas",
            "CSVParser_Python",
        ]

        embedder = create_embedder_from_config(rag_config)
        assert embedder["type"] == "OllamaEmbedder"
        assert embedder["batch_size"] > 0

        # Test runtime initialization (models list optional in schema)
        runtime = config["runtime"]
        runtime_instance = {
            "provider": runtime["provider"],
            "model_name": runtime["model"],
            "initialized": True,
        }
        assert runtime_instance["initialized"] is True


def test_cross_module_config_sharing():
    """Test configuration sharing between multiple simulated modules."""
    # This simulates how config would be shared in a real application

    # Create a shared config instance
    test_dir = Path(__file__).parent
    config_path = test_dir / "sample_config.yaml"
    # Load dict for cross-module sharing, then create a strongly-typed model for services
    shared_config_dict = load_config_dict(config_path=config_path)
    shared_config = LlamaFarmConfig(**shared_config_dict)

    # Module 1: RAG Service
    class RAGService:
        def __init__(self, config: LlamaFarmConfig):
            strat = config.rag.data_processing_strategies[0]
            db = config.rag.databases[0]
            self.parser_type = strat.parsers[0].type
            self.embedder_type = db.embedding_strategies[0].type
            self.collection_type = db.type

    # Module 3: Prompt Service
    class PromptService:
        def __init__(self, config: LlamaFarmConfig):
            # New schema Prompt has only role/content; keep as list for typed access
            self.prompts = config.prompts

    # Initialize all services with shared config
    rag_service = RAGService(shared_config)
    prompt_service = PromptService(shared_config)

    # Verify each service extracted its configuration correctly (support enum or str)
    parser_type = getattr(rag_service.parser_type, "value", rag_service.parser_type)
    embedder_type = getattr(rag_service.embedder_type, "value", rag_service.embedder_type)
    collection_type = getattr(rag_service.collection_type, "value", rag_service.collection_type)
    assert parser_type == "CSVParser_LlamaIndex"
    assert embedder_type == "OllamaEmbedder"
    assert collection_type == "ChromaStore"

    # Typed prompts don't carry names in the new schema; validate first prompt content
    if prompt_service.prompts:
        cs_prompt = prompt_service.prompts[0]
        assert hasattr(cs_prompt, "content") and isinstance(cs_prompt.content, str)
        assert "assistant" in cs_prompt.content.lower()

    # Test that all services are working with the same config version
    assert getattr(shared_config.version, "value", shared_config.version) == "v1"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
