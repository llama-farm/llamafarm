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

        # Parser configuration extraction
        parser_type = rag_config["parsers"]["csv"]["type"]
        parser_config = rag_config["parsers"]["csv"]["config"]
        content_fields = parser_config["content_fields"]
        metadata_fields = parser_config["metadata_fields"]

        assert parser_type == "CustomerSupportCSVParser"
        assert isinstance(content_fields, list)
        assert isinstance(metadata_fields, list)
        assert len(content_fields) >= 1
        assert len(metadata_fields) >= 1

        # Embedder configuration extraction
        embedder_type = rag_config["embedders"]["default"]["type"]
        embedder_config = rag_config["embedders"]["default"]["config"]
        embedding_model = embedder_config["model"]
        batch_size = embedder_config["batch_size"]

        assert embedder_type == "OllamaEmbedder"
        assert isinstance(embedding_model, str)
        assert isinstance(batch_size, int)
        assert batch_size > 0

        # Vector store configuration extraction
        vector_store_type = rag_config["vector_stores"]["default"]["type"]
        vector_store_config = rag_config["vector_stores"]["default"]["config"]
        collection_name = vector_store_config["collection_name"]
        persist_directory = vector_store_config["persist_directory"]

        assert vector_store_type == "ChromaStore"
        assert isinstance(collection_name, str)
        assert isinstance(persist_directory, str)
        assert len(collection_name) > 0
        assert len(persist_directory) > 0

    def test_model_manager_usage(self, sample_config_dir):
        """Test how a model manager module would use the configuration."""
        config_path = sample_config_dir / "sample_config.yaml"
        config: LlamaFarmConfig = load_config_dict(config_path=config_path)

        # Simulate model manager extracting model configurations
        models = config["models"]

        # Group models by provider (common pattern)
        models_by_provider = {}
        for model in models:
            provider = model["provider"]
            if provider not in models_by_provider:
                models_by_provider[provider] = []
            models_by_provider[provider].append(model)

        # Verify we have models for different providers
        assert "local" in models_by_provider
        assert "openai" in models_by_provider
        assert len(models_by_provider["local"]) >= 1
        assert len(models_by_provider["openai"]) >= 1

        # Test model filtering (common use case)
        local_models = [m for m in models if m["provider"] == "local"]
        cloud_models = [
            m for m in models if m["provider"] in ["openai", "anthropic", "google"]
        ]

        assert len(local_models) >= 1
        assert len(cloud_models) >= 1

        # Verify model structure
        for model in models:
            assert "provider" in model
            assert "model" in model
            assert isinstance(model["provider"], str)
            assert isinstance(model["model"], str)
            assert len(model["model"]) > 0

    def test_prompt_manager_usage(self, sample_config_dir):
        """Test how a prompt manager module would use the configuration."""
        config_path = sample_config_dir / "sample_config.yaml"
        config: LlamaFarmConfig = load_config_dict(config_path=config_path)

        # Handle optional prompts field
        prompts = config.get("prompts", [])
        if prompts:
            # Simulate prompt manager creating a lookup dictionary
            prompt_lookup = {}
            for prompt in prompts:
                if "name" in prompt:
                    prompt_lookup[prompt["name"]] = prompt

            # Test accessing specific prompts
            if "customer_support" in prompt_lookup:
                cs_prompt = prompt_lookup["customer_support"]
                assert "prompt" in cs_prompt
                assert isinstance(cs_prompt["prompt"], str)
                assert len(cs_prompt["prompt"]) > 0

                # Check optional fields
                if "description" in cs_prompt:
                    assert isinstance(cs_prompt["description"], str)

            # Test prompt filtering by name pattern
            support_prompts = [
                p for p in prompts if "support" in p.get("name", "").lower()
            ]

            # Should have at least one support prompt in our sample
            assert len(support_prompts) >= 1

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
                assert "models" in config, f"Missing models section in {config_file}"

                # Check RAG structure
                rag = config["rag"]
                required_rag_keys = ["parsers", "embedders", "vector_stores", "retrieval_strategies", "defaults"]
                for key in required_rag_keys:
                    assert key in rag, f"Missing {key} in RAG config in {config_file}"

                # Check models structure
                models = config["models"]
                assert isinstance(models, list), (
                    f"Models should be a list in {config_file}"
                )
                assert len(models) > 0, f"No models defined in {config_file}"

                for i, model in enumerate(models):
                    assert "provider" in model, (
                        f"Model {i} missing provider in {config_file}"
                    )
                    assert "model" in model, (
                        f"Model {i} missing model name in {config_file}"
                    )

    def test_runtime_config_reload(self, sample_config_dir):
        """Test configuration reloading during runtime (common pattern)."""
        config_path = sample_config_dir / "sample_config.yaml"

        # Initial load
        config1 = load_config_dict(config_path=config_path)
        initial_model_count = len(config1["models"])

        # Reload (simulating runtime configuration reload)
        config2 = load_config_dict(config_path=config_path)
        reloaded_model_count = len(config2["models"])

        # Should be identical
        assert initial_model_count == reloaded_model_count
        assert config1["version"] == config2["version"]
        assert config1["rag"]["parsers"]["csv"]["type"] == config2["rag"]["parsers"]["csv"]["type"]

    def test_environment_specific_configs(self, temp_config_file):
        """Test loading different configurations for different environments."""
        # Development config
        dev_config = """version: v1

name: dev_config
namespace: test

rag:
  parsers:
    csv:
      type: "CustomerSupportCSVParser"
      config:
        content_fields: ["question"]
        metadata_fields: ["category"]
        id_field: "id"
        combine_content: true
      file_extensions: [".csv"]
      mime_types: ["text/csv"]
  embedders:
    default:
      type: "OllamaEmbedder"
      config:
        model: "nomic-embed-text"  # Smaller model for dev
        base_url: "http://localhost:11434"
        batch_size: 8  # Smaller batch for dev
        timeout: 30
  vector_stores:
    default:
      type: "ChromaStore"
      config:
        collection_name: "dev_collection"
        persist_directory: "./data/dev"
  retrieval_strategies:
    default:
      type: "BasicSimilarityStrategy"
      config:
        distance_metric: "cosine"
  defaults:
    parser: "auto"
    embedder: "default"
    vector_store: "default"
    retrieval_strategy: "default"

models:
  - provider: "local"
    model: "llama3.1:8b"  # Smaller model for dev

datasets:
  - name: "dev_dataset"
    files: ["test_file.csv"]
    parser: "csv"
    embedder: "default"
    vector_store: "default"
    retrieval_strategy: "default"

prompts:
  - name: "dev_prompt"
    prompt: "This is a dev prompt."
    description: "This is a description of the dev prompt."
"""

        # Production config
        prod_config = """version: v1

name: prod_config
namespace: test

rag:
  parsers:
    csv:
      type: "CustomerSupportCSVParser"
      config:
        content_fields: ["question", "answer", "solution"]
        metadata_fields: ["category", "priority", "timestamp"]
        id_field: "id"
        combine_content: true
      file_extensions: [".csv"]
      mime_types: ["text/csv"]
  embedders:
    default:
      type: "OllamaEmbedder"
      config:
        model: "mxbai-embed-large"  # Better model for prod
        base_url: "http://localhost:11434"
        batch_size: 64  # Larger batch for prod
        timeout: 60
  vector_stores:
    default:
      type: "ChromaStore"
      config:
        collection_name: "production_collection"
        persist_directory: "./data/production"
  retrieval_strategies:
    default:
      type: "BasicSimilarityStrategy"
      config:
        distance_metric: "cosine"
  defaults:
    parser: "auto"
    embedder: "default"
    vector_store: "default"
    retrieval_strategy: "default"

models:
  - provider: "local"
    model: "llama3.1:70b"  # Larger model for prod
  - provider: "openai"
    model: "gpt-4"  # Backup cloud model

datasets:
  - name: "prod_dataset"
    files: ["test_file.csv"]
    parser: "csv"
    embedder: "default"
    vector_store: "default"
    retrieval_strategy: "default"

prompts:
  - name: "prod_prompt"
    prompt: "This is a prod prompt."
    description: "This is a description of the prod prompt."
"""

        dev_path = temp_config_file(dev_config, ".yaml")
        prod_path = temp_config_file(prod_config, ".yaml")

        # Load development config
        dev_cfg = load_config_dict(config_path=dev_path)
        assert dev_cfg["rag"]["embedders"]["default"]["config"]["batch_size"] == 8
        assert (
            dev_cfg["rag"]["vector_stores"]["default"]["config"]["collection_name"]
            == "dev_collection"
        )
        assert len(dev_cfg["models"]) == 1

        # Load production config
        prod_cfg = load_config_dict(config_path=prod_path)
        assert prod_cfg["rag"]["embedders"]["default"]["config"]["batch_size"] == 64
        assert (
            prod_cfg["rag"]["vector_stores"]["default"]["config"]["collection_name"]
            == "production_collection"
        )
        assert len(prod_cfg["models"]) == 2

    def test_config_driven_component_initialization(self, sample_config_dir):
        """Test how components would be initialized based on configuration."""
        config_path = sample_config_dir / "sample_config.yaml"
        config: LlamaFarmConfig = load_config_dict(config_path=config_path)

        # Simulate component factory pattern based on config
        def create_parser_from_config(rag_config):
            """Simulate parser factory."""
            parser_type = rag_config["parsers"]["csv"]["type"]
            parser_config = rag_config["parsers"]["csv"]["config"]

            if parser_type == "CustomerSupportCSVParser":
                return {
                    "type": parser_type,
                    "content_fields": parser_config["content_fields"],
                    "metadata_fields": parser_config["metadata_fields"],
                }
            else:
                raise ValueError(f"Unknown parser type: {parser_type}")

        def create_embedder_from_config(rag_config):
            """Simulate embedder factory."""
            embedder_type = rag_config["embedders"]["default"]["type"]
            embedder_config = rag_config["embedders"]["default"]["config"]

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
        assert parser["type"] == "CustomerSupportCSVParser"
        assert len(parser["content_fields"]) > 0

        embedder = create_embedder_from_config(rag_config)
        assert embedder["type"] == "OllamaEmbedder"
        assert embedder["batch_size"] > 0

        # Test model initialization
        models = config["models"]
        initialized_models = []

        for model_config in models:
            model_instance = {
                "provider": model_config["provider"],
                "model_name": model_config["model"],
                "initialized": True,
            }
            initialized_models.append(model_instance)

        assert len(initialized_models) == len(models)
        assert all(m["initialized"] for m in initialized_models)


def test_cross_module_config_sharing():
    """Test configuration sharing between multiple simulated modules."""
    # This simulates how config would be shared in a real application

    # Create a shared config instance
    test_dir = Path(__file__).parent
    config_path = test_dir / "sample_config.yaml"
    shared_config = load_config_dict(config_path=config_path)

    # Module 1: RAG Service
    class RAGService:
        def __init__(self, config: LlamaFarmConfig):
            self.parser_type = config["rag"]["parsers"]["csv"]["type"]
            self.embedder_model = config["rag"]["embedders"]["default"]["config"]["model"]
            self.collection_name = config["rag"]["vector_stores"]["default"]["config"][
                "collection_name"
            ]

    # Module 2: Model Manager
    class ModelManager:
        def __init__(self, config: LlamaFarmConfig):
            self.models = config["models"]
            self.local_models = [m for m in self.models if m["provider"] == "local"]
            self.cloud_models = [m for m in self.models if m["provider"] != "local"]

    # Module 3: Prompt Service
    class PromptService:
        def __init__(self, config: LlamaFarmConfig):
            prompts = config.get("prompts", [])
            self.prompt_lookup = {p["name"]: p for p in prompts if "name" in p}

    # Initialize all services with shared config
    rag_service = RAGService(shared_config)
    model_manager = ModelManager(shared_config)
    prompt_service = PromptService(shared_config)

    # Verify each service extracted its configuration correctly
    assert rag_service.parser_type == "CustomerSupportCSVParser"
    assert rag_service.embedder_model == "mxbai-embed-large"
    assert rag_service.collection_name == "customer_support_knowledge_base"

    assert len(model_manager.models) >= 1
    assert len(model_manager.local_models) >= 1

    if "customer_support" in prompt_service.prompt_lookup:
        cs_prompt = prompt_service.prompt_lookup["customer_support"]
        assert "assistant" in cs_prompt["prompt"].lower()

    # Test that all services are working with the same config version
    assert shared_config["version"] == "v1"


if __name__ == "__main__":
    # Run tests when executed directly
    pytest.main([__file__, "-v"])
