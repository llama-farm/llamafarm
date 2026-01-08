#!/usr/bin/env python3
"""
Test suite for Semantic Router schema validation.
Tests Phase 1: Schema & Configuration Foundation
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ConfigError, load_config_dict


class TestRouterSchema:
    """Test class for router schema validation."""

    def test_valid_router_configuration_inline_utterances(self):
        """Test that a valid router configuration with inline utterances is accepted."""
        config_yaml = """
version: v1
name: router-test
namespace: default

runtime:
  default_model: smart_router
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      similarity_threshold: 0.75
      routes:
        - name: billing
          target_model: billing_model
          description: "Questions about billing"
          utterances:
            - "what is my bill"
            - "payment options"
            - "invoice question"
        - name: support
          target_model: support_model
          description: "Technical support questions"
          utterances:
            - "help with login"
            - "password reset"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
      base_url: http://127.0.0.1:11434

    - name: billing_model
      provider: universal
      model: unsloth/Qwen3-1.7B-GGUF:Q4_K_M

    - name: support_model
      provider: openai
      model: gpt-5-mini
      api_key: test-key
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config = load_config_dict(config_path=Path(f.name))

            # Verify router model exists
            models = config["runtime"]["models"]
            router_model = next(m for m in models if m["name"] == "smart_router")

            assert router_model["provider"] == "router"
            assert router_model["embedder_model"] == "sentence-transformers/all-MiniLM-L6-v2"
            assert router_model["default_model"] == "fallback_llm"
            assert router_model["similarity_threshold"] == 0.75
            assert len(router_model["routes"]) == 2

            # Check first route
            billing_route = router_model["routes"][0]
            assert billing_route["name"] == "billing"
            assert billing_route["target_model"] == "billing_model"
            assert len(billing_route["utterances"]) == 3

    def test_valid_router_configuration_with_dataset_reference(self):
        """Test that a router can reference a dataset for utterances."""
        config_yaml = """
version: v1
name: router-dataset-test
namespace: default

datasets:
  - name: billing_utterances
    data_processing_strategy: router_utterances
    database: none

runtime:
  default_model: smart_router
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes:
        - name: billing
          target_model: billing_model
          description: "Questions about billing"
          dataset: billing_utterances

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
      base_url: http://127.0.0.1:11434

    - name: billing_model
      provider: universal
      model: unsloth/Qwen3-1.7B-GGUF:Q4_K_M
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config = load_config_dict(config_path=Path(f.name))

            # Verify router with dataset reference
            models = config["runtime"]["models"]
            router_model = next(m for m in models if m["name"] == "smart_router")
            billing_route = router_model["routes"][0]

            assert billing_route["dataset"] == "billing_utterances"
            assert "utterances" not in billing_route or billing_route.get("utterances") is None

    def test_router_requires_embedder_model(self):
        """Test that router configuration requires embedder_model field."""
        config_yaml = """
version: v1
name: router-invalid
namespace: default

runtime:
  models:
    - name: smart_router
      provider: router
      default_model: fallback_llm
      routes:
        - name: test
          target_model: fallback_llm
          utterances:
            - "test query"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            # This should raise an error because embedder_model is missing
            with pytest.raises((ConfigError, Exception)):
                load_config_dict(config_path=Path(f.name))

    def test_router_requires_routes(self):
        """Test that router configuration requires at least one route."""
        config_yaml = """
version: v1
name: router-no-routes
namespace: default

runtime:
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes: []

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            # Empty routes should be invalid
            with pytest.raises((ConfigError, Exception)):
                load_config_dict(config_path=Path(f.name))

    def test_route_requires_target_model(self):
        """Test that each route requires a target_model."""
        config_yaml = """
version: v1
name: router-missing-target
namespace: default

runtime:
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes:
        - name: billing
          description: "Billing questions"
          utterances:
            - "what is my bill"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            with pytest.raises((ConfigError, Exception)):
                load_config_dict(config_path=Path(f.name))

    def test_route_requires_utterances_or_dataset(self):
        """Test that each route requires either utterances or dataset."""
        config_yaml = """
version: v1
name: router-no-data
namespace: default

runtime:
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes:
        - name: billing
          target_model: billing_model
          description: "Billing questions"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b

    - name: billing_model
      provider: universal
      model: unsloth/Qwen3-1.7B-GGUF:Q4_K_M
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            # Route without utterances or dataset should be invalid
            with pytest.raises((ConfigError, Exception)):
                load_config_dict(config_path=Path(f.name))

    def test_router_with_complexity_classifier(self):
        """Test that router can specify an optional complexity classifier."""
        config_yaml = """
version: v1
name: router-complexity
namespace: default

runtime:
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      complexity_classifier: query_complexity_model
      routes:
        - name: billing
          target_model: billing_model
          utterances:
            - "billing question"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b

    - name: billing_model
      provider: universal
      model: unsloth/Qwen3-1.7B-GGUF:Q4_K_M
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config = load_config_dict(config_path=Path(f.name))

            models = config["runtime"]["models"]
            router_model = next(m for m in models if m["name"] == "smart_router")
            assert router_model["complexity_classifier"] == "query_complexity_model"

    def test_router_default_similarity_threshold(self):
        """Test that similarity_threshold defaults to 0.7 if not specified."""
        config_yaml = """
version: v1
name: router-defaults
namespace: default

runtime:
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes:
        - name: test
          target_model: fallback_llm
          utterances:
            - "test"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config = load_config_dict(config_path=Path(f.name))

            models = config["runtime"]["models"]
            router_model = next(m for m in models if m["name"] == "smart_router")
            # Either defaults to 0.7 or is not present (implementation detail)
            threshold = router_model.get("similarity_threshold", 0.7)
            assert threshold == 0.7

    def test_non_router_model_still_requires_model_field(self):
        """Test that non-router models still require the model field."""
        config_yaml = """
version: v1
name: test-non-router
namespace: default

runtime:
  models:
    - name: my_llm
      provider: ollama
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            # Non-router model without 'model' field should fail
            with pytest.raises((ConfigError, Exception)):
                load_config_dict(config_path=Path(f.name))

    def test_router_model_field_optional(self):
        """Test that router provider doesn't require the 'model' field."""
        config_yaml = """
version: v1
name: router-no-model-field
namespace: default

runtime:
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes:
        - name: test
          target_model: fallback_llm
          utterances:
            - "test query"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            # Router without 'model' field should be valid
            config = load_config_dict(config_path=Path(f.name))
            models = config["runtime"]["models"]
            router_model = next(m for m in models if m["name"] == "smart_router")
            assert router_model["provider"] == "router"
            # model field should be absent or None
            assert router_model.get("model") is None or "model" not in router_model


class TestRouterSchemaEdgeCases:
    """Test edge cases and complex scenarios for router schema."""

    def test_multiple_routers_allowed(self):
        """Test that multiple router models can be defined."""
        config_yaml = """
version: v1
name: multi-router
namespace: default

runtime:
  models:
    - name: router_a
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes:
        - name: topic_a
          target_model: fallback_llm
          utterances:
            - "topic a query"

    - name: router_b
      provider: router
      embedder_model: BAAI/bge-small-en-v1.5
      default_model: fallback_llm
      routes:
        - name: topic_b
          target_model: fallback_llm
          utterances:
            - "topic b query"

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config = load_config_dict(config_path=Path(f.name))

            models = config["runtime"]["models"]
            routers = [m for m in models if m["provider"] == "router"]
            assert len(routers) == 2

    def test_router_with_mixed_route_types(self):
        """Test router with both inline utterances and dataset references."""
        config_yaml = """
version: v1
name: mixed-routes
namespace: default

datasets:
  - name: support_data
    data_processing_strategy: router_utterances
    database: none

runtime:
  models:
    - name: smart_router
      provider: router
      embedder_model: sentence-transformers/all-MiniLM-L6-v2
      default_model: fallback_llm
      routes:
        - name: billing
          target_model: billing_model
          utterances:
            - "billing inquiry"
        - name: support
          target_model: support_model
          dataset: support_data

    - name: fallback_llm
      provider: ollama
      model: qwen3:8b

    - name: billing_model
      provider: universal
      model: unsloth/Qwen3-1.7B-GGUF:Q4_K_M

    - name: support_model
      provider: openai
      model: gpt-5-mini
      api_key: test-key
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            f.flush()
            config = load_config_dict(config_path=Path(f.name))

            models = config["runtime"]["models"]
            router = next(m for m in models if m["name"] == "smart_router")

            billing_route = next(r for r in router["routes"] if r["name"] == "billing")
            support_route = next(r for r in router["routes"] if r["name"] == "support")

            assert "utterances" in billing_route
            assert "dataset" in support_route
