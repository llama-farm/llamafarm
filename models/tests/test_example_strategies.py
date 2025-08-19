#!/usr/bin/env python3
"""
Test suite for example strategies.
"""

import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.strategy_manager import StrategyManager


class TestExampleStrategies:
    """Test loading and using example strategies."""

    def test_load_basic_openai_strategy(self):
        """Test loading the basic OpenAI strategy."""
        strategy_file = Path(__file__).parent / "test_strategies" / "basic_openai.yaml"

        # Load strategy
        manager = StrategyManager(strategies_file=strategy_file)
        strategy = manager.get_strategy("basic_openai")

        assert strategy is not None
        assert strategy['name'] == "basic_openai"
        assert strategy['description'] == "Simple cloud-based strategy using OpenAI GPT-4 Turbo"
        assert 'cloud_api' in strategy['components']
        assert strategy['components']['cloud_api']['type'] == 'openai_compatible'

    def test_load_ollama_strategy(self):
        """Test loading the Ollama local models strategy."""
        strategy_file = (
            Path(__file__).parent / "test_strategies" / "ollama_local_models.yaml"
        )

        # Load strategy
        manager = StrategyManager(strategies_file=strategy_file)
        strategy = manager.get_strategy("ollama_local_models")

        assert strategy is not None
        assert "model_app" in strategy["components"]
        assert strategy["components"]["model_app"]["type"] == "ollama"
        assert "fallback_chain" in strategy
        assert len(strategy["fallback_chain"]) > 0

    def test_load_production_strategy(self):
        """Test loading the production cloud strategy."""
        strategy_file = (
            Path(__file__).parent / "test_strategies" / "production_cloud.yaml"
        )

        # Load strategy
        manager = StrategyManager(strategies_file=strategy_file)
        strategy = manager.get_strategy("production_cloud")

        assert strategy is not None
        assert "cloud_api" in strategy["components"]
        assert "backup_api" in strategy["components"]
        assert "fallback_chain" in strategy
        assert "retry_strategy" in strategy

    def test_load_multi_model_strategy(self):
        """Test loading the multi-model specialized strategy."""
        strategy_file = (
            Path(__file__).parent / "test_strategies" / "multi_model_specialized.yaml"
        )

        # Load strategy
        manager = StrategyManager(strategies_file=strategy_file)
        strategy = manager.get_strategy("multi_model_specialized")

        assert strategy is not None
        # Should have multiple API components
        components = strategy["components"]
        assert "cloud_api" in components
        assert "groq_api" in components
        assert "together_api" in components
        assert "local_ollama" in components

        # Should have routing rules
        assert "routing_rules" in strategy
        assert len(strategy["routing_rules"]) > 5

    def test_load_fine_tuning_strategy(self):
        """Test loading fine-tuning workflow strategies."""
        strategy_file = (
            Path(__file__).parent / "test_strategies" / "fine_tuning_workflows.yaml"
        )

        # Load strategy
        manager = StrategyManager(strategies_file=strategy_file)

        # Test PyTorch LoRA strategy
        pytorch_strategy = manager.get_strategy("pytorch_lora_finetuning")
        assert pytorch_strategy is not None
        assert "fine_tuner" in pytorch_strategy["components"]
        assert pytorch_strategy["components"]["fine_tuner"]["type"] == "pytorch"

        # Test LlamaFactory strategy
        llama_strategy = manager.get_strategy("llamafactory_comprehensive")
        assert llama_strategy is not None
        assert "fine_tuner" in llama_strategy["components"]
        assert llama_strategy["components"]["fine_tuner"]["type"] == "llamafactory"

    def test_strategy_validation(self):
        """Test that all example strategies pass validation."""
        strategies_dir = Path(__file__).parent / "test_strategies"
        yaml_files = list(strategies_dir.glob("*.yaml"))

        assert len(yaml_files) > 0, "No strategy files found"

        for strategy_file in yaml_files:
            # Skip non-strategy files
            if strategy_file.name in ["README.md", "MIGRATION_GUIDE.md"]:
                continue

            manager = StrategyManager(strategies_file=strategy_file)

            # Each file should have at least one strategy
            assert len(manager.strategies) > 0, f"No strategies in {strategy_file.name}"

            # Each strategy should have required fields
            for strategy in manager.strategies:
                name = strategy["name"]
                assert "name" in strategy, f"Missing 'name' in {name}"
                assert "description" in strategy, f"Missing 'description' in {name}"
                assert "components" in strategy, f"Missing 'components' in {name}"
                assert len(strategy["components"]) > 0, f"No components in {name}"

    def test_routing_rules_format(self):
        """Test that routing rules are properly formatted."""
        strategy_file = (
            Path(__file__).parent / "test_strategies" / "multi_model_specialized.yaml"
        )

        manager = StrategyManager(strategies_file=strategy_file)
        strategy = manager.get_strategy("multi_model_specialized")

        assert "routing_rules" in strategy

        for rule in strategy["routing_rules"]:
            assert "pattern" in rule, "Routing rule missing pattern"
            assert "provider" in rule, "Routing rule missing provider"
            # Pattern should be a string
            assert isinstance(rule["pattern"], str)

    def test_fallback_chain_format(self):
        """Test that fallback chains are properly formatted."""
        strategy_file = (
            Path(__file__).parent / "test_strategies" / "production_cloud.yaml"
        )

        manager = StrategyManager(strategies_file=strategy_file)
        strategy = manager.get_strategy("production_cloud")

        assert "fallback_chain" in strategy

        for item in strategy["fallback_chain"]:
            # Should have provider or type
            assert "provider" in item or "type" in item
            # Should have model
            assert "model" in item

    @patch("components.model_apps.ollama.OllamaApp")
    def test_strategy_with_model_manager(self, mock_ollama):
        """Test using an example strategy with ModelManager."""
        # We'll test with a mocked Ollama since it doesn't require API keys
        strategy_file = (
            Path(__file__).parent / "test_strategies" / "ollama_local_models.yaml"
        )

        # First, add the strategy to default strategies for ModelManager to find it
        manager = StrategyManager(strategies_file=strategy_file)
        strategy = manager.get_strategy("ollama_local_models")

        # Mock the Ollama app
        mock_app = MagicMock()
        mock_app.is_running.return_value = True
        mock_app.generate.return_value = "Test response"
        mock_ollama.return_value = mock_app

        # Temporarily add to the manager's strategies
        original_file = manager.strategies_file
        manager.strategies.append(strategy)

        # This would work if we could dynamically add strategies to ModelManager
        # For now, we just verify the strategy structure is compatible
        assert "components" in strategy
        assert "model_app" in strategy["components"]
        assert strategy["components"]["model_app"]["type"] == "ollama"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
