#!/usr/bin/env python3
"""
Test suite for Models Components system.
Tests the new component-based architecture.
"""

import os
import sys
import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, Mock

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from components import (
    FineTunerFactory,
    ModelAppFactory,
    ModelRepositoryFactory,
    CloudAPIFactory,
    BaseFineTuner,
    BaseModelApp,
    BaseModelRepository,
    BaseCloudAPI
)
from core import ModelManager, StrategyManager, ConfigLoader


class TestComponentFactories:
    """Test component factory system."""
    
    def test_fine_tuner_factory_registration(self):
        """Test registering fine-tuner components."""
        # Create mock fine-tuner
        class MockFineTuner(BaseFineTuner):
            def validate_config(self):
                return []
            def prepare_model(self):
                return "mock_model"
            def prepare_dataset(self):
                return "mock_dataset"
            def start_training(self, job_id=None):
                return {"job_id": "test"}
            def stop_training(self):
                pass
            def get_training_status(self):
                return None
            def export_model(self, output_path, format="pytorch"):
                pass
            def get_supported_methods(self):
                return ["test"]
            def get_supported_models(self):
                return ["test-model"]
        
        # Register and create
        FineTunerFactory.register("mock", MockFineTuner)
        
        config = {"type": "mock", "config": {"test": True}}
        tuner = FineTunerFactory.create(config)
        
        assert isinstance(tuner, MockFineTuner)
        assert tuner.config["test"] is True
    
    def test_model_app_factory_registration(self):
        """Test registering model app components."""
        # Create mock model app
        class MockModelApp(BaseModelApp):
            def is_running(self):
                return True
            def start_service(self):
                return True
            def stop_service(self):
                pass
            def list_models(self):
                return []
            def generate(self, prompt, model=None, stream=False, **kwargs):
                return "Generated text"
            def chat(self, messages, model=None, stream=False, **kwargs):
                return "Chat response"
        
        # Register and create
        ModelAppFactory.register("mock", MockModelApp)
        
        config = {"type": "mock", "config": {"test": True}}
        app = ModelAppFactory.create(config)
        
        assert isinstance(app, MockModelApp)
        assert app.is_running() is True
    
    def test_factory_error_handling(self):
        """Test factory error handling."""
        # Test missing type
        with pytest.raises(ValueError) as exc_info:
            FineTunerFactory.create({"config": {}})
        assert "type not specified" in str(exc_info.value)
        
        # Test unknown type
        with pytest.raises(ValueError) as exc_info:
            FineTunerFactory.create({"type": "unknown_type"})
        assert "Unknown" in str(exc_info.value)


class TestModelManager:
    """Test ModelManager functionality."""
    
    @pytest.fixture
    def mock_config(self):
        """Provide mock configuration."""
        return {
            "fine_tuner": {
                "type": "pytorch",
                "config": {
                    "base_model": {"name": "test-model"},
                    "method": {"type": "lora"}
                }
            },
            "model_app": {
                "type": "ollama",
                "config": {
                    "default_model": "llama3.2"
                }
            }
        }
    
    def test_model_manager_initialization(self):
        """Test ModelManager initialization."""
        manager = ModelManager(strategy="local_development")
        
        assert manager.current_strategy == "local_development"
        assert hasattr(manager, "strategy_manager")
        assert hasattr(manager, "_components")
    
    def test_from_strategy(self):
        """Test creating ModelManager from strategy."""
        # Use an existing strategy instead of mocking
        manager = ModelManager.from_strategy("local_development")
        
        assert manager.current_strategy == "local_development"
        assert manager.strategy_config["components"]["model_app"]["type"] == "ollama"
    
    def test_get_info(self):
        """Test getting strategy information."""
        manager = ModelManager.from_strategy("local_development")
        info = manager.get_info()
        
        assert info["strategy"] == "local_development"
        assert "components" in info
        assert "model_app" in info["components"]


class TestStrategyManager:
    """Test StrategyManager functionality."""
    
    def test_strategy_manager_initialization(self):
        """Test StrategyManager initialization."""
        manager = StrategyManager()
        
        assert hasattr(manager, 'strategies')
        assert 'local_development' in manager.list_strategies()
    
    def test_list_strategies(self):
        """Test listing available strategies."""
        manager = StrategyManager()
        strategies = manager.list_strategies()
        
        assert len(strategies) > 0
        assert "local_development" in strategies
    
    def test_get_strategy(self):
        """Test getting a strategy."""
        manager = StrategyManager()
        config = manager.get_strategy("local_development")
        
        assert config is not None
        assert "components" in config
        assert "model_app" in config["components"]
        assert config["components"]["model_app"]["type"] == "ollama"
    
    def test_validate_strategy(self):
        """Test strategy validation."""
        manager = StrategyManager()
        
        # Valid strategy
        errors = manager.validate_strategy("local_development")
        assert len(errors) == 0
        
        # Invalid strategy (non-existent)
        errors = manager.validate_strategy("non_existent_strategy")
        assert len(errors) > 0
    
    def test_export_strategy(self, tmp_path):
        """Test exporting strategy."""
        manager = StrategyManager()
        
        # Export an existing strategy
        output_path = tmp_path / "test_export.yaml"
        success = manager.export_strategy("local_development", output_path)
        assert success
        
        # Verify file was created
        assert output_path.exists()


class TestPyTorchFineTuner:
    """Test PyTorch fine-tuner component."""
    
    @pytest.fixture
    def pytorch_config(self):
        """Provide PyTorch configuration."""
        return {
            "base_model": {
                "name": "test-model",
                "huggingface_id": "test/model"
            },
            "method": {
                "type": "lora",
                "r": 16,
                "alpha": 32
            },
            "dataset": {
                "path": "test_data.jsonl"
            },
            "training_args": {
                "output_dir": "./test_output",
                "num_train_epochs": 1
            }
        }
    
    @patch('components.fine_tuners.pytorch.pytorch_fine_tuner.PYTORCH_AVAILABLE', False)
    def test_pytorch_unavailable(self):
        """Test handling when PyTorch is not available."""
        with pytest.raises(ImportError):
            from components.fine_tuners.pytorch.pytorch_fine_tuner import PyTorchFineTuner
            tuner = PyTorchFineTuner({})
    
    def test_pytorch_config_validation(self, pytorch_config):
        """Test PyTorch configuration validation."""
        try:
            from components.fine_tuners.pytorch import PyTorchFineTuner
        except ImportError:
            pytest.skip("PyTorch not available")
        
        tuner = PyTorchFineTuner(pytorch_config)
        errors = tuner.validate_config()
        
        # Should have no errors with valid config
        assert len(errors) == 0
        
        # Test missing required fields
        bad_config = {}
        tuner_bad = PyTorchFineTuner(bad_config)
        errors = tuner_bad.validate_config()
        assert len(errors) > 0


class TestOllamaApp:
    """Test Ollama model app component."""
    
    @pytest.fixture
    def ollama_config(self):
        """Provide Ollama configuration."""
        return {
            "base_url": "http://localhost:11434",
            "default_model": "llama3.2",
            "timeout": 30
        }
    
    def test_ollama_initialization(self, ollama_config):
        """Test Ollama initialization."""
        from components.model_apps.ollama import OllamaApp
        
        app = OllamaApp(ollama_config)
        assert app.base_url == "http://localhost:11434"
        assert app.config["default_model"] == "llama3.2"
    
    @patch('requests.get')
    def test_ollama_is_running(self, mock_get, ollama_config):
        """Test checking if Ollama is running."""
        from components.model_apps.ollama import OllamaApp
        
        # Mock successful response
        mock_get.return_value.status_code = 200
        
        app = OllamaApp(ollama_config)
        assert app.is_running() is True
        
        # Mock failed response
        mock_get.side_effect = Exception("Connection error")
        assert app.is_running() is False
    
    @patch('requests.post')
    def test_ollama_generate(self, mock_post, ollama_config):
        """Test Ollama text generation."""
        from components.model_apps.ollama import OllamaApp
        
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Hello, world!"}
        mock_post.return_value = mock_response
        
        app = OllamaApp(ollama_config)
        response = app.generate("Say hello")
        
        assert response == "Hello, world!"
        mock_post.assert_called_once()


class TestOpenAICompatibleAPI:
    """Test OpenAI-compatible cloud API component."""
    
    @pytest.fixture
    def openai_config(self):
        """Provide OpenAI-compatible configuration."""
        return {
            "provider": "openai",
            "api_key": "test-key",
            "default_model": "gpt-3.5-turbo",
            "timeout": 60
        }
    
    @patch('components.cloud_apis.openai_compatible.openai_compatible_api.OPENAI_AVAILABLE', False)
    def test_openai_unavailable(self):
        """Test handling when OpenAI package is not available."""
        with pytest.raises(ImportError):
            from components.cloud_apis.openai_compatible.openai_compatible_api import OpenAICompatibleAPI
            api = OpenAICompatibleAPI({})
    
    @patch('components.cloud_apis.openai_compatible.openai_compatible_api.OPENAI_AVAILABLE', True)
    @patch('components.cloud_apis.openai_compatible.openai_compatible_api.OpenAI')
    @patch('components.cloud_apis.openai_compatible.openai_compatible_api.os.environ', {})
    def test_openai_initialization(self, mock_openai_class, openai_config):
        """Test OpenAI-compatible initialization."""
        from components.cloud_apis.openai_compatible.openai_compatible_api import OpenAICompatibleAPI
        
        api = OpenAICompatibleAPI(openai_config)
        # The OpenAI client should be called with api_key and organization=None for OpenAI provider
        mock_openai_class.assert_called_once_with(api_key="test-key", organization=None)


class TestHuggingFaceRepository:
    """Test HuggingFace repository component."""
    
    @pytest.fixture
    def hf_config(self):
        """Provide HuggingFace configuration."""
        return {
            "token": "test-token",
            "cache_dir": "/tmp/hf_cache"
        }
    
    @patch('components.model_repositories.huggingface.huggingface_repository.HUGGINGFACE_AVAILABLE', False)
    def test_hf_unavailable(self):
        """Test handling when HuggingFace Hub is not available."""
        with pytest.raises(ImportError):
            from components.model_repositories.huggingface.huggingface_repository import HuggingFaceRepository
            repo = HuggingFaceRepository({})
    
    @patch('components.model_repositories.huggingface.huggingface_repository.HUGGINGFACE_AVAILABLE', True)
    @patch('components.model_repositories.huggingface.huggingface_repository.HfApi')
    def test_hf_initialization(self, mock_hf_api, hf_config):
        """Test HuggingFace initialization."""
        from components.model_repositories.huggingface.huggingface_repository import HuggingFaceRepository
        
        repo = HuggingFaceRepository(hf_config)
        assert repo.token == "test-token"
        mock_hf_api.assert_called_once()


class TestIntegration:
    """Test integration between components."""
    
    def test_manual_registration(self):
        """Test manual component registration."""
        # Import should trigger manual registration
        import components
        
        # Check that essential components are registered
        from components.factory import ModelAppFactory, FineTunerFactory
        
        # Verify components are registered
        registered_model_apps = ModelAppFactory.list_components()
        registered_fine_tuners = FineTunerFactory.list_components()
        
        # Check that at least some components are registered
        assert len(registered_model_apps) > 0 or len(registered_fine_tuners) > 0
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with mocked components."""
        # Mock the factory
        mock_app = MagicMock()
        mock_app.is_running.return_value = True
        mock_app.generate.return_value = "Test response"
        
        with patch.object(ModelAppFactory, 'create', return_value=mock_app):
            # Use the local_development strategy which has ollama configured
            manager = ModelManager(strategy="local_development")
            
            # Test generation
            response = manager.generate("Test prompt")
            assert response == "Test response"
            mock_app.generate.assert_called_once()


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])