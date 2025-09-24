"""
Tests for the RuntimeManager multi-model support.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import subprocess

# Add parent directory to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.core.runtime_manager import RuntimeManager


class TestRuntimeManager:
    """Test suite for RuntimeManager."""
    
    def test_init_with_runtime_models(self):
        """Test initialization with runtime_models configuration."""
        config = {
            'default_model': 'primary',
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b',
                    'base_url': 'http://localhost:11434',
                    'parameters': {'temperature': 0.7}
                },
                {
                    'name': 'creative',
                    'provider': 'ollama',
                    'model': 'mixtral:8x7b',
                    'parameters': {'temperature': 0.9}
                }
            ]
        }
        
        manager = RuntimeManager(config)
        
        assert len(manager.runtime_models) == 2
        assert manager.default_model == 'primary'
        assert 'primary' in manager.runtime_models
        assert 'creative' in manager.runtime_models
    
    def test_get_default_model(self):
        """Test getting the default model."""
        config = {
            'default_model': 'primary',
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b',
                    'parameters': {'temperature': 0.7}
                }
            ]
        }
        
        manager = RuntimeManager(config)
        model = manager.get_model()  # No name specified, should return default
        
        assert model['name'] == 'primary'
        assert model['provider'] == 'ollama'
        assert model['model'] == 'llama3.1:8b'
    
    def test_get_model_by_name(self):
        """Test getting a specific model by name."""
        config = {
            'default_model': 'primary',
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                },
                {
                    'name': 'creative',
                    'provider': 'ollama',
                    'model': 'mixtral:8x7b',
                    'parameters': {'temperature': 0.9}
                }
            ]
        }
        
        manager = RuntimeManager(config)
        model = manager.get_model('creative')
        
        assert model['name'] == 'creative'
        assert model['model'] == 'mixtral:8x7b'
        assert model['parameters']['temperature'] == 0.9
    
    def test_get_model_not_found(self):
        """Test error when requesting non-existent model."""
        config = {
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                }
            ]
        }
        
        manager = RuntimeManager(config)
        
        with pytest.raises(ValueError) as excinfo:
            manager.get_model('nonexistent')
        
        assert "Model 'nonexistent' not found" in str(excinfo.value)
        assert "Available models: primary" in str(excinfo.value)
    
    def test_list_models(self):
        """Test listing all available models."""
        config = {
            'default_model': 'primary',
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                },
                {
                    'name': 'backup',
                    'provider': 'openai',
                    'model': 'gpt-4'
                }
            ]
        }
        
        manager = RuntimeManager(config)
        models = manager.list_models()
        
        assert len(models) == 2
        assert models[0]['name'] == 'primary'
        assert models[0]['is_default'] is True
        assert models[1]['name'] == 'backup'
        assert models[1]['is_default'] is False
    
    def test_set_default(self):
        """Test setting a new default model."""
        config = {
            'default_model': 'primary',
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                },
                {
                    'name': 'backup',
                    'provider': 'ollama',
                    'model': 'mistral:7b'
                }
            ]
        }
        
        manager = RuntimeManager(config)
        manager.set_default('backup')
        
        assert manager.default_model == 'backup'
    
    def test_set_default_invalid_model(self):
        """Test error when setting non-existent model as default."""
        config = {
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                }
            ]
        }
        
        manager = RuntimeManager(config)
        
        with pytest.raises(ValueError) as excinfo:
            manager.set_default('nonexistent')
        
        assert "Model 'nonexistent' not found" in str(excinfo.value)
    
    
    def test_to_runtime_config(self):
        """Test converting model to legacy runtime format."""
        config = {
            'default_model': 'primary',
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b',
                    'base_url': 'http://localhost:11434',
                    'instructor_mode': 'json',
                    'parameters': {
                        'temperature': 0.7,
                        'max_tokens': 2048
                    }
                }
            ]
        }
        
        manager = RuntimeManager(config)
        runtime = manager.to_runtime_config()
        
        assert runtime['provider'] == 'ollama'
        assert runtime['model'] == 'llama3.1:8b'
        assert runtime['base_url'] == 'http://localhost:11434'
        assert runtime['temperature'] == 0.7
        assert runtime['max_tokens'] == 2048
        assert runtime['instructor_mode'] == 'json'
    
    @patch('subprocess.run')
    def test_import_ollama_models(self, mock_run):
        """Test importing models from Ollama."""
        # Mock ollama list output
        mock_run.return_value = MagicMock(
            stdout="NAME                    ID              SIZE      MODIFIED\n"
                   "llama3.2:3b            abc123          2.0 GB    2 hours ago\n"
                   "mistral:7b             def456          4.1 GB    3 days ago\n",
            returncode=0
        )
        
        config = {
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                }
            ]
        }
        
        manager = RuntimeManager(config)
        imported = manager.import_ollama_models()
        
        assert len(imported) == 2
        assert 'llama3-2-3b' in imported
        assert 'mistral-7b' in imported
        
        # Check the models were added
        assert 'llama3-2-3b' in manager.runtime_models
        assert 'mistral-7b' in manager.runtime_models
        
        # Check model configuration
        llama_model = manager.runtime_models['llama3-2-3b']
        assert llama_model['provider'] == 'ollama'
        assert llama_model['model'] == 'llama3.2:3b'
    
    @patch('subprocess.run')
    def test_import_ollama_models_with_prefix(self, mock_run):
        """Test importing models with a custom prefix."""
        mock_run.return_value = MagicMock(
            stdout="NAME                    ID              SIZE      MODIFIED\n"
                   "llama3.2:3b            abc123          2.0 GB    2 hours ago\n",
            returncode=0
        )
        
        config = {'runtime_models': []}
        
        manager = RuntimeManager(config)
        imported = manager.import_ollama_models(prefix='ollama-')
        
        assert len(imported) == 1
        assert 'ollama-llama3-2-3b' in imported
        assert 'ollama-llama3-2-3b' in manager.runtime_models
    
    @patch('subprocess.run')
    def test_import_ollama_models_with_filter(self, mock_run):
        """Test importing models with filter patterns."""
        mock_run.return_value = MagicMock(
            stdout="NAME                    ID              SIZE      MODIFIED\n"
                   "llama3.2:3b            abc123          2.0 GB    2 hours ago\n"
                   "mistral:7b             def456          4.1 GB    3 days ago\n"
                   "codellama:13b          ghi789          7.3 GB    1 week ago\n",
            returncode=0
        )
        
        config = {'runtime_models': []}
        
        manager = RuntimeManager(config)
        imported = manager.import_ollama_models(filter_patterns=['llama3*', 'code*'])
        
        # Should only import models matching the patterns
        assert len(imported) == 2
        assert 'llama3-2-3b' in imported
        assert 'codellama-13b' in imported
        assert 'mistral-7b' not in imported
    
    @patch('subprocess.run')
    def test_import_ollama_models_skip_existing(self, mock_run):
        """Test that import skips already configured models."""
        mock_run.return_value = MagicMock(
            stdout="NAME                    ID              SIZE      MODIFIED\n"
                   "llama3.1:8b            abc123          4.7 GB    1 day ago\n",
            returncode=0
        )
        
        config = {
            'runtime_models': [
                {
                    'name': 'llama3-1-8b',  # Already exists
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                }
            ]
        }
        
        manager = RuntimeManager(config)
        imported = manager.import_ollama_models()
        
        # Should not import the existing model
        assert len(imported) == 0
    
    def test_validation_missing_provider(self):
        """Test validation error for missing provider."""
        config = {
            'runtime_models': [
                {
                    'name': 'invalid',
                    'model': 'llama3.1:8b'  # Missing provider
                }
            ]
        }
        
        with pytest.raises(ValueError) as excinfo:
            RuntimeManager(config)
        
        assert "missing required field: provider" in str(excinfo.value)
    
    def test_validation_missing_model(self):
        """Test validation error for missing model."""
        config = {
            'runtime_models': [
                {
                    'name': 'invalid',
                    'provider': 'ollama'  # Missing model
                }
            ]
        }
        
        with pytest.raises(ValueError) as excinfo:
            RuntimeManager(config)
        
        assert "missing required field: model" in str(excinfo.value)
    
    def test_validation_invalid_default(self):
        """Test validation error for invalid default_model reference."""
        config = {
            'default_model': 'nonexistent',
            'runtime_models': [
                {
                    'name': 'primary',
                    'provider': 'ollama',
                    'model': 'llama3.1:8b'
                }
            ]
        }
        
        with pytest.raises(ValueError) as excinfo:
            RuntimeManager(config)
        
        assert "default_model 'nonexistent' not found" in str(excinfo.value)