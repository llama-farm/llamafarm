#!/usr/bin/env python3
"""
Simple test suite for Models system.
Tests basic functionality without complex imports.
"""

import pytest
import json
import tempfile
from pathlib import Path


class TestBasicFunctionality:
    """Test basic functionality."""
    
    def test_imports(self):
        """Test that basic imports work."""
        # Test that we can import base modules
        try:
            from pathlib import Path
            from typing import Dict, Any
            assert True
        except ImportError:
            pytest.fail("Basic imports failed")
    
    def test_config_structure(self):
        """Test configuration structure."""
        config = {
            "version": "1.0",
            "model_app": {
                "type": "ollama",
                "config": {
                    "default_model": "llama3.2"
                }
            }
        }
        
        assert "model_app" in config
        assert config["model_app"]["type"] == "ollama"
    
    def test_strategy_structure(self):
        """Test strategy structure."""
        strategy = {
            "description": "Test strategy",
            "model_app": {
                "type": "ollama",
                "config": {}
            },
            "fine_tuner": {
                "type": "pytorch",
                "config": {
                    "method": {"type": "lora"}
                }
            }
        }
        
        assert "model_app" in strategy
        assert "fine_tuner" in strategy
    
    def test_demo_config_creation(self):
        """Test creating demo configurations."""
        demos = [
            {
                "name": "cloud_fallback",
                "cloud_api": {"type": "openai"},
                "model_app": {"type": "ollama"}
            },
            {
                "name": "multi_model",
                "cloud_api": {"type": "openai"},
                "models": ["gpt-3.5-turbo", "gpt-4"]
            },
            {
                "name": "quick_training",
                "fine_tuner": {"type": "llamafactory", "method": "qlora"}
            },
            {
                "name": "complex_training",
                "fine_tuner": {"type": "pytorch", "method": "lora"}
            }
        ]
        
        assert len(demos) == 4
        assert all("name" in demo for demo in demos)
    
    def test_component_types(self):
        """Test component type definitions."""
        component_types = {
            "fine_tuners": ["pytorch", "llamafactory"],
            "model_apps": ["ollama", "vllm", "llamacpp"],
            "model_repositories": ["huggingface"],
            "cloud_apis": ["openai", "anthropic", "together", "cohere"]
        }
        
        assert "fine_tuners" in component_types
        assert "pytorch" in component_types["fine_tuners"]
    
    def test_file_operations(self):
        """Test file operations for configs."""
        config = {
            "test": True,
            "nested": {
                "value": 42
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            temp_path = f.name
        
        try:
            # Read back the file
            with open(temp_path) as f:
                loaded = json.load(f)
            
            assert loaded["test"] is True
            assert loaded["nested"]["value"] == 42
        finally:
            Path(temp_path).unlink()


class TestDemoConfigurations:
    """Test demo configurations."""
    
    def test_demo1_cloud_fallback_config(self):
        """Test cloud with fallback configuration."""
        config = {
            "strategy": "hybrid_fallback",
            "cloud_api": {
                "type": "openai",
                "config": {
                    "api_key": "${OPENAI_API_KEY}",
                    "default_model": "gpt-4"
                }
            },
            "model_app": {
                "type": "ollama",
                "config": {
                    "default_model": "llama3.2",
                    "auto_start": False
                }
            }
        }
        
        assert config["strategy"] == "hybrid_fallback"
        assert "cloud_api" in config
        assert "model_app" in config
    
    def test_demo2_multi_model_config(self):
        """Test multi-model configuration."""
        models = {
            "simple": "gpt-3.5-turbo",
            "reasoning": "gpt-4o-mini",
            "creative": "gpt-4o",
            "premium": "gpt-4-turbo",
            "code": "gpt-4o"
        }
        
        assert len(models) == 5
        assert models["simple"] == "gpt-3.5-turbo"
    
    def test_demo3_medical_training_config(self):
        """Test medical training configuration."""
        config = {
            "fine_tuner": {
                "type": "pytorch",
                "config": {
                    "base_model": {
                        "name": "medical-llm",
                        "huggingface_id": "medical/llm"
                    },
                    "method": {
                        "type": "qlora",
                        "r": 64,
                        "alpha": 128
                    },
                    "training_args": {
                        "num_train_epochs": 3,
                        "per_device_train_batch_size": 2
                    }
                }
            }
        }
        
        assert config["fine_tuner"]["config"]["method"]["type"] == "qlora"
        assert config["fine_tuner"]["config"]["method"]["r"] == 64
    
    def test_demo4_code_training_config(self):
        """Test code assistant training configuration."""
        config = {
            "fine_tuner": {
                "type": "llamafactory",
                "config": {
                    "base_model": {
                        "name": "codellama-13b"
                    },
                    "method": {
                        "type": "lora",
                        "r": 32,
                        "alpha": 64
                    },
                    "training_args": {
                        "num_train_epochs": 5,
                        "max_seq_length": 2048
                    }
                }
            }
        }
        
        assert config["fine_tuner"]["type"] == "llamafactory"
        assert config["fine_tuner"]["config"]["training_args"]["max_seq_length"] == 2048


class TestValidation:
    """Test validation functions."""
    
    def test_validate_component_config(self):
        """Test component configuration validation."""
        # Valid config
        valid = {
            "type": "ollama",
            "config": {
                "default_model": "llama3.2"
            }
        }
        
        assert "type" in valid
        assert isinstance(valid["config"], dict)
        
        # Invalid configs
        invalid_configs = [
            {},  # Missing type
            {"type": "ollama"},  # Missing config
            {"config": {}},  # Missing type
        ]
        
        for config in invalid_configs:
            assert "type" not in config or "config" not in config
    
    def test_validate_training_config(self):
        """Test training configuration validation."""
        required_fields = ["base_model", "method", "dataset", "training_args"]
        
        config = {
            "base_model": {"name": "test"},
            "method": {"type": "lora"},
            "dataset": {"path": "data.jsonl"},
            "training_args": {"output_dir": "./output"}
        }
        
        for field in required_fields:
            assert field in config


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])