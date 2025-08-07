"""
Tests for fine-tuning strategy management.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch

from ..core.strategies import StrategyManager, Strategy
from ..core.base import FineTuningConfig


class TestStrategyManager:
    """Test StrategyManager class."""
    
    def test_load_strategies_from_file(self):
        """Test loading strategies from YAML file."""
        # Create temporary strategy file
        strategies_data = {
            "strategies": {
                "test_strategy": {
                    "name": "Test Strategy",
                    "description": "A test strategy",
                    "use_cases": ["testing"],
                    "hardware_requirements": {
                        "type": "cpu",
                        "memory_gb": 8
                    },
                    "components": {
                        "base_model": {"name": "test-model"},
                        "method": {"type": "lora"},
                        "framework": {"type": "pytorch"},
                        "training_args": {"output_dir": "./test"},
                        "dataset": {"path": "./test.jsonl"}
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(strategies_data, f)
            temp_file = Path(f.name)
        
        try:
            # Test loading
            manager = StrategyManager(temp_file)
            strategies = manager.list_strategies()
            
            assert "test_strategy" in strategies
            assert len(strategies) == 1
            
            # Test getting strategy info
            info = manager.get_strategy_info("test_strategy")
            assert info is not None
            assert info["name"] == "Test Strategy"
            assert info["description"] == "A test strategy"
            assert "testing" in info["use_cases"]
            
        finally:
            temp_file.unlink()
    
    def test_load_strategies_missing_file(self):
        """Test loading strategies when file doesn't exist."""
        nonexistent_file = Path("/nonexistent/strategies.yaml")
        manager = StrategyManager(nonexistent_file)
        
        strategies = manager.list_strategies()
        assert strategies == []
    
    def test_get_strategy_config(self):
        """Test converting strategy to configuration."""
        # Create temporary strategy file
        strategies_data = {
            "strategies": {
                "test_strategy": {
                    "components": {
                        "base_model": {"name": "test-model"},
                        "method": {"type": "lora", "r": 16},
                        "framework": {"type": "pytorch"},
                        "training_args": {
                            "output_dir": "./test",
                            "num_train_epochs": 3
                        },
                        "dataset": {"path": "./test.jsonl"}
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(strategies_data, f)
            temp_file = Path(f.name)
        
        try:
            manager = StrategyManager(temp_file)
            
            # Get config without overrides
            config = manager.get_strategy_config("test_strategy")
            assert config is not None
            assert config["base_model"]["name"] == "test-model"
            assert config["method"]["type"] == "lora"
            assert config["method"]["r"] == 16
            
            # Get config with overrides
            overrides = {
                "method": {"r": 32},
                "training_args": {"num_train_epochs": 5}
            }
            config = manager.get_strategy_config("test_strategy", overrides)
            assert config["method"]["r"] == 32  # Overridden
            assert config["method"]["type"] == "lora"  # Original
            assert config["training_args"]["num_train_epochs"] == 5  # Overridden
            assert config["training_args"]["output_dir"] == "./test"  # Original
            
            # Test non-existent strategy
            config = manager.get_strategy_config("nonexistent")
            assert config is None
            
        finally:
            temp_file.unlink()
    
    def test_recommend_strategies(self):
        """Test strategy recommendations."""
        # Create temporary strategy file with multiple strategies
        strategies_data = {
            "strategies": {
                "mac_strategy": {
                    "name": "Mac Strategy",
                    "description": "For Mac users",
                    "use_cases": ["coding", "personal"],
                    "hardware_requirements": {
                        "type": "mac",
                        "memory_gb": 16
                    },
                    "recommended_models": ["llama3.2-3b", "llama3.1-8b"]
                },
                "gpu_strategy": {
                    "name": "GPU Strategy", 
                    "description": "For GPU users",
                    "use_cases": ["production", "research"],
                    "hardware_requirements": {
                        "type": "gpu",
                        "memory_gb": 24
                    },
                    "recommended_models": ["llama3.1-8b", "llama3.1-70b"]
                },
                "cpu_strategy": {
                    "name": "CPU Strategy",
                    "description": "For CPU-only users",
                    "use_cases": ["testing", "experimentation"],
                    "hardware_requirements": {
                        "type": "cpu",
                        "memory_gb": 8
                    },
                    "recommended_models": ["llama3.2-3b"]
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(strategies_data, f)
            temp_file = Path(f.name)
        
        try:
            manager = StrategyManager(temp_file)
            
            # Test hardware-based recommendation
            recommendations = manager.recommend_strategies(hardware="mac")
            assert len(recommendations) > 0
            assert recommendations[0]["strategy_id"] == "mac_strategy"  # Should be top match
            assert recommendations[0]["name"] == "Mac Strategy"  # Strategy name should be preserved
            
            # Test model size-based recommendation
            recommendations = manager.recommend_strategies(model_size="3b")
            mac_found = any(r["name"] == "Mac Strategy" for r in recommendations)
            cpu_found = any(r["name"] == "CPU Strategy" for r in recommendations)
            assert mac_found or cpu_found  # Both support 3b models
            
            # Test use case-based recommendation
            recommendations = manager.recommend_strategies(use_case="coding")
            assert any(r["name"] == "Mac Strategy" for r in recommendations)
            
            # Test combined criteria
            recommendations = manager.recommend_strategies(
                hardware="gpu",
                use_case="production"
            )
            assert len(recommendations) > 0
            assert recommendations[0]["name"] == "GPU Strategy"  # Should be perfect match
            
            # Test no matches
            recommendations = manager.recommend_strategies(hardware="quantum")
            # Should return empty list or low-scored matches
            
        finally:
            temp_file.unlink()
    
    def test_merge_configs(self):
        """Test configuration merging."""
        manager = StrategyManager(Path("/nonexistent"))  # Won't load any strategies
        
        base_config = {
            "base_model": {"name": "llama3.2-3b"},
            "method": {"type": "lora", "r": 16, "alpha": 32},
            "training_args": {
                "output_dir": "./test",
                "num_train_epochs": 3,
                "learning_rate": 2e-4
            }
        }
        
        overrides = {
            "method": {"r": 32},  # Override r but keep other method params
            "training_args": {
                "num_train_epochs": 5,  # Override epochs
                "batch_size": 4  # Add new param
            },
            "new_section": {"param": "value"}  # Add new section
        }
        
        merged = manager._merge_configs(base_config, overrides)
        
        # Check overrides applied
        assert merged["method"]["r"] == 32
        assert merged["training_args"]["num_train_epochs"] == 5
        assert merged["training_args"]["batch_size"] == 4
        assert merged["new_section"]["param"] == "value"
        
        # Check original values preserved
        assert merged["method"]["type"] == "lora"
        assert merged["method"]["alpha"] == 32
        assert merged["training_args"]["output_dir"] == "./test"
        assert merged["training_args"]["learning_rate"] == 2e-4
        assert merged["base_model"]["name"] == "llama3.2-3b"


class TestStrategy:
    """Test Strategy class."""
    
    def test_strategy_to_config(self):
        """Test converting strategy to FineTuningConfig."""
        strategy_config = {
            "name": "Test Strategy",
            "components": {
                "base_model": {"name": "llama3.2-3b"},
                "method": {"type": "lora", "r": 16},
                "framework": {"type": "pytorch"},
                "training_args": {"output_dir": "./test"},
                "dataset": {"path": "./test.jsonl"}
            }
        }
        
        strategy = Strategy(strategy_config)
        config = strategy.to_config()
        
        assert isinstance(config, FineTuningConfig)
        assert config.base_model["name"] == "llama3.2-3b"
        assert config.method["type"] == "lora"
        assert config.method["r"] == 16
    
    def test_strategy_to_config_with_overrides(self):
        """Test converting strategy to config with overrides."""
        strategy_config = {
            "name": "Test Strategy",
            "components": {
                "base_model": {"name": "llama3.2-3b"},
                "method": {"type": "lora", "r": 16},
                "framework": {"type": "pytorch"},
                "training_args": {"output_dir": "./test"},
                "dataset": {"path": "./test.jsonl"}
            }
        }
        
        overrides = {
            "method": {"r": 32},
            "training_args": {"num_train_epochs": 5}
        }
        
        strategy = Strategy(strategy_config)
        config = strategy.to_config(overrides)
        
        assert config.method["r"] == 32  # Overridden
        assert config.method["type"] == "lora"  # Original
        assert config.training_args.get("num_train_epochs") == 5  # Added
    
    def test_strategy_validate_hardware(self):
        """Test strategy hardware validation."""
        # High memory requirement
        strategy_config = {
            "name": "High Memory Strategy",
            "hardware_requirements": {
                "type": "gpu",
                "memory_gb": 128  # Very high
            }
        }
        
        strategy = Strategy(strategy_config)
        errors = strategy.validate_hardware()
        
        assert any("memory" in error.lower() for error in errors)
        
        # Many GPUs requirement
        strategy_config = {
            "name": "Many GPU Strategy",
            "hardware_requirements": {
                "type": "cloud",
                "gpu_count": 16  # Very high
            }
        }
        
        strategy = Strategy(strategy_config)
        errors = strategy.validate_hardware()
        
        assert any("gpu" in error.lower() for error in errors)
        
        # Normal requirements
        strategy_config = {
            "name": "Normal Strategy",
            "hardware_requirements": {
                "type": "mac",
                "memory_gb": 16
            }
        }
        
        strategy = Strategy(strategy_config)
        errors = strategy.validate_hardware()
        
        assert len(errors) == 0  # No errors for normal requirements
    
    def test_get_strategy_info(self):
        """Test getting strategy information."""
        strategy_config = {
            "name": "Test Strategy",
            "description": "A test strategy for unit tests",
            "use_cases": ["testing", "development"],
            "hardware_requirements": {
                "type": "cpu",
                "memory_gb": 8
            },
            "performance_priority": "speed",
            "resource_usage": "low",
            "complexity": "simple",
            "training_time_estimate": "1-2 hours"
        }
        
        strategy = Strategy(strategy_config)
        info = strategy.get_strategy_info()
        
        assert info["name"] == "Test Strategy"
        assert info["description"] == "A test strategy for unit tests"
        assert info["use_cases"] == ["testing", "development"]
        assert info["performance_priority"] == "speed"
        assert info["resource_usage"] == "low"
        assert info["complexity"] == "simple"
        assert info["training_time_estimate"] == "1-2 hours"


if __name__ == "__main__":
    pytest.main([__file__])