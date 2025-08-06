"""
Strategy Manager for model system configurations.

This module manages predefined strategies and configurations for different
use cases and scenarios.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages predefined strategies for model operations."""
    
    def __init__(self, strategies_dir: Optional[Path] = None):
        """Initialize strategy manager.
        
        Args:
            strategies_dir: Directory containing strategy files
        """
        if strategies_dir:
            self.strategies_dir = Path(strategies_dir)
        else:
            # Default to models/strategies directory
            self.strategies_dir = Path(__file__).parent.parent / "strategies"
        
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
        
        # Load default strategies
        self._default_strategies = self._load_default_strategies()
    
    def _load_default_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load default strategies from file."""
        default_strategies_file = self.strategies_dir.parent / "default_strategies.yaml"
        
        if default_strategies_file.exists():
            with open(default_strategies_file) as f:
                return yaml.safe_load(f) or {}
        
        # Return hardcoded defaults if file doesn't exist
        return {
            "local_development": {
                "description": "Local development with Ollama",
                "model_app": {
                    "type": "ollama",
                    "config": {
                        "default_model": "llama3.2:3b",
                        "auto_start": True
                    }
                }
            },
            "cloud_production": {
                "description": "Production with OpenAI API",
                "cloud_api": {
                    "type": "openai",
                    "config": {
                        "default_model": "gpt-3.5-turbo",
                        "timeout": 60
                    }
                }
            },
            "fine_tuning_lora": {
                "description": "LoRA fine-tuning with PyTorch",
                "fine_tuner": {
                    "type": "pytorch",
                    "config": {
                        "method": {
                            "type": "lora",
                            "r": 16,
                            "alpha": 32
                        },
                        "training_args": {
                            "num_train_epochs": 3,
                            "per_device_train_batch_size": 4
                        }
                    }
                }
            },
            "hybrid_with_fallback": {
                "description": "Cloud API with local fallback",
                "cloud_api": {
                    "type": "openai",
                    "config": {
                        "api_key": "${OPENAI_API_KEY}",
                        "default_model": "gpt-3.5-turbo"
                    }
                },
                "model_app": {
                    "type": "ollama",
                    "config": {
                        "default_model": "llama3.2:3b",
                        "auto_start": False
                    }
                }
            }
        }
    
    def list_strategies(self) -> List[Dict[str, Any]]:
        """List all available strategies."""
        strategies = []
        
        # Add default strategies
        for name, config in self._default_strategies.items():
            strategies.append({
                "name": name,
                "description": config.get("description", ""),
                "source": "default",
                "components": list(k for k in config.keys() if k != "description")
            })
        
        # Add custom strategies from directory
        for file in self.strategies_dir.glob("*.yaml"):
            try:
                with open(file) as f:
                    strategy = yaml.safe_load(f)
                    if strategy:
                        strategies.append({
                            "name": file.stem,
                            "description": strategy.get("description", ""),
                            "source": "custom",
                            "components": list(k for k in strategy.keys() if k != "description")
                        })
            except Exception as e:
                logger.warning(f"Failed to load strategy {file}: {e}")
        
        return strategies
    
    def get_strategy_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy information dict or None if not found
        """
        # Check default strategies first
        if name in self._default_strategies:
            strategy = self._default_strategies[name]
            return {
                "name": name,
                "description": strategy.get("description", ""),
                "source": "default",
                "components": list(k for k in strategy.keys() if k != "description"),
                "use_cases": self._infer_use_cases(strategy),
                "hardware_requirements": self._infer_hardware_requirements(strategy),
                "resource_usage": self._infer_resource_usage(strategy),
                "complexity": self._infer_complexity(strategy)
            }
        
        # Check custom strategies
        strategy_file = self.strategies_dir / f"{name}.yaml"
        if strategy_file.exists():
            try:
                with open(strategy_file) as f:
                    strategy = yaml.safe_load(f)
                    return {
                        "name": name,
                        "description": strategy.get("description", ""),
                        "source": "custom",
                        "components": list(k for k in strategy.keys() if k != "description"),
                        "use_cases": self._infer_use_cases(strategy),
                        "hardware_requirements": self._infer_hardware_requirements(strategy),
                        "resource_usage": self._infer_resource_usage(strategy),
                        "complexity": self._infer_complexity(strategy)
                    }
            except Exception as e:
                logger.error(f"Failed to load strategy {name}: {e}")
        
        return None
    
    def _infer_use_cases(self, strategy: Dict[str, Any]) -> List[str]:
        """Infer use cases from strategy configuration."""
        use_cases = []
        if "fine_tuner" in strategy:
            use_cases.append("model fine-tuning")
            if "medical" in str(strategy).lower():
                use_cases.append("medical AI")
            if "code" in str(strategy).lower():
                use_cases.append("code generation")
        if "cloud_api" in strategy:
            use_cases.append("cloud deployment")
        if "model_app" in strategy:
            use_cases.append("local deployment")
        return use_cases
    
    def _infer_hardware_requirements(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Infer hardware requirements from strategy configuration."""
        reqs = {"type": "Unknown", "minimum_ram": "Unknown"}
        
        if "fine_tuner" in strategy:
            config = strategy["fine_tuner"].get("config", {})
            device = config.get("training_args", {}).get("device", "")
            
            if device == "mps":
                reqs = {"type": "M1/M2 Mac", "minimum_ram": "8GB"}
            elif device == "cuda":
                reqs = {"type": "NVIDIA GPU", "minimum_ram": "16GB"}
            elif device == "cpu":
                reqs = {"type": "CPU only", "minimum_ram": "4GB"}
        
        return reqs
    
    def _infer_resource_usage(self, strategy: Dict[str, Any]) -> str:
        """Infer resource usage from strategy configuration."""
        if "fine_tuner" in strategy:
            config = strategy["fine_tuner"].get("config", {})
            batch_size = config.get("training_args", {}).get("per_device_train_batch_size", 1)
            if batch_size == 1:
                return "Low"
            elif batch_size <= 4:
                return "Medium"
            else:
                return "High"
        return "Unknown"
    
    def _infer_complexity(self, strategy: Dict[str, Any]) -> str:
        """Infer complexity from strategy configuration."""
        component_count = len([k for k in strategy.keys() if k != "description"])
        if component_count <= 2:
            return "Simple"
        elif component_count <= 4:
            return "Medium"
        else:
            return "Complex"
    
    def load_strategy(self, name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load a strategy configuration.
        
        Args:
            name: Strategy name
            overrides: Optional configuration overrides
            
        Returns:
            Complete configuration dict
        """
        # Check default strategies first
        if name in self._default_strategies:
            config = self._default_strategies[name].copy()
        else:
            # Check custom strategies
            strategy_file = self.strategies_dir / f"{name}.yaml"
            if not strategy_file.exists():
                raise ValueError(f"Strategy not found: {name}")
            
            with open(strategy_file) as f:
                config = yaml.safe_load(f) or {}
        
        # Remove description from config
        config.pop("description", None)
        
        # Apply overrides
        if overrides:
            config = self._merge_configs(config, overrides)
        
        # Process environment variables
        config = self._process_env_vars(config)
        
        # Add strategy metadata
        config["strategy"] = name
        
        return config
    
    def save_strategy(self, name: str, config: Dict[str, Any], 
                     description: Optional[str] = None) -> bool:
        """Save a custom strategy.
        
        Args:
            name: Strategy name
            config: Strategy configuration
            description: Optional description
            
        Returns:
            Success status
        """
        try:
            strategy_file = self.strategies_dir / f"{name}.yaml"
            
            # Add description if provided
            save_config = config.copy()
            if description:
                save_config["description"] = description
            
            with open(strategy_file, "w") as f:
                yaml.dump(save_config, f, default_flow_style=False)
            
            logger.info(f"Saved strategy: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save strategy: {e}")
            return False
    
    def delete_strategy(self, name: str) -> bool:
        """Delete a custom strategy.
        
        Args:
            name: Strategy name
            
        Returns:
            Success status
        """
        # Can't delete default strategies
        if name in self._default_strategies:
            logger.error(f"Cannot delete default strategy: {name}")
            return False
        
        strategy_file = self.strategies_dir / f"{name}.yaml"
        if not strategy_file.exists():
            logger.error(f"Strategy not found: {name}")
            return False
        
        try:
            strategy_file.unlink()
            logger.info(f"Deleted strategy: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete strategy: {e}")
            return False
    
    def validate_strategy(self, config: Dict[str, Any]) -> List[str]:
        """Validate a strategy configuration.
        
        Args:
            config: Strategy configuration
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check for at least one component
        components = ["fine_tuner", "model_app", "repository", "cloud_api"]
        if not any(comp in config for comp in components):
            errors.append("Strategy must contain at least one component")
        
        # Validate component configurations
        for comp in components:
            if comp in config:
                if not isinstance(config[comp], dict):
                    errors.append(f"{comp} must be a dictionary")
                elif "type" not in config[comp]:
                    errors.append(f"{comp} must specify a type")
        
        return errors
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two configurations.
        
        Args:
            base: Base configuration
            override: Override configuration
            
        Returns:
            Merged configuration
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursive merge for nested dicts
                result[key] = self._merge_configs(result[key], value)
            else:
                # Direct override
                result[key] = value
        
        return result
    
    def export_strategy(self, name: str, format: str = "yaml") -> str:
        """Export a strategy to string format.
        
        Args:
            name: Strategy name
            format: Export format (yaml or json)
            
        Returns:
            Exported strategy string
        """
        config = self.load_strategy(name)
        
        if format == "json":
            return json.dumps(config, indent=2)
        else:
            return yaml.dump(config, default_flow_style=False)
    
    def import_strategy(self, name: str, data: str, format: str = "yaml") -> bool:
        """Import a strategy from string format.
        
        Args:
            name: Strategy name
            data: Strategy data
            format: Data format (yaml or json)
            
        Returns:
            Success status
        """
        try:
            if format == "json":
                config = json.loads(data)
            else:
                config = yaml.safe_load(data)
            
            # Validate before saving
            errors = self.validate_strategy(config)
            if errors:
                logger.error(f"Strategy validation failed: {errors}")
                return False
            
            return self.save_strategy(name, config)
            
        except Exception as e:
            logger.error(f"Failed to import strategy: {e}")
            return False
    
    def _process_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Process environment variables in configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with environment variables resolved
        """
        def process_value(value):
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract environment variable name
                env_var = value[2:-1]
                # Get value from environment or return original if not found
                return os.getenv(env_var, value)
            elif isinstance(value, dict):
                return {k: process_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [process_value(item) for item in value]
            else:
                return value
        
        return process_value(config)
    
    def load_model_catalog(self) -> Dict[str, Any]:
        """Load model catalog with fallback chains."""
        catalog_file = self.strategies_dir.parent / "model_catalog.yaml"
        
        if catalog_file.exists():
            with open(catalog_file) as f:
                return yaml.safe_load(f) or {}
        
        return {}
    
    def get_fallback_chain(self, chain_name: str) -> List[str]:
        """Get fallback chain for a given use case.
        
        Args:
            chain_name: Name of the fallback chain
            
        Returns:
            List of model names in fallback order
        """
        catalog = self.load_model_catalog()
        fallback_chains = catalog.get("fallback_chains", {})
        
        if chain_name in fallback_chains:
            chain = fallback_chains[chain_name]
            models = [chain["primary"]] + chain.get("fallbacks", [])
            return models
        
        return []
    
    def get_models_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get models by category from catalog.
        
        Args:
            category: Model category (e.g., 'medical', 'code_generation')
            
        Returns:
            List of model information dictionaries
        """
        catalog = self.load_model_catalog()
        categories = catalog.get("categories", {})
        
        return categories.get(category, [])