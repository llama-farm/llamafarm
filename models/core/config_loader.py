"""
Configuration loader for the models system.

Handles loading, validation, and processing of configuration files.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Handles configuration loading and processing."""
    
    def __init__(self, default_config_path: Optional[Path] = None):
        """Initialize configuration loader.
        
        Args:
            default_config_path: Path to default configuration file
        """
        if default_config_path:
            self.default_config_path = Path(default_config_path)
        else:
            # Default to models/strategies/default.yaml
            self.default_config_path = Path(__file__).parent.parent / "strategies" / "default.yaml"
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Loaded configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Determine file type and load
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path) as f:
                config = yaml.safe_load(f) or {}
        elif config_path.suffix == '.json':
            with open(config_path) as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {config_path.suffix}")
        
        # Process environment variables
        config = self._process_env_vars(config)
        
        # Validate configuration
        self._validate_config(config)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        if self.default_config_path.exists():
            return self.load_config(self.default_config_path)
        
        # Return minimal default if file doesn't exist
        return {
            "version": "1.0",
            "model_app": {
                "type": "ollama",
                "config": {
                    "default_model": "llama3.2"
                }
            }
        }
    
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
    
    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        # Check for at least one component
        components = ["fine_tuner", "model_app", "repository", "cloud_api"]
        if not any(comp in config for comp in components) and "strategy" not in config:
            logger.warning("Configuration contains no components or strategy")
        
        # Validate component structures
        for comp in components:
            if comp in config:
                if not isinstance(config[comp], dict):
                    raise ValueError(f"{comp} must be a dictionary")
                if "type" not in config[comp]:
                    raise ValueError(f"{comp} must specify a type")
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations.
        
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
                result[key] = self.merge_configs(result[key], value)
            else:
                # Direct override
                result[key] = value
        
        return result
    
    def save_config(self, config: Dict[str, Any], path: Union[str, Path], 
                   format: str = "yaml") -> None:
        """Save configuration to file.
        
        Args:
            config: Configuration to save
            path: Output path
            format: Output format (yaml or json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            with open(path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Saved configuration to {path}")