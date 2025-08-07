#!/usr/bin/env python3
"""Configuration loader for the prompts system with YAML support."""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to configuration file (can be relative or absolute)
        base_dir: Base directory for relative path resolution
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    if base_dir:
        base_path = Path(base_dir)
    else:
        base_path = Path.cwd()
    
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = base_path / config_file
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    try:
        with open(config_file, "r") as f:
            file_content = f.read()
        
        # Try to determine file type and parse accordingly
        if config_file.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                config = yaml.safe_load(file_content)
            except ImportError:
                raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
            except yaml.YAMLError as e:
                raise ValueError(f"Invalid YAML in config file: {e}")
        else:
            # Default to JSON
            try:
                config = json.loads(file_content)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in config file: {e}")
        
        if config is None:
            raise ValueError("Configuration file is empty or contains only null values")
        
        return config
        
    except Exception as e:
        if isinstance(e, (FileNotFoundError, ImportError, ValueError)):
            raise
        raise ValueError(f"Failed to load configuration: {str(e)}")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate basic configuration structure.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a dictionary")
    
    # Check for required top-level keys
    if "prompts" not in config:
        raise ValueError("Configuration must contain 'prompts' section")
    
    prompts_config = config["prompts"]
    if not isinstance(prompts_config, dict):
        raise ValueError("'prompts' section must be a dictionary")
    
    # Check for required prompts configuration
    if "strategy" not in prompts_config:
        raise ValueError("'prompts' section must contain 'strategy'")
    
    if "templates" not in prompts_config and "config" not in prompts_config:
        raise ValueError("'prompts' section must contain either 'templates' or 'config'")
    
    return True


def load_and_validate_config(config_path: str, base_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate configuration file.
    
    Args:
        config_path: Path to configuration file
        base_dir: Base directory for relative path resolution
        
    Returns:
        Validated configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file is invalid
    """
    config = load_config(config_path, base_dir)
    validate_config(config)
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to merge in (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result