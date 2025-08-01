"""Base agent for configuration management."""

import json
import subprocess
import tempfile
import yaml
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional

from llama_brain.config import get_settings
from llama_brain.integrations import LlamaFarmClient


class BaseAgent(ABC):
    """Base class for configuration agents."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.settings = get_settings()
        self.client = LlamaFarmClient()
        
    @abstractmethod
    async def create_config(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new configuration based on requirements."""
        pass
        
    @abstractmethod
    async def edit_config(self, config: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Edit an existing configuration."""
        pass
        
    @abstractmethod
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a configuration and return validation results."""
        pass
    
    def get_examples_path(self) -> Path:
        """Get the path to example configurations."""
        return self.settings.llamafarm_root / self.component_name / "config_examples"
    
    def get_config_path(self) -> Path:
        """Get the path to default configurations."""
        return self.settings.llamafarm_root / self.component_name / "config"
    
    async def load_example_config(self, example_name: str) -> Optional[Dict[str, Any]]:
        """Load an example configuration."""
        examples_path = self.get_examples_path()
        
        # Try different file extensions
        for ext in ['.yaml', '.yml', '.json']:
            config_file = examples_path / f"{example_name}{ext}"
            if config_file.exists():
                return await self._load_config_file(config_file)
        
        return None
    
    async def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from a file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix in ['.yaml', '.yml']:
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load config from {config_path}: {e}")
    
    async def save_config(self, config: Dict[str, Any], filename: str, format: str = "yaml") -> Path:
        """Save configuration to a file."""
        output_dir = self.settings.generated_configs_dir / self.component_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "yaml":
            config_path = output_dir / f"{filename}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            config_path = output_dir / f"{filename}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config_path
    
    def get_available_examples(self) -> List[str]:
        """Get list of available example configurations."""
        examples_path = self.get_examples_path()
        if not examples_path.exists():
            return []
        
        examples = []
        for config_file in examples_path.glob("*.{yaml,yml,json}"):
            examples.append(config_file.stem)
        
        return sorted(examples)
    
    def _merge_configs(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        result = base.copy()
        
        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result