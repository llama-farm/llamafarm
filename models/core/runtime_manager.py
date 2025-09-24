"""
Runtime Manager for multi-model configurations.

This module manages runtime model configurations, allowing easy switching
between different models and their parameters.
"""

import subprocess
import json
import logging
import hashlib
import fnmatch
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy

logger = logging.getLogger(__name__)


class RuntimeManager:
    """Manages runtime model configurations for multi-model support."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RuntimeManager with configuration.
        
        Args:
            config: Configuration dictionary containing runtime_models and default_model
        """
        self.config = config
        self.runtime_models = self._load_runtime_models()
        self.default_model = config.get('default_model')
        self.safe_name_to_tag = {}  # Track safe name to original tag mapping
        
        # Validate configuration
        self._validate_config()
    
    def _load_runtime_models(self) -> Dict[str, Dict[str, Any]]:
        """Load runtime models into a dictionary keyed by name.
        
        Returns:
            Dictionary mapping model names to configurations
        """
        models = {}
        for model_config in self.config.get('runtime_models', []):
            name = model_config.get('name')
            if name:
                models[name] = model_config
        return models
    
    def _validate_config(self):
        """Validate the runtime configuration."""
        # Validate default_model references an existing model if specified
        if self.default_model and self.default_model not in self.runtime_models:
            raise ValueError(f"default_model '{self.default_model}' not found in runtime_models")
        
        # Ensure unique model names (already handled by dict structure)
        # Validate required fields for each model
        for name, model in self.runtime_models.items():
            if 'provider' not in model:
                raise ValueError(f"Model '{name}' missing required field: provider")
            if 'model' not in model:
                raise ValueError(f"Model '{name}' missing required field: model")
    
    
    def get_model(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get a model configuration by name.
        
        Args:
            name: Model name. If None, returns default model.
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ValueError: If model not found
        """
        if name is None:
            name = self.default_model
            
        if name is None:
            # No default specified, use first model
            if self.runtime_models:
                name = next(iter(self.runtime_models))
            else:
                raise ValueError("No models configured and no default specified")
        
        if name not in self.runtime_models:
            available = ', '.join(self.runtime_models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available}")
        
        return deepcopy(self.runtime_models[name])
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all available models.
        
        Returns:
            List of model summaries with name, provider, model, and default status
        """
        models = []
        for name, config in self.runtime_models.items():
            models.append({
                'name': name,
                'provider': config.get('provider'),
                'model': config.get('model'),
                'is_default': name == self.default_model
            })
        return models
    
    def set_default(self, name: str) -> None:
        """Set the default model.
        
        Args:
            name: Name of model to set as default
            
        Raises:
            ValueError: If model not found
        """
        if name not in self.runtime_models:
            raise ValueError(f"Model '{name}' not found")
        self.default_model = name
    
    def _generate_safe_name(self, model_tag: str, prefix: str = None) -> str:
        """Generate a safe name for a model tag with collision detection.
        
        Args:
            model_tag: Original model tag (e.g., "llama3.1:8b")
            prefix: Optional prefix to add
            
        Returns:
            Safe name that avoids collisions
            
        Raises:
            ValueError: If a collision is detected with different model tags
        """
        # Basic replacement
        base = model_tag.replace(':', '-').replace('.', '-')
        
        # Add hash to ensure uniqueness
        tag_hash = hashlib.sha256(model_tag.encode('utf-8')).hexdigest()[:8]
        safe_name = f"{base}-{tag_hash}"
        
        if prefix:
            safe_name = f"{prefix}{safe_name}"
        
        # Check for collisions
        if safe_name in self.safe_name_to_tag:
            if self.safe_name_to_tag[safe_name] != model_tag:
                logger.error(f"Safe name collision detected for '{model_tag}' and '{self.safe_name_to_tag[safe_name]}' as '{safe_name}'")
                raise ValueError(f"Safe name collision detected for '{model_tag}' and '{self.safe_name_to_tag[safe_name]}' as '{safe_name}'")
        else:
            self.safe_name_to_tag[safe_name] = model_tag
        
        return safe_name
    
    def import_ollama_models(self, prefix: str = "", filter_patterns: List[str] = None) -> List[str]:
        """Import models from Ollama.
        
        Args:
            prefix: Prefix to add to model names
            filter_patterns: List of patterns to filter models (e.g., ["llama*", "mistral*"])
            
        Returns:
            List of imported model names
        """
        imported = []
        
        try:
            # Run ollama list to get available models
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return imported
            
            # Skip header line
            for line in lines[1:]:
                parts = line.split()
                if not parts:
                    continue
                    
                model_tag = parts[0]
                
                # Apply filters if specified
                if filter_patterns:
                    matched = any(
                        fnmatch.fnmatch(model_tag, pattern) 
                        for pattern in filter_patterns
                    )
                    if not matched:
                        continue
                
                # Generate safe name with collision detection
                safe_name = self._generate_safe_name(model_tag, prefix)
                
                # Skip if already configured
                if safe_name in self.runtime_models:
                    logger.info(f"Model '{model_tag}' already configured as '{safe_name}'")
                    continue
                
                # Determine sensible defaults based on model type
                temperature = 0.7  # default
                if 'code' in model_tag.lower() or 'codellama' in model_tag.lower():
                    temperature = 0.3
                elif 'creative' in model_tag.lower() or 'story' in model_tag.lower():
                    temperature = 0.9
                
                # Add model configuration
                self.runtime_models[safe_name] = {
                    'name': safe_name,
                    'provider': 'ollama',
                    'model': model_tag,
                    'base_url': 'http://localhost:11434/v1',
                    'instructor_mode': 'json',
                    'parameters': {
                        'temperature': temperature,
                        'max_tokens': 2048
                    }
                }
                
                imported.append(safe_name)
                logger.info(f"Imported model '{model_tag}' as '{safe_name}'")
                
        except subprocess.CalledProcessError as e:
            error_details = e.stderr.decode() if hasattr(e, "stderr") and e.stderr else "No stderr output."
            logger.error(
                f"Failed to run 'ollama list': {e}\n"
                f"Stderr: {error_details}\n"
                "Troubleshooting steps:\n"
                "- Ensure Ollama is installed and accessible in your PATH.\n"
                "- Verify you have permission to run Ollama commands.\n"
                "- Make sure Ollama is running with 'ollama serve'."
            )
            raise RuntimeError(
                "Cannot connect to Ollama. Please check the following:\n"
                "- Ollama is installed and available in your PATH.\n"
                "- You have permission to run Ollama commands.\n"
                "- Ollama is running with 'ollama serve'.\n"
                f"Stderr output: {error_details}"
            ) from e
        except Exception as e:
            logger.error(f"Error importing Ollama models: {e}")
            raise
        
        return imported
    
    def to_runtime_config(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Convert a model configuration to legacy runtime format for backward compatibility.
        
        Args:
            model_name: Name of model to convert. If None, uses default.
            
        Returns:
            Runtime configuration in legacy format
        """
        model = self.get_model(model_name)
        
        # Extract parameters
        params = model.get('parameters', {})
        
        runtime = {
            'provider': model.get('provider'),
            'model': model.get('model'),
            'base_url': model.get('base_url'),
            'api_key': model.get('api_key'),
            'instructor_mode': model.get('instructor_mode', 'json'),
        }
        
        # Add temperature at top level for backward compatibility
        if 'temperature' in params:
            runtime['temperature'] = params['temperature']
        
        # Add other parameters
        for key in ['max_tokens', 'top_p', 'top_k', 'presence_penalty', 'frequency_penalty']:
            if key in params:
                runtime[key] = params[key]
        
        # Remove None values
        runtime = {k: v for k, v in runtime.items() if v is not None}
        
        return runtime