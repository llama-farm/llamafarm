"""Factory classes for creating model components."""

import json
from pathlib import Path
from typing import Dict, Any, Type, Optional, List
import logging

from .base import BaseFineTuner, BaseModelApp, BaseModelRepository, BaseCloudAPI

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Base factory class for component creation."""
    
    _registry: Dict[str, Type] = {}
    _component_type: str = "component"
    
    @classmethod
    def register(cls, name: str, component_class: Type) -> None:
        """Register a component type."""
        cls._registry[name] = component_class
        logger.info(f"Registered {cls._component_type} type: {name}")
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> Any:
        """Create component from configuration."""
        component_type = config.get("type")
        if not component_type:
            raise ValueError(f"{cls._component_type} type not specified in config")
        
        if component_type not in cls._registry:
            raise ValueError(f"Unknown {cls._component_type} type: {component_type}")
        
        component_class = cls._registry[component_type]
        component_config = config.get("config", {})
        
        # Load defaults if available
        defaults = cls._load_defaults(component_type)
        if defaults:
            # Merge defaults with provided config
            merged_config = {**defaults, **component_config}
        else:
            merged_config = component_config
        
        return component_class(merged_config)
    
    @classmethod
    def list_components(cls) -> List[str]:
        """List available component types."""
        return list(cls._registry.keys())
    
    @classmethod
    def get_schema(cls, component_type: str) -> Optional[Dict[str, Any]]:
        """Get schema for a component type."""
        schema_path = cls._get_component_path(component_type) / "schema.json"
        if schema_path.exists():
            with open(schema_path) as f:
                return json.load(f)
        return None
    
    @classmethod
    def _load_defaults(cls, component_type: str) -> Optional[Dict[str, Any]]:
        """Load default configuration for a component type."""
        defaults_path = cls._get_component_path(component_type) / "defaults.json"
        if defaults_path.exists():
            with open(defaults_path) as f:
                return json.load(f)
        return None
    
    @classmethod
    def _get_component_path(cls, component_type: str) -> Path:
        """Get the directory path for a component type."""
        # This should be overridden in subclasses
        return Path(__file__).parent / cls._component_type / component_type


class FineTunerFactory(ComponentFactory):
    """Factory for creating fine-tuner instances."""
    
    _registry: Dict[str, Type[BaseFineTuner]] = {}
    _component_type = "fine_tuners"
    
    @classmethod
    def _get_component_path(cls, component_type: str) -> Path:
        """Get the directory path for a fine-tuner type."""
        return Path(__file__).parent / "fine_tuners" / component_type


class ModelAppFactory(ComponentFactory):
    """Factory for creating model app instances."""
    
    _registry: Dict[str, Type[BaseModelApp]] = {}
    _component_type = "model_apps"
    
    @classmethod
    def _get_component_path(cls, component_type: str) -> Path:
        """Get the directory path for a model app type."""
        return Path(__file__).parent / "model_apps" / component_type


class ModelRepositoryFactory(ComponentFactory):
    """Factory for creating model repository instances."""
    
    _registry: Dict[str, Type[BaseModelRepository]] = {}
    _component_type = "model_repositories"
    
    @classmethod
    def _get_component_path(cls, component_type: str) -> Path:
        """Get the directory path for a repository type."""
        return Path(__file__).parent / "model_repositories" / component_type


class CloudAPIFactory(ComponentFactory):
    """Factory for creating cloud API instances."""
    
    _registry: Dict[str, Type[BaseCloudAPI]] = {}
    _component_type = "cloud_apis"
    
    @classmethod
    def _get_component_path(cls, component_type: str) -> Path:
        """Get the directory path for a cloud API type."""
        return Path(__file__).parent / "cloud_apis" / component_type


# Auto-register components
def auto_register_components():
    """Automatically register all available components."""
    import importlib
    import pkgutil
    
    # Get the components directory
    components_dir = Path(__file__).parent
    
    # Map of subdirectories to their factories
    factory_map = {
        "fine_tuners": (FineTunerFactory, BaseFineTuner),
        "model_apps": (ModelAppFactory, BaseModelApp),
        "model_repositories": (ModelRepositoryFactory, BaseModelRepository),
        "cloud_apis": (CloudAPIFactory, BaseCloudAPI)
    }
    
    for subdir, (factory, base_class) in factory_map.items():
        subdir_path = components_dir / subdir
        if not subdir_path.exists():
            continue
        
        # Find all component directories
        for item in subdir_path.iterdir():
            if item.is_dir() and not item.name.startswith("_"):
                try:
                    # Try to import the module
                    module_name = f"models.components.{subdir}.{item.name}"
                    module = importlib.import_module(module_name)
                    
                    # Find the component class
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            issubclass(attr, base_class) and 
                            attr is not base_class):
                            # Register the component
                            factory.register(item.name, attr)
                            break
                            
                except Exception as e:
                    logger.warning(f"Failed to auto-register {item.name}: {e}")


# Auto-register on import
try:
    auto_register_components()
except Exception as e:
    logger.error(f"Failed to auto-register components: {e}")