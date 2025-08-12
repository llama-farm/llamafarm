"""
Model Manager for coordinating all model operations.

This module provides a unified interface for model operations using
the new strategy-based configuration system.
"""

import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path

from components import (
    FineTunerFactory,
    ModelAppFactory,
    ModelRepositoryFactory,
    CloudAPIFactory
)
from core.strategy_manager import StrategyManager

logger = logging.getLogger(__name__)


class ModelManager:
    """Central manager for all model operations using strategies."""
    
    def __init__(self, strategy: Optional[str] = None, strategy_file: Optional[Path] = None):
        """Initialize Model Manager with a strategy.
        
        Args:
            strategy: Strategy name to use
            strategy_file: Custom strategy file path
        """
        self.strategy_manager = StrategyManager(strategy_file)
        
        # Set default strategy if not provided
        self.current_strategy = strategy or "local_development"
        
        # Validate strategy exists
        if self.current_strategy not in self.strategy_manager.list_strategies():
            available = ", ".join(self.strategy_manager.list_strategies())
            raise ValueError(f"Strategy '{self.current_strategy}' not found. Available: {available}")
        
        # Component instances
        self._components = {}
        self._fallback_chain = []
        
        # Load strategy configuration
        self._load_strategy()
    
    def _load_strategy(self):
        """Load the current strategy configuration."""
        strategy_config = self.strategy_manager.get_strategy(self.current_strategy)
        if not strategy_config:
            raise ValueError(f"Failed to load strategy: {self.current_strategy}")
        
        self.strategy_config = strategy_config
        self._fallback_chain = self.strategy_manager.get_fallback_chain(self.current_strategy)
    
    @classmethod
    def from_strategy(cls, strategy_name: str, overrides: Optional[Dict[str, Any]] = None):
        """Create ModelManager from a predefined strategy.
        
        Args:
            strategy_name: Name of the strategy to use
            overrides: Optional configuration overrides
        """
        manager = cls(strategy=strategy_name)
        
        if overrides:
            # Apply overrides to strategy config
            manager.strategy_config = manager.strategy_manager.merge_strategies(
                strategy_name, overrides
            )
        
        return manager
    
    @classmethod
    def for_use_case(cls, use_case: str):
        """Create ModelManager for a specific use case.
        
        Args:
            use_case: Use case name (e.g., 'chatbot', 'code_generation')
        """
        strategy_manager = StrategyManager()
        strategies = strategy_manager.get_strategies_for_use_case(use_case)
        
        if not strategies:
            raise ValueError(f"No strategies found for use case: {use_case}")
        
        # Use the first recommended strategy
        return cls(strategy=strategies[0])
    
    def get_component(self, component_type: str) -> Optional[Any]:
        """Get or create a component instance.
        
        Args:
            component_type: Type of component (cloud_api, model_app, fine_tuner, repository)
        """
        if component_type in self._components:
            return self._components[component_type]
        
        # Get component config from strategy
        component_config = self.strategy_manager.build_component_config(
            self.current_strategy, component_type
        )
        
        if not component_config:
            return None
        
        # Create component based on type
        component = None
        
        if component_type == "cloud_api":
            component = CloudAPIFactory.create(component_config)
        elif component_type == "model_app":
            component = ModelAppFactory.create(component_config)
        elif component_type == "fine_tuner":
            component = FineTunerFactory.create(component_config)
        elif component_type == "repository":
            component = ModelRepositoryFactory.create(component_config)
        
        if component:
            self._components[component_type] = component
        
        return component
    
    def get_cloud_api(self) -> Optional[Any]:
        """Get cloud API component."""
        return self.get_component("cloud_api")
    
    def get_model_app(self) -> Optional[Any]:
        """Get model app component."""
        return self.get_component("model_app")
    
    def get_fine_tuner(self) -> Optional[Any]:
        """Get fine-tuner component."""
        return self.get_component("fine_tuner")
    
    def get_repository(self) -> Optional[Any]:
        """Get repository component."""
        return self.get_component("repository")
    
    # Inference operations
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using configured model with fallback support.
        
        Args:
            prompt: Input prompt
            **kwargs: Additional generation parameters
        """
        # Try primary cloud API
        cloud_api = self.get_cloud_api()
        if cloud_api:
            try:
                return self._generate_with_component(cloud_api, prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Primary cloud API failed: {e}")
        
        # Try model app
        model_app = self.get_model_app()
        if model_app:
            try:
                return self._generate_with_component(model_app, prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Model app failed: {e}")
        
        # Try fallback chain
        for fallback in self._fallback_chain:
            try:
                component = self._create_fallback_component(fallback)
                if component:
                    return self._generate_with_component(component, prompt, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback {fallback.get('name', 'unnamed')} failed: {e}")
        
        raise RuntimeError("All model sources failed")
    
    def _generate_with_component(self, component: Any, prompt: str, **kwargs) -> str:
        """Generate text using a specific component.
        
        Args:
            component: Component instance
            prompt: Input prompt
            **kwargs: Additional parameters
        """
        # Apply strategy constraints
        constraints = self.strategy_manager.get_constraints(self.current_strategy)
        
        if "max_tokens" in constraints and "max_tokens" not in kwargs:
            kwargs["max_tokens"] = constraints["max_tokens"]
        
        # Apply optimization settings
        optimization = self.strategy_manager.get_optimization_config(self.current_strategy)
        
        if "timeout_seconds" in optimization and "timeout" not in kwargs:
            kwargs["timeout"] = optimization["timeout_seconds"]
        
        # Generate based on component type
        if hasattr(component, "generate"):
            return component.generate(prompt, **kwargs)
        elif hasattr(component, "chat"):
            messages = [{"role": "user", "content": prompt}]
            return component.chat(messages, **kwargs)
        else:
            raise ValueError(f"Component {type(component).__name__} doesn't support generation")
    
    def _create_fallback_component(self, fallback_config: Dict[str, Any]) -> Optional[Any]:
        """Create a component from fallback configuration.
        
        Args:
            fallback_config: Fallback component configuration
        """
        comp_type = fallback_config.get("type")
        config = fallback_config.get("config", {})
        
        if comp_type == "openai_compatible":
            return CloudAPIFactory.create("openai_compatible", config)
        elif comp_type == "ollama":
            return ModelAppFactory.create("ollama", config)
        
        return None
    
    # Fine-tuning operations
    def fine_tune(self, dataset_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
        """Start fine-tuning process.
        
        Args:
            dataset_path: Path to training dataset
            output_dir: Output directory for fine-tuned model
            **kwargs: Additional training parameters
        """
        fine_tuner = self.get_fine_tuner()
        if not fine_tuner:
            raise ValueError("No fine-tuner configured in strategy")
        
        # Apply strategy-specific training config
        training_config = self.strategy_config.get("components", {}).get("fine_tuner", {}).get("config", {})
        
        # Merge with provided kwargs
        merged_config = {**training_config, **kwargs}
        merged_config["dataset_path"] = dataset_path
        merged_config["output_dir"] = output_dir
        
        # Start training
        return fine_tuner.train(merged_config)
    
    # Model management
    def list_available_models(self) -> Dict[str, List[str]]:
        """List all available models from all sources.
        
        Returns:
            Dictionary mapping source to list of model names
        """
        models = {}
        
        # Cloud API models
        cloud_api = self.get_cloud_api()
        if cloud_api and hasattr(cloud_api, "list_models"):
            try:
                models["cloud_api"] = cloud_api.list_models()
            except Exception as e:
                logger.error(f"Failed to list cloud API models: {e}")
        
        # Model app models
        model_app = self.get_model_app()
        if model_app and hasattr(model_app, "list_models"):
            try:
                models["model_app"] = model_app.list_models()
            except Exception as e:
                logger.error(f"Failed to list model app models: {e}")
        
        # Repository models
        repository = self.get_repository()
        if repository and hasattr(repository, "list_models"):
            try:
                models["repository"] = repository.list_models()
            except Exception as e:
                logger.error(f"Failed to list repository models: {e}")
        
        return models
    
    def validate_configuration(self) -> List[str]:
        """Validate current configuration.
        
        Returns:
            List of validation errors (empty if valid)
        """
        return self.strategy_manager.validate_strategy(self.current_strategy)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about current configuration.
        
        Returns:
            Configuration information
        """
        return {
            "strategy": self.current_strategy,
            "strategy_description": self.strategy_config.get("description", ""),
            "components": {
                comp_type: comp_config.get("type") 
                for comp_type, comp_config in self.strategy_config.get("components", {}).items()
            },
            "fallback_count": len(self._fallback_chain),
            "optimization": self.strategy_manager.get_optimization_config(self.current_strategy),
            "monitoring": self.strategy_manager.get_monitoring_config(self.current_strategy),
            "constraints": self.strategy_manager.get_constraints(self.current_strategy)
        }
    
    def switch_strategy(self, strategy_name: str):
        """Switch to a different strategy.
        
        Args:
            strategy_name: Name of strategy to switch to
        """
        if strategy_name not in self.strategy_manager.list_strategies():
            raise ValueError(f"Strategy '{strategy_name}' not found")
        
        # Clear cached components
        self._components.clear()
        
        # Load new strategy
        self.current_strategy = strategy_name
        self._load_strategy()
        
        logger.info(f"Switched to strategy: {strategy_name}")