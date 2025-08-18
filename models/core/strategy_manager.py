"""
Strategy Manager for model system configurations.

This module manages strategies defined in the new RAG-style schema format.
"""

import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages strategies for model operations using the new schema format."""
    
    def __init__(self, strategies_file: Optional[Path] = None):
        """Initialize strategy manager.
        
        Args:
            strategies_file: Path to strategies YAML file
        """
        if strategies_file:
            self.strategies_file = Path(strategies_file)
        else:
            # Default to models/default_strategies.yaml
            self.strategies_file = Path(__file__).parent.parent / "default_strategies.yaml"
        
        # Load strategies
        self.strategies = self._load_strategies()
        self.use_case_mapping = self._load_use_case_mapping()
    
    def _load_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load strategies from YAML file."""
        if not self.strategies_file.exists():
            logger.warning(f"Strategies file not found: {self.strategies_file}")
            return {}
        
        try:
            with open(self.strategies_file) as f:
                data = yaml.safe_load(f) or {}
                strategies_list = data.get("strategies", [])
                
                # Convert array of strategies to dict keyed by name
                strategies_dict = {}
                for strategy in strategies_list:
                    if isinstance(strategy, dict) and "name" in strategy:
                        strategies_dict[strategy["name"]] = strategy
                    else:
                        logger.warning(f"Invalid strategy format: {strategy}")
                
                return strategies_dict
        except Exception as e:
            logger.error(f"Failed to load strategies: {e}")
            return {}
    
    def _load_use_case_mapping(self) -> Dict[str, List[str]]:
        """Load use case to strategy mappings."""
        if not self.strategies_file.exists():
            return {}
        
        try:
            with open(self.strategies_file) as f:
                data = yaml.safe_load(f) or {}
                return data.get("use_case_mapping", {})
        except Exception as e:
            logger.error(f"Failed to load use case mappings: {e}")
            return {}
    
    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a specific strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Strategy configuration or None if not found
        """
        return deepcopy(self.strategies.get(name))
    
    def list_strategies(self) -> List[str]:
        """List all available strategy names.
        
        Returns:
            List of strategy names
        """
        return list(self.strategies.keys())
    
    def get_strategies_for_use_case(self, use_case: str) -> List[str]:
        """Get recommended strategies for a use case.
        
        Args:
            use_case: Use case name
            
        Returns:
            List of recommended strategy names
        """
        return self.use_case_mapping.get(use_case, [])
    
    def build_component_config(self, strategy_name: str, component_type: str) -> Optional[Dict[str, Any]]:
        """Build component configuration from strategy.
        
        Args:
            strategy_name: Strategy name
            component_type: Component type (cloud_api, model_app, fine_tuner, repository)
            
        Returns:
            Component configuration or None
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return None
        
        components = strategy.get("components", {})
        component_config = components.get(component_type)
        
        if component_config:
            # Expand environment variables in config
            return self._expand_env_vars(component_config)
        
        return None
    
    def get_fallback_chain(self, strategy_name: str) -> List[Dict[str, Any]]:
        """Get fallback chain for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            List of fallback configurations
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return []
        
        fallback_chain = strategy.get("fallback_chain", [])
        return [self._expand_env_vars(fb) for fb in fallback_chain]
    
    def get_optimization_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get optimization configuration for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Optimization configuration
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {}
        
        return strategy.get("optimization", {})
    
    def get_monitoring_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get monitoring configuration for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Monitoring configuration
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {}
        
        return strategy.get("monitoring", {})
    
    def get_constraints(self, strategy_name: str) -> Dict[str, Any]:
        """Get constraints for a strategy.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            Constraints configuration
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return {}
        
        return strategy.get("constraints", {})
    
    def validate_strategy(self, strategy_name: str) -> List[str]:
        """Validate a strategy configuration.
        
        Args:
            strategy_name: Strategy name
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return [f"Strategy '{strategy_name}' not found"]
        
        # Check required fields
        if "name" not in strategy:
            errors.append("Strategy missing 'name' field")
        if "description" not in strategy:
            errors.append("Strategy missing 'description' field")
        if "components" not in strategy:
            errors.append("Strategy missing 'components' field")
        
        # Validate component types
        components = strategy.get("components", {})
        valid_component_types = ["cloud_api", "model_app", "fine_tuner", "repository"]
        
        for comp_type, comp_config in components.items():
            if comp_type not in valid_component_types:
                errors.append(f"Invalid component type: {comp_type}")
            
            if not isinstance(comp_config, dict):
                errors.append(f"Component {comp_type} configuration must be a dictionary")
            elif "type" not in comp_config:
                errors.append(f"Component {comp_type} missing 'type' field")
        
        # Validate fallback chain
        fallback_chain = strategy.get("fallback_chain", [])
        for i, fallback in enumerate(fallback_chain):
            if "type" not in fallback:
                errors.append(f"Fallback {i} missing 'type' field")
            if "config" not in fallback:
                errors.append(f"Fallback {i} missing 'config' field")
        
        return errors
    
    def _expand_env_vars(self, config: Union[Dict, List, str, Any]) -> Union[Dict, List, str, Any]:
        """Recursively expand environment variables in configuration.
        
        Args:
            config: Configuration to expand
            
        Returns:
            Configuration with expanded environment variables
        """
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            # Expand ${VAR_NAME} format
            if config.startswith("${") and config.endswith("}"):
                var_name = config[2:-1]
                return os.getenv(var_name, config)
            return config
        else:
            return config
    
    def merge_strategies(self, base_strategy: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Merge a base strategy with custom overrides.
        
        Args:
            base_strategy: Base strategy name
            overrides: Custom configuration overrides
            
        Returns:
            Merged strategy configuration
        """
        base = self.get_strategy(base_strategy)
        if not base:
            return overrides
        
        # Deep merge overrides into base
        merged = deepcopy(base)
        self._deep_merge(merged, overrides)
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> None:
        """Deep merge override dictionary into base dictionary.
        
        Args:
            base: Base dictionary (modified in place)
            override: Override dictionary
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def export_strategy(self, strategy_name: str, output_path: Path, format: str = "yaml") -> bool:
        """Export a strategy to a file.
        
        Args:
            strategy_name: Strategy name
            output_path: Output file path
            format: Output format (yaml or json)
            
        Returns:
            True if successful
        """
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Strategy '{strategy_name}' not found")
            return False
        
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == "json":
                with open(output_path, "w") as f:
                    json.dump(strategy, f, indent=2)
            else:  # yaml
                with open(output_path, "w") as f:
                    yaml.dump(strategy, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Exported strategy '{strategy_name}' to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export strategy: {e}")
            return False
    
    def import_strategy(self, name: str, config_path: Path) -> bool:
        """Import a strategy from a file.
        
        Args:
            name: Strategy name to use
            config_path: Path to configuration file
            
        Returns:
            True if successful
        """
        try:
            config_path = Path(config_path)
            
            if config_path.suffix == ".json":
                with open(config_path) as f:
                    config = json.load(f)
            else:  # yaml
                with open(config_path) as f:
                    config = yaml.safe_load(f)
            
            # Add to strategies
            self.strategies[name] = config
            
            logger.info(f"Imported strategy '{name}' from {config_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import strategy: {e}")
            return False