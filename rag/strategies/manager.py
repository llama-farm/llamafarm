"""
Strategy Manager

High-level interface for managing and applying RAG strategies.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config import StrategyConfig
from .loader import StrategyLoader

logger = logging.getLogger(__name__)


class StrategyManager:
    """Manages RAG strategies and converts them to standard configurations."""
    
    def __init__(self, strategies_file: Optional[str] = None):
        """
        Initialize strategy manager.
        
        Args:
            strategies_file: Path to strategies YAML file
        """
        self.loader = StrategyLoader(strategies_file)
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return self.loader.list_strategies()
    
    def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy information or None if not found
        """
        strategy = self.loader.get_strategy(strategy_name)
        if not strategy:
            return None
        
        return {
            "name": strategy.name,
            "description": strategy.description,
            "use_cases": strategy.use_cases,
            "performance_priority": strategy.performance_priority.value,
            "resource_usage": strategy.resource_usage.value,
            "complexity": strategy.complexity.value,
            "components": {
                "parser": strategy.components.parser.type,
                "extractors": [e.type for e in strategy.components.extractors],
                "embedder": strategy.components.embedder.type,
                "vector_store": strategy.components.vector_store.type,
                "retrieval_strategy": strategy.components.retrieval_strategy.type
            }
        }
    
    def convert_strategy_to_config(self, 
                                 strategy_name: str, 
                                 overrides: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Convert a strategy to a standard RAG configuration.
        
        Args:
            strategy_name: Name of the strategy to use
            overrides: Optional configuration overrides
            
        Returns:
            Standard RAG configuration dictionary or None if strategy not found
        """
        strategy = self.loader.get_strategy(strategy_name)
        if not strategy:
            logger.error(f"Strategy '{strategy_name}' not found")
            return None
        
        # Apply overrides if provided
        if overrides:
            strategy = strategy.apply_overrides(overrides)
        
        # Convert to standard configuration format
        config = {
            "parser": {
                "type": strategy.components.parser.type,
                "config": strategy.components.parser.config
            },
            "embedder": {
                "type": strategy.components.embedder.type,
                "config": strategy.components.embedder.config
            },
            "vector_store": {
                "type": strategy.components.vector_store.type,
                "config": strategy.components.vector_store.config
            },
            "retrieval_strategy": {
                "type": strategy.components.retrieval_strategy.type,
                "config": strategy.components.retrieval_strategy.config
            }
        }
        
        # Add extractors if any
        if strategy.components.extractors:
            config["extractors"] = [
                {
                    "type": extractor.type,
                    "config": extractor.config
                }
                for extractor in strategy.components.extractors
            ]
        
        # Add metadata about the strategy
        config["_strategy_info"] = {
            "name": strategy.name,
            "description": strategy.description,
            "use_cases": strategy.use_cases,
            "performance_priority": strategy.performance_priority.value,
            "resource_usage": strategy.resource_usage.value,
            "complexity": strategy.complexity.value
        }
        
        return config
    
    def recommend_strategies(self, **criteria) -> List[Dict[str, Any]]:
        """
        Recommend strategies based on criteria.
        
        Args:
            **criteria: Criteria for recommendation (use_case, performance_priority, etc.)
            
        Returns:
            List of strategy information dictionaries
        """
        recommendations = self.loader.recommend_strategy(**criteria)
        return [
            {
                "name": strategy.name,
                "description": strategy.description,
                "use_cases": strategy.use_cases,
                "performance_priority": strategy.performance_priority.value,
                "resource_usage": strategy.resource_usage.value,
                "complexity": strategy.complexity.value,
                "components": {
                    "parser": strategy.components.parser.type,
                    "extractors": [e.type for e in strategy.components.extractors],
                    "embedder": strategy.components.embedder.type,
                    "vector_store": strategy.components.vector_store.type,
                    "retrieval_strategy": strategy.components.retrieval_strategy.type
                }
            }
            for strategy in recommendations
        ]
    
    def create_custom_strategy(self,
                             name: str,
                             description: str,
                             base_strategy: Optional[str] = None,
                             components: Optional[Dict[str, Any]] = None,
                             **metadata) -> Dict[str, Any]:
        """
        Create a custom strategy configuration.
        
        Args:
            name: Custom strategy name
            description: Strategy description
            base_strategy: Base strategy to extend (optional)
            components: Component configurations
            **metadata: Additional metadata (use_cases, performance_priority, etc.)
            
        Returns:
            Strategy configuration dictionary
        """
        if base_strategy:
            # Start with base strategy
            base_config = self.convert_strategy_to_config(base_strategy)
            if not base_config:
                raise ValueError(f"Base strategy '{base_strategy}' not found")
            
            # Remove strategy metadata
            base_config.pop("_strategy_info", None)
        else:
            # Start with minimal configuration
            base_config = {
                "parser": {"type": "CSVParser", "config": {}},
                "embedder": {"type": "OllamaEmbedder", "config": {}},
                "vector_store": {"type": "ChromaStore", "config": {}},
                "retrieval_strategy": {"type": "BasicSimilarityStrategy", "config": {}}
            }
        
        # Apply component overrides
        if components:
            for component_type, component_config in components.items():
                if component_type in base_config:
                    if isinstance(component_config, dict) and "type" in component_config:
                        base_config[component_type] = component_config
                    else:
                        # Assume it's just a type string
                        base_config[component_type] = {
                            "type": component_config,
                            "config": base_config[component_type].get("config", {})
                        }
        
        # Add custom strategy metadata
        base_config["_strategy_info"] = {
            "name": name,
            "description": description,
            "use_cases": metadata.get("use_cases", []),
            "performance_priority": metadata.get("performance_priority", "balanced"),
            "resource_usage": metadata.get("resource_usage", "medium"),
            "complexity": metadata.get("complexity", "moderate"),
            "custom": True
        }
        
        return base_config
    
    def validate_strategy_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a strategy configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            List of validation errors
        """
        errors = []
        
        # Check required components
        required_components = ["parser", "embedder", "vector_store", "retrieval_strategy"]
        for component in required_components:
            if component not in config:
                errors.append(f"Missing required component: {component}")
                continue
            
            if not isinstance(config[component], dict):
                errors.append(f"Component {component} must be a dictionary")
                continue
            
            if "type" not in config[component]:
                errors.append(f"Component {component} must have a 'type' field")
        
        # Validate extractors if present
        if "extractors" in config:
            if not isinstance(config["extractors"], list):
                errors.append("Extractors must be a list")
            else:
                for i, extractor in enumerate(config["extractors"]):
                    if not isinstance(extractor, dict):
                        errors.append(f"Extractor {i} must be a dictionary")
                    elif "type" not in extractor:
                        errors.append(f"Extractor {i} must have a 'type' field")
        
        return errors
    
    def print_strategy_summary(self, strategy_name: str) -> None:
        """Print a summary of a strategy."""
        info = self.get_strategy_info(strategy_name)
        if not info:
            print(f"Strategy '{strategy_name}' not found")
            return
        
        print(f"\n=== {info['name']} ===")
        print(f"Description: {info['description']}")
        print(f"Use Cases: {', '.join(info['use_cases'])}")
        print(f"Performance Priority: {info['performance_priority']}")
        print(f"Resource Usage: {info['resource_usage']}")
        print(f"Complexity: {info['complexity']}")
        print("\nComponents:")
        print(f"  Parser: {info['components']['parser']}")
        if info['components']['extractors']:
            print(f"  Extractors: {', '.join(info['components']['extractors'])}")
        print(f"  Embedder: {info['components']['embedder']}")
        print(f"  Vector Store: {info['components']['vector_store']}")
        print(f"  Retrieval Strategy: {info['components']['retrieval_strategy']}")
        print()
    
    def print_all_strategies(self) -> None:
        """Print a summary of all available strategies."""
        strategies = self.get_available_strategies()
        if not strategies:
            print("No strategies available")
            return
        
        print(f"\n=== Available Strategies ({len(strategies)}) ===\n")
        
        for strategy_name in sorted(strategies):
            info = self.get_strategy_info(strategy_name)
            if info:
                print(f"{strategy_name:20} - {info['description']}")
                print(f"{'':20}   Use cases: {', '.join(info['use_cases'])}")
                print(f"{'':20}   {info['performance_priority']} performance, {info['resource_usage']} resources, {info['complexity']} complexity")
                print()
    
    def export_strategy_as_config(self, strategy_name: str, output_file: str) -> bool:
        """
        Export a strategy as a standalone configuration file.
        
        Args:
            strategy_name: Name of the strategy to export
            output_file: Path to output configuration file
            
        Returns:
            True if successful, False otherwise
        """
        import yaml
        
        config = self.convert_strategy_to_config(strategy_name)
        if not config:
            return False
        
        try:
            # Add header comment
            header = f"""# RAG Configuration generated from strategy: {strategy_name}
# Strategy description: {config.get('_strategy_info', {}).get('description', 'N/A')}
# Generated automatically - modify as needed

"""
            
            with open(output_file, 'w') as f:
                f.write(header)
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"Strategy '{strategy_name}' exported to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export strategy: {e}")
            return False