"""
Strategy Loader

Loads strategy configurations from YAML files and validates them.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

from .config import StrategyConfig

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Loads and manages strategy configurations."""
    
    def __init__(self, strategies_file: Optional[str] = None):
        """
        Initialize strategy loader.
        
        Args:
            strategies_file: Path to strategies YAML file. If None, uses default.
        """
        if strategies_file is None:
            # Default to default_strategies.yaml in the same directory as this script
            self.strategies_file = Path(__file__).parent.parent / "default_strategies.yaml"
        else:
            self.strategies_file = Path(strategies_file)
        
        self._strategies: Dict[str, StrategyConfig] = {}
        self._loaded = False
    
    def load_strategies(self) -> Dict[str, StrategyConfig]:
        """
        Load all strategies from the YAML file.
        
        Returns:
            Dictionary mapping strategy names to StrategyConfig objects.
        """
        if self._loaded:
            return self._strategies
        
        try:
            if not self.strategies_file.exists():
                logger.error(f"Strategies file not found: {self.strategies_file}")
                return {}
            
            with open(self.strategies_file, 'r') as file:
                data = yaml.safe_load(file)
            
            # Skip metadata fields like usage_notes
            strategy_names = [key for key in data.keys() if not key.startswith('usage_')]
            
            for strategy_name in strategy_names:
                strategy_data = data[strategy_name]
                
                # Skip if this is not a complete strategy definition
                if not isinstance(strategy_data, dict) or "components" not in strategy_data:
                    continue
                
                try:
                    strategy_config = StrategyConfig.from_dict(strategy_data)
                    self._strategies[strategy_name] = strategy_config
                    logger.debug(f"Loaded strategy: {strategy_name}")
                except Exception as e:
                    logger.error(f"Failed to load strategy {strategy_name}: {e}")
                    continue
            
            self._loaded = True
            logger.info(f"Loaded {len(self._strategies)} strategies from {self.strategies_file}")
            
        except Exception as e:
            logger.error(f"Failed to load strategies file {self.strategies_file}: {e}")
            
        return self._strategies
    
    def get_strategy(self, name: str) -> Optional[StrategyConfig]:
        """
        Get a specific strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            StrategyConfig object or None if not found
        """
        strategies = self.load_strategies()
        return strategies.get(name)
    
    def list_strategies(self) -> List[str]:
        """
        List all available strategy names.
        
        Returns:
            List of strategy names
        """
        strategies = self.load_strategies()
        return list(strategies.keys())
    
    def get_strategies_by_use_case(self, use_case: str) -> List[StrategyConfig]:
        """
        Get strategies that support a specific use case.
        
        Args:
            use_case: Use case to search for
            
        Returns:
            List of matching StrategyConfig objects
        """
        strategies = self.load_strategies()
        matching = []
        
        for strategy in strategies.values():
            if use_case in strategy.use_cases:
                matching.append(strategy)
        
        return matching
    
    def get_strategies_by_performance(self, performance_priority: str) -> List[StrategyConfig]:
        """
        Get strategies with a specific performance priority.
        
        Args:
            performance_priority: "speed", "accuracy", or "balanced"
            
        Returns:
            List of matching StrategyConfig objects
        """
        strategies = self.load_strategies()
        matching = []
        
        for strategy in strategies.values():
            if strategy.performance_priority.value == performance_priority:
                matching.append(strategy)
        
        return matching
    
    def get_strategies_by_complexity(self, complexity: str) -> List[StrategyConfig]:
        """
        Get strategies with a specific complexity level.
        
        Args:
            complexity: "simple", "moderate", or "complex"
            
        Returns:
            List of matching StrategyConfig objects
        """
        strategies = self.load_strategies()
        matching = []
        
        for strategy in strategies.values():
            if strategy.complexity.value == complexity:
                matching.append(strategy)
        
        return matching
    
    def recommend_strategy(self, 
                         use_case: Optional[str] = None,
                         performance_priority: Optional[str] = None,
                         resource_usage: Optional[str] = None,
                         complexity: Optional[str] = None) -> List[StrategyConfig]:
        """
        Recommend strategies based on criteria.
        
        Args:
            use_case: Desired use case
            performance_priority: "speed", "accuracy", or "balanced"
            resource_usage: "low", "medium", or "high"
            complexity: "simple", "moderate", or "complex"
            
        Returns:
            List of recommended StrategyConfig objects, sorted by relevance
        """
        strategies = list(self.load_strategies().values())
        scores = []
        
        for strategy in strategies:
            score = 0
            
            # Score based on use case match
            if use_case and use_case in strategy.use_cases:
                score += 10
            
            # Score based on performance priority match
            if performance_priority and strategy.performance_priority.value == performance_priority:
                score += 5
            
            # Score based on resource usage match
            if resource_usage and strategy.resource_usage.value == resource_usage:
                score += 3
            
            # Score based on complexity match
            if complexity and strategy.complexity.value == complexity:
                score += 2
            
            scores.append((strategy, score))
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return strategies with score > 0, or top 3 if no matches
        recommendations = [strategy for strategy, score in scores if score > 0]
        if not recommendations:
            recommendations = [strategy for strategy, score in scores[:3]]
        
        return recommendations
    
    def validate_strategy(self, strategy_config: StrategyConfig) -> List[str]:
        """
        Validate a strategy configuration.
        
        Args:
            strategy_config: Strategy to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation
        if not strategy_config.name:
            errors.append("Strategy name is required")
        
        if not strategy_config.description:
            errors.append("Strategy description is required")
        
        if not strategy_config.components.parser.type:
            errors.append("Parser type is required")
        
        if not strategy_config.components.embedder.type:
            errors.append("Embedder type is required")
        
        if not strategy_config.components.vector_store.type:
            errors.append("Vector store type is required")
        
        if not strategy_config.components.retrieval_strategy.type:
            errors.append("Retrieval strategy type is required")
        
        # TODO: Add more sophisticated validation
        # - Check if component types exist
        # - Validate configuration parameters
        # - Check component compatibility
        
        return errors