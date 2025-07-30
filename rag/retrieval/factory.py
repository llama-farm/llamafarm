"""Factory for creating retrieval strategies with database capability integration."""

from typing import Dict, Any, Optional
from .base import RetrievalStrategy
from .registry import get_registry


class RetrievalStrategyFactory:
    """Factory for creating retrieval strategy instances with database optimization."""
    
    def __init__(self):
        self._registry = get_registry()
    
    def create(
        self,
        strategy_type: str,
        config: Dict[str, Any] = None,
        database_type: Optional[str] = None,
        name: Optional[str] = None
    ) -> RetrievalStrategy:
        """Create a retrieval strategy instance with database optimization.
        
        Args:
            strategy_type: Type of strategy to create
            config: Configuration dictionary for the strategy
            database_type: Target database type for optimization
            name: Optional custom name for the strategy
            
        Returns:
            Configured retrieval strategy instance
            
        Raises:
            ValueError: If strategy_type is not available
        """
        # Use provided name or default to strategy type
        strategy_name = name or strategy_type
        
        # Create strategy with database optimization
        strategy = self._registry.create_strategy(
            strategy_name=strategy_type,
            config=config or {},
            database_type=database_type
        )
        
        # Override name if provided
        if name:
            strategy.name = name
        
        return strategy
    
    def create_optimal(
        self,
        database_type: str,
        use_case: str = "general",
        config: Dict[str, Any] = None
    ) -> RetrievalStrategy:
        """Create the optimal strategy for a database and use case.
        
        Args:
            database_type: Target database type
            use_case: Use case ("general", "simple", "filtered", "complex", "performance")
            config: Optional configuration override
            
        Returns:
            Optimal retrieval strategy instance
        """
        return self._registry.create_optimal_strategy(database_type, use_case, config)
    
    def list_strategies(self) -> Dict[str, str]:
        """List all available retrieval strategies.
        
        Returns:
            Dictionary mapping strategy names to their descriptions
        """
        return self._registry.list_all_strategies()
    
    def get_strategies_for_vector_store(self, vector_store_type: str) -> Dict[str, Dict[str, Any]]:
        """Get strategies compatible with a specific vector store type.
        
        Args:
            vector_store_type: Type of vector store (e.g., "ChromaStore")
            
        Returns:
            Dictionary of compatible strategies with compatibility information
        """
        return self._registry.get_compatible_strategies(vector_store_type)
    
    def validate_config(
        self,
        database_type: str,
        strategy_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a strategy configuration for a specific database.
        
        Args:
            database_type: Target database type
            strategy_type: Strategy type to validate
            config: Configuration to validate
            
        Returns:
            Validation result with errors, warnings, and recommendations
        """
        return self._registry.validate_strategy_config(database_type, strategy_type, config)
    
    def get_database_info(self, database_type: str) -> Dict[str, Any]:
        """Get comprehensive information about a database's capabilities.
        
        Args:
            database_type: Database type to get info for
            
        Returns:
            Database capability information
        """
        return self._registry.get_database_info(database_type)
    
    def register(self, strategy_type: str, strategy_class):
        """Register a new universal retrieval strategy type.
        
        Args:
            strategy_type: Name to register the strategy under
            strategy_class: Strategy class to register
        """
        self._registry.register_strategy(strategy_type, strategy_class)
    
    def register_database_specific(
        self,
        database_type: str,
        strategy_type: str,
        strategy_class
    ):
        """Register a database-specific strategy.
        
        Args:
            database_type: Database type this strategy is optimized for
            strategy_type: Name to register the strategy under
            strategy_class: Strategy class to register
        """
        self._registry.register_database_specific_strategy(
            database_type, strategy_type, strategy_class
        )


# Global factory instance
_factory = RetrievalStrategyFactory()


def create_retrieval_strategy_from_config(
    config: Dict[str, Any],
    database_type: Optional[str] = None
) -> RetrievalStrategy:
    """Create a retrieval strategy from configuration.
    
    Args:
        config: Configuration dictionary with 'type' and optional 'config' keys
        database_type: Target database type for optimization
        
    Returns:
        Configured retrieval strategy instance
        
    Example:
        config = {
            "type": "HybridUniversalStrategy",
            "config": {
                "strategies": [
                    {"type": "BasicSimilarityStrategy", "weight": 0.7},
                    {"type": "MetadataFilteredStrategy", "weight": 0.3}
                ]
            }
        }
        strategy = create_retrieval_strategy_from_config(config, "ChromaStore")
    """
    strategy_type = config.get("type")
    if not strategy_type:
        raise ValueError("Retrieval strategy config must include 'type' field")
    
    strategy_config = config.get("config", {})
    name = config.get("name")
    
    return _factory.create(
        strategy_type=strategy_type,
        config=strategy_config,
        database_type=database_type,
        name=name
    )


# Backward compatibility
class RetrievalStrategyFactory_Legacy:
    """Legacy factory interface for backward compatibility."""
    
    @classmethod
    def create(cls, strategy_type: str, config: Dict[str, Any] = None, name: str = None):
        return _factory.create(strategy_type, config, name=name)
    
    @classmethod
    def list_strategies(cls):
        return _factory.list_strategies()
    
    @classmethod
    def get_strategies_for_vector_store(cls, vector_store_type: str):
        return _factory.get_strategies_for_vector_store(vector_store_type)
    
    @classmethod
    def register(cls, strategy_type: str, strategy_class):
        return _factory.register(strategy_type, strategy_class)


# Maintain backward compatibility
RetrievalStrategyFactory = RetrievalStrategyFactory_Legacy