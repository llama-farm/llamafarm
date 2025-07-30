"""Plugin registry system for retrieval strategies."""

from typing import Dict, List, Type, Any, Optional
from .base import RetrievalStrategy
from .strategies.universal import (
    BasicSimilarityStrategy,
    MetadataFilteredStrategy, 
    MultiQueryStrategy,
    RerankedStrategy,
    HybridUniversalStrategy
)
from stores.capabilities import DatabaseCapabilities


class StrategyRegistry:
    """Central registry for all retrieval strategies with database capability integration."""
    
    def __init__(self):
        self._strategies: Dict[str, Type[RetrievalStrategy]] = {}
        self._database_specific: Dict[str, Dict[str, Type[RetrievalStrategy]]] = {}
        self._initialize_core_strategies()
    
    def _initialize_core_strategies(self):
        """Initialize core universal strategies."""
        # Universal strategies (work with any database)
        self.register_strategy("BasicSimilarityStrategy", BasicSimilarityStrategy)
        self.register_strategy("MetadataFilteredStrategy", MetadataFilteredStrategy)
        self.register_strategy("MultiQueryStrategy", MultiQueryStrategy)
        self.register_strategy("RerankedStrategy", RerankedStrategy)
        self.register_strategy("HybridUniversalStrategy", HybridUniversalStrategy)
        
        # Aliases for convenience
        self.register_strategy("basic", BasicSimilarityStrategy)
        self.register_strategy("filtered", MetadataFilteredStrategy) 
        self.register_strategy("multi_query", MultiQueryStrategy)
        self.register_strategy("reranked", RerankedStrategy)
        self.register_strategy("hybrid", HybridUniversalStrategy)
        
        # Load database-specific strategies
        self._load_database_specific_strategies()
    
    def _load_database_specific_strategies(self):
        """Load database-specific optimized strategies."""
        try:
            # Import ChromaDB-specific strategies
            from .strategies.database_specific import (
                ChromaBasicStrategy,
                ChromaMetadataFilterStrategy,
                ChromaMultiQueryStrategy, 
                ChromaRerankedStrategy,
                ChromaHybridStrategy
            )
            
            # Register ChromaDB-specific strategies
            self.register_database_specific_strategy("ChromaStore", "ChromaBasicStrategy", ChromaBasicStrategy)
            self.register_database_specific_strategy("ChromaStore", "ChromaMetadataFilterStrategy", ChromaMetadataFilterStrategy)
            self.register_database_specific_strategy("ChromaStore", "ChromaMultiQueryStrategy", ChromaMultiQueryStrategy)
            self.register_database_specific_strategy("ChromaStore", "ChromaRerankedStrategy", ChromaRerankedStrategy)
            self.register_database_specific_strategy("ChromaStore", "ChromaHybridStrategy", ChromaHybridStrategy)
            
        except ImportError:
            # ChromaDB-specific strategies not available
            pass
    
    def register_strategy(self, name: str, strategy_class: Type[RetrievalStrategy]):
        """Register a universal retrieval strategy."""
        self._strategies[name] = strategy_class
    
    def register_database_specific_strategy(
        self, 
        database_type: str, 
        strategy_name: str, 
        strategy_class: Type[RetrievalStrategy]
    ):
        """Register a database-specific optimized strategy."""
        if database_type not in self._database_specific:
            self._database_specific[database_type] = {}
        self._database_specific[database_type][strategy_name] = strategy_class
    
    def get_strategy_class(
        self, 
        strategy_name: str, 
        database_type: Optional[str] = None,
        prefer_database_specific: bool = True
    ) -> Type[RetrievalStrategy]:
        """Get strategy class with database-specific optimization preference."""
        
        # First check for database-specific strategy if requested
        if database_type and prefer_database_specific:
            db_strategies = self._database_specific.get(database_type, {})
            if strategy_name in db_strategies:
                return db_strategies[strategy_name]
        
        # Fall back to universal strategy
        if strategy_name in self._strategies:
            return self._strategies[strategy_name]
        
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def create_strategy(
        self,
        strategy_name: str,
        config: Dict[str, Any] = None,
        database_type: Optional[str] = None,
        prefer_database_specific: bool = True
    ) -> RetrievalStrategy:
        """Create a strategy instance with optimal database selection."""
        
        strategy_class = self.get_strategy_class(
            strategy_name, 
            database_type, 
            prefer_database_specific
        )
        
        return strategy_class(name=strategy_name, config=config or {})
    
    def get_compatible_strategies(self, database_type: str) -> Dict[str, Dict[str, Any]]:
        """Get all strategies compatible with a database type."""
        return DatabaseCapabilities.get_supported_strategies(database_type)
    
    def get_optimal_strategy_name(self, database_type: str, use_case: str = "general") -> str:
        """Get the optimal strategy name for a database and use case."""
        return DatabaseCapabilities.get_optimal_strategy(database_type, use_case)
    
    def create_optimal_strategy(
        self,
        database_type: str,
        use_case: str = "general",
        config: Dict[str, Any] = None
    ) -> RetrievalStrategy:
        """Create the optimal strategy for a database and use case."""
        strategy_name = self.get_optimal_strategy_name(database_type, use_case)
        return self.create_strategy(strategy_name, config, database_type)
    
    def list_all_strategies(self) -> Dict[str, str]:
        """List all available strategies."""
        strategies = {}
        
        # Universal strategies
        for name, cls in self._strategies.items():
            strategies[name] = f"{cls.__name__} (Universal)"
        
        # Database-specific strategies
        for db_type, db_strategies in self._database_specific.items():
            for name, cls in db_strategies.items():
                strategies[f"{name}@{db_type}"] = f"{cls.__name__} ({db_type} optimized)"
        
        return strategies
    
    def validate_strategy_config(
        self,
        database_type: str,
        strategy_name: str, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate a strategy configuration for a specific database."""
        return DatabaseCapabilities.validate_configuration(database_type, strategy_name, config)
    
    def get_database_info(self, database_type: str) -> Dict[str, Any]:
        """Get comprehensive database capability information."""
        return DatabaseCapabilities.get_database_info(database_type)


# Global registry instance
_registry = StrategyRegistry()


def get_registry() -> StrategyRegistry:
    """Get the global strategy registry."""
    return _registry


def register_strategy(name: str, strategy_class: Type[RetrievalStrategy]):
    """Register a new universal strategy (convenience function)."""
    _registry.register_strategy(name, strategy_class)


def register_database_strategy(
    database_type: str,
    strategy_name: str, 
    strategy_class: Type[RetrievalStrategy]
):
    """Register a database-specific strategy (convenience function)."""
    _registry.register_database_specific_strategy(database_type, strategy_name, strategy_class)


def create_strategy(
    strategy_name: str,
    config: Dict[str, Any] = None,
    database_type: Optional[str] = None
) -> RetrievalStrategy:
    """Create a strategy instance (convenience function)."""
    return _registry.create_strategy(strategy_name, config, database_type)


def get_compatible_strategies(database_type: str) -> Dict[str, Dict[str, Any]]:
    """Get compatible strategies for a database (convenience function)."""
    return _registry.get_compatible_strategies(database_type)