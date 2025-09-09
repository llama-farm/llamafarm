"""
Strategy Loader - New Schema Only

Loads strategy configurations from YAML files using only the new RAG schema format.
NO BACKWARD COMPATIBILITY - everything uses new schema directly.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class StrategyLoader:
    """Loads and manages strategy configurations using new RAG schema only."""
    
    def __init__(self, strategies_file: Optional[str] = None):
        """
        Initialize strategy loader.
        
        Args:
            strategies_file: Path to strategies YAML file. If None, uses default.
        """
        if strategies_file is None:
            # Default to default_strategies.yaml in the root directory
            self.strategies_file = Path(__file__).parent.parent.parent / "default_strategies.yaml"
        else:
            self.strategies_file = Path(strategies_file)
        
        self._rag_data: Optional[Dict[str, Any]] = None
        self._loaded = False
    
    def load_strategies(self) -> Dict[str, Any]:
        """
        Load RAG configuration from the YAML file.
        
        Returns:
            The RAG configuration dictionary.
        """
        if self._loaded:
            return self._rag_data
        
        try:
            if not self.strategies_file.exists():
                logger.error(f"Strategies file not found: {self.strategies_file}")
                return {}
            
            with open(self.strategies_file, 'r') as file:
                data = yaml.safe_load(file)
            
            # ONLY support new RAG schema format
            if "rag" not in data or not isinstance(data["rag"], dict):
                raise ValueError(f"File {self.strategies_file} does not contain valid RAG schema format. Must have 'rag:' top-level key.")
            
            self._rag_data = data["rag"]
            self._loaded = True
            
            # Log what we loaded
            db_count = len(self._rag_data.get("databases", []))
            proc_count = len(self._rag_data.get("data_processing_strategies", []))
            logger.info(f"Loaded RAG config: {db_count} databases, {proc_count} data processing strategies")
            
        except Exception as e:
            logger.error(f"Failed to load strategies file {self.strategies_file}: {e}")
            self._rag_data = {}
            
        return self._rag_data
    
    def get_database(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific database configuration by name.
        
        Args:
            name: Database name
            
        Returns:
            Database configuration or None if not found
        """
        rag_data = self.load_strategies()
        for db in rag_data.get("databases", []):
            if db.get("name") == name:
                return db
        return None
    
    def get_data_processing_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific data processing strategy by name.
        
        Args:
            name: Strategy name
            
        Returns:
            Data processing strategy configuration or None if not found
        """
        rag_data = self.load_strategies()
        for strategy in rag_data.get("data_processing_strategies", []):
            if strategy.get("name") == name:
                return strategy
        return None
    
    def get_combined_strategy(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get a combined strategy (processing + database) by name.
        Strategy name format: {processing_strategy}_{database_name}
        
        Args:
            strategy_name: Combined strategy name
            
        Returns:
            Combined configuration or None if not found
        """
        # Parse the strategy name
        parts = strategy_name.rsplit('_', 1)
        if len(parts) != 2:
            logger.error(f"Invalid strategy name format: {strategy_name}. Expected format: processing_database")
            return None
        
        proc_name, db_name = parts
        
        # Get the configurations
        proc_strategy = self.get_data_processing_strategy(proc_name)
        database = self.get_database(db_name)
        
        if not proc_strategy:
            logger.error(f"Data processing strategy not found: {proc_name}")
            return None
        
        if not database:
            logger.error(f"Database not found: {db_name}")
            return None
        
        # Return combined configuration
        return {
            "data_processing": proc_strategy,
            "database": database,
            "name": strategy_name
        }
    
    def list_strategies(self) -> List[str]:
        """
        List all available combined strategy names.
        
        Returns:
            List of strategy names in format: processing_database
        """
        rag_data = self.load_strategies()
        databases = rag_data.get("databases", [])
        proc_strategies = rag_data.get("data_processing_strategies", [])
        
        strategies = []
        for proc in proc_strategies:
            for db in databases:
                strategy_name = f"{proc['name']}_{db['name']}"
                strategies.append(strategy_name)
        
        return strategies
    
    def list_databases(self) -> List[str]:
        """
        List all available database names.
        
        Returns:
            List of database names
        """
        rag_data = self.load_strategies()
        return [db["name"] for db in rag_data.get("databases", [])]
    
    def list_data_processing_strategies(self) -> List[str]:
        """
        List all available data processing strategy names.
        
        Returns:
            List of data processing strategy names
        """
        rag_data = self.load_strategies()
        return [s["name"] for s in rag_data.get("data_processing_strategies", [])]
    
    def get_database_defaults(self, database_name: str) -> Dict[str, str]:
        """
        Get the default embedding and retrieval strategies for a database.
        
        Args:
            database_name: Name of the database
            
        Returns:
            Dictionary with 'embedding' and 'retrieval' default strategy names
        """
        db = self.get_database(database_name)
        if not db:
            return {}
        
        return {
            "embedding": db.get("default_embedding_strategy"),
            "retrieval": db.get("default_retrieval_strategy")
        }
    
    def get_database_strategy(self, database_name: str, strategy_type: str, strategy_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a specific strategy from a database, using defaults if no name specified.
        
        Args:
            database_name: Name of the database
            strategy_type: Either 'embedding' or 'retrieval'
            strategy_name: Optional specific strategy name; uses default if not provided
            
        Returns:
            Strategy configuration dictionary or None if not found
        """
        db = self.get_database(database_name)
        if not db:
            return None
        
        # Determine which strategy list to use
        if strategy_type == "embedding":
            strategies = db.get("embedding_strategies", [])
            default_name = db.get("default_embedding_strategy")
        elif strategy_type == "retrieval":
            strategies = db.get("retrieval_strategies", [])
            default_name = db.get("default_retrieval_strategy")
        else:
            return None
        
        # If no specific strategy requested, use default
        if not strategy_name:
            strategy_name = default_name
        
        # Find the strategy by name
        if strategy_name:
            for strategy in strategies:
                if strategy.get("name") == strategy_name:
                    return strategy
        
        # Fallback: look for one marked as default
        for strategy in strategies:
            if strategy.get("default", False):
                return strategy
        
        # Last fallback: return first strategy if available
        if strategies:
            return strategies[0]
        
        return None
    
    def get_raw_rag_data(self) -> Dict[str, Any]:
        """
        Get the raw RAG configuration data.
        
        Returns:
            The complete RAG configuration dictionary
        """
        return self.load_strategies()