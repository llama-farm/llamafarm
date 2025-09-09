"""Direct handler for new RAG schema - NO LEGACY CONVERSION."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class SchemaHandler:
    """Handle new RAG schema directly without any legacy conversion."""
    
    def __init__(self, strategy_file: str):
        """Initialize with a strategy file path."""
        self.strategy_file = Path(strategy_file)
        self.rag_config = None
        
        if self.strategy_file.exists():
            with open(self.strategy_file, 'r') as f:
                config = yaml.safe_load(f)
                self.rag_config = config.get('rag', {})
        else:
            logger.error(f"Strategy file not found: {strategy_file}")
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available combined strategy names."""
        if not self.rag_config:
            return []
        
        strategies = []
        databases = self.rag_config.get('databases', [])
        processing_strategies = self.rag_config.get('data_processing_strategies', [])
        
        for proc_strategy in processing_strategies:
            for db in databases:
                strategy_name = f"{proc_strategy['name']}_{db['name']}"
                strategies.append(strategy_name)
        
        return strategies
    
    def parse_strategy_name(self, strategy_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse combined strategy name into processing and database parts.
        
        Strategy names are in format: {processing_strategy}_{database_name}
        We need to match against known strategies and databases.
        """
        # Get known strategies and databases
        processing_strategies = [s['name'] for s in self.rag_config.get('data_processing_strategies', [])]
        databases = [db['name'] for db in self.rag_config.get('databases', [])]
        
        # Try to find the best match
        for proc in processing_strategies:
            if strategy_name.startswith(proc + '_'):
                # Found processing strategy prefix
                db_part = strategy_name[len(proc) + 1:]
                if db_part in [db['name'] for db in self.rag_config.get('databases', [])]:
                    return proc, db_part
        
        # Fallback to simple split at last underscore
        parts = strategy_name.rsplit('_', 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None
    
    def get_database_config(self, db_name: str) -> Optional[Dict[str, Any]]:
        """Get database configuration by name."""
        if not self.rag_config:
            return None
        
        for db in self.rag_config.get('databases', []):
            if db.get('name') == db_name:
                return db
        return None
    
    def get_processing_strategy_config(self, proc_name: str) -> Optional[Dict[str, Any]]:
        """Get processing strategy configuration by name."""
        if not self.rag_config:
            return None
        
        for strategy in self.rag_config.get('data_processing_strategies', []):
            if strategy.get('name') == proc_name:
                return strategy
        return None
    
    def get_combined_config(self, strategy_name: str, source_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get combined configuration for a strategy (processing + database).
        
        Returns the actual new schema config without any conversion.
        """
        proc_name, db_name = self.parse_strategy_name(strategy_name)
        
        if not proc_name or not db_name:
            # Try using the name directly as processing strategy with first database
            proc_name = strategy_name
            databases = self.rag_config.get('databases', []) if self.rag_config else []
            db_name = databases[0]['name'] if databases else None
        
        proc_config = self.get_processing_strategy_config(proc_name)
        db_config = self.get_database_config(db_name)
        
        if not proc_config:
            logger.error(f"Processing strategy not found: {proc_name}")
            return {}
        
        if not db_config:
            logger.error(f"Database not found: {db_name}")
            return {}
        
        # Return the actual new schema configuration
        return {
            "processing_strategy": proc_config,
            "database": db_config,
            "strategy_name": strategy_name,
            "source_path": str(source_path) if source_path else None
        }
    
    def get_embedder_config(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get embedder configuration from database config."""
        default_name = db_config.get('default_embedding_strategy')
        strategies = db_config.get('embedding_strategies', [])
        
        # Find the default strategy
        for strategy in strategies:
            if strategy.get('name') == default_name or strategy.get('default'):
                return {
                    'type': strategy.get('type', 'OllamaEmbedder'),
                    'config': strategy.get('config', {})
                }
        
        # Fallback to first strategy
        if strategies:
            return {
                'type': strategies[0].get('type', 'OllamaEmbedder'),
                'config': strategies[0].get('config', {})
            }
        
        return {'type': 'OllamaEmbedder', 'config': {}}
    
    def get_vector_store_config(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get vector store configuration from database config."""
        return {
            'type': db_config.get('type', 'ChromaStore'),
            'config': db_config.get('config', {})
        }
    
    def get_retrieval_strategy_config(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get retrieval strategy configuration from database config."""
        default_name = db_config.get('default_retrieval_strategy')
        strategies = db_config.get('retrieval_strategies', [])
        
        # Find the default strategy
        for strategy in strategies:
            if strategy.get('name') == default_name or strategy.get('default'):
                return {
                    'type': strategy.get('type', 'BasicSimilarityStrategy'),
                    'config': strategy.get('config', {})
                }
        
        # Fallback to first strategy
        if strategies:
            return {
                'type': strategies[0].get('type', 'BasicSimilarityStrategy'),
                'config': strategies[0].get('config', {})
            }
        
        return {'type': 'BasicSimilarityStrategy', 'config': {}}
    
    def get_parser_config(self, proc_config: Dict[str, Any], source_path: Optional[Path] = None) -> Dict[str, Any]:
        """Get parser configuration from processing strategy.
        
        DirectoryParser is ALWAYS active at the strategy level.
        """
        # DirectoryParser is always on with directory_config
        directory_config = proc_config.get('directory_config', {})
        
        # Add parsers from the processing strategy
        parsers = proc_config.get('parsers', [])
        if parsers:
            directory_config['parsers'] = parsers
        
        return {
            'type': 'DirectoryParser',
            'config': directory_config
        }
    
    def get_extractors_config(self, proc_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get extractors configuration from processing strategy."""
        extractors = []
        for ext in proc_config.get('extractors', []):
            extractors.append({
                'type': ext.get('type'),
                'config': ext.get('config', {})
            })
        return extractors
    
    def create_component_config(self, strategy_name: str, source_path: Optional[Path] = None) -> Dict[str, Any]:
        """Create a component configuration that can be used by CLI.
        
        This creates a flat structure with individual component configs
        that the CLI can use to create components.
        """
        combined = self.get_combined_config(strategy_name, source_path)
        
        if not combined:
            return {}
        
        db_config = combined.get('database', {})
        proc_config = combined.get('processing_strategy', {})
        
        return {
            'embedder': self.get_embedder_config(db_config),
            'vector_store': self.get_vector_store_config(db_config),
            'retrieval_strategy': self.get_retrieval_strategy_config(db_config),
            'parser': self.get_parser_config(proc_config, source_path),
            'extractors': self.get_extractors_config(proc_config),
            'strategy_name': strategy_name,
            'database_name': db_config.get('name'),
            'processing_strategy_name': proc_config.get('name')
        }