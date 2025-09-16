"""Direct handler for new RAG schema - NO LEGACY CONVERSION."""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import yaml

logger = logging.getLogger(__name__)


class SchemaHandler:
    """Handle new RAG schema directly with global config support."""

    def __init__(self, config_source: str):
        """Initialize with global LlamaFarm config file only."""
        self.config_source = Path(config_source)
        self.global_config = None
        self.rag_config = None

        if self.config_source.exists():
            # Load global config directly with YAML
            with open(self.config_source, "r") as f:
                config = yaml.safe_load(f)
                # Check if this is a global config (has 'rag' section) or direct RAG config
                if "rag" in config:
                    self.global_config = config
                    self.rag_config = config["rag"]
                else:
                    # Direct RAG config format
                    self.rag_config = config
        else:
            raise ValueError(f"Global config file not found: {config_source}")

    def get_available_strategies(self) -> List[str]:
        """Get list of available combined strategy names."""
        if not self.rag_config:
            return []

        strategies = []
        databases = self.rag_config.get("databases", [])
        processing_strategies = self.rag_config.get("data_processing_strategies", [])

        for proc_strategy in processing_strategies:
            for db in databases:
                strategy_name = f"{proc_strategy['name']}_{db['name']}"
                strategies.append(strategy_name)

        return strategies

    def get_database_names(self) -> List[str]:
        """Get list of available database names."""
        return [db["name"] for db in self.rag_config.get("databases", [])]

    def get_data_processing_strategy_names(self) -> List[str]:
        """Get list of available data processing strategy names."""
        return [
            strategy["name"]
            for strategy in self.rag_config.get("data_processing_strategies", [])
        ]

    def get_database_retrieval_strategies(self, database_name: str) -> List[str]:
        """Get available retrieval strategies for a database."""
        for db in self.rag_config.get("databases", []):
            if db["name"] == database_name:
                return [rs["name"] for rs in db.get("retrieval_strategies", [])]
        return []

    def create_database_config(self, database_name: str) -> Dict[str, Any]:
        """Create database configuration for factories."""
        for db in self.rag_config.get("databases", []):
            if db["name"] == database_name:
                # Return the database config as-is from the YAML
                return db
        raise ValueError(f"Database '{database_name}' not found")

    def create_processing_config(self, strategy_name: str) -> Dict[str, Any]:
        """Create data processing strategy configuration."""
        for strategy in self.rag_config.get("data_processing_strategies", []):
            if strategy["name"] == strategy_name:
                return {
                    "parsers": strategy.get("parsers", []),
                    "extractors": strategy.get("extractors", []),
                }
        raise ValueError(f"Data processing strategy '{strategy_name}' not found")

    def parse_strategy_name(
        self, strategy_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Parse combined strategy name into processing and database parts.

        Strategy names are in format: {processing_strategy}_{database_name}
        We need to match against known strategies and databases.
        """
        # Get known strategies and databases
        processing_strategies = [
            s["name"] for s in self.rag_config.get("data_processing_strategies", [])
        ]
        databases = [db["name"] for db in self.rag_config.get("databases", [])]

        # Try to find the best match
        for proc in processing_strategies:
            if strategy_name.startswith(proc + "_"):
                # Found processing strategy prefix
                db_part = strategy_name[len(proc) + 1 :]
                if db_part in [
                    db["name"] for db in self.rag_config.get("databases", [])
                ]:
                    return proc, db_part

        # Fallback to simple split at last underscore
        parts = strategy_name.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    def get_database_config(self, db_name: str) -> Optional[Dict[str, Any]]:
        """Get database configuration by name."""
        if not self.rag_config:
            return None

        for db in self.rag_config.get("databases", []):
            if db.get("name") == db_name:
                return db
        return None

    def get_processing_strategy_config(
        self, proc_name: str
    ) -> Optional[Dict[str, Any]]:
        """Get processing strategy configuration by name."""
        if not self.rag_config:
            return None

        for strategy in self.rag_config.get("data_processing_strategies", []):
            if strategy.get("name") == proc_name:
                return strategy
        return None

    def get_combined_config(
        self, strategy_name: str, source_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Get combined configuration for a strategy (processing + database).

        Returns the actual new schema config without any conversion.
        """
        proc_name, db_name = self.parse_strategy_name(strategy_name)

        if not proc_name or not db_name:
            # Try using the name directly as processing strategy with first database
            proc_name = strategy_name
            databases = self.rag_config.get("databases", []) if self.rag_config else []
            db_name = databases[0]["name"] if databases else None

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
            "source_path": str(source_path) if source_path else None,
        }

    def get_embedder_config(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get embedder configuration from database config."""
        default_name = db_config.get("default_embedding_strategy")
        strategies = db_config.get("embedding_strategies", [])

        # Find the default strategy
        for strategy in strategies:
            if strategy.get("name") == default_name or strategy.get("default"):
                return {
                    "type": strategy.get("type", "OllamaEmbedder"),
                    "config": strategy.get("config", {}),
                }

        # Fallback to first strategy
        if strategies:
            return {
                "type": strategies[0].get("type", "OllamaEmbedder"),
                "config": strategies[0].get("config", {}),
            }

        return {"type": "OllamaEmbedder", "config": {}}

    def get_vector_store_config(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get vector store configuration from database config."""
        # Database config has 'type' and 'config' at the top level
        return {
            "type": db_config.get("type", "ChromaStore"),
            "config": db_config.get("config", {}),
        }

    def get_retrieval_strategy_config(
        self, db_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get retrieval strategy configuration from database config."""
        default_name = db_config.get("default_retrieval_strategy")
        strategies = db_config.get("retrieval_strategies", [])

        # Find the default strategy
        for strategy in strategies:
            if strategy.get("name") == default_name or strategy.get("default"):
                return {
                    "type": strategy.get("type", "BasicSimilarityStrategy"),
                    "config": strategy.get("config", {}),
                }

        # Fallback to first strategy
        if strategies:
            return {
                "type": strategies[0].get("type", "BasicSimilarityStrategy"),
                "config": strategies[0].get("config", {}),
            }

        return {"type": "BasicSimilarityStrategy", "config": {}}

    def get_parsers_config(
        self, proc_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get all parser configurations from processing strategy.
        
        Returns all parsers configured for the strategy.
        """
        return proc_config.get("parsers", [])
    
    def get_parser_config(
        self, proc_config: Dict[str, Any], source_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Get first parser configuration (for backward compatibility).
        
        DEPRECATED: Use get_parsers_config to get all parsers.
        """
        parsers = self.get_parsers_config(proc_config)
        if parsers:
            return parsers[0]
        return {"type": "TextParser_Python", "config": {}}

    def get_extractors_config(
        self, proc_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get extractors configuration from processing strategy."""
        extractors = []
        for ext in proc_config.get("extractors", []):
            extractors.append(
                {"type": ext.get("type"), "config": ext.get("config", {})}
            )
        return extractors

    def create_component_config(
        self, strategy_name: str, source_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """Create a component configuration that can be used by CLI.

        This creates a structure that matches what the CLI expects,
        with components in arrays as per the schema.
        """
        combined = self.get_combined_config(strategy_name, source_path)

        if not combined:
            return {}

        db_config = combined.get("database", {})
        proc_config = combined.get("processing_strategy", {})

        # Get individual component configs
        embedder = self.get_embedder_config(db_config)
        vector_store = self.get_vector_store_config(db_config)
        retrieval = self.get_retrieval_strategy_config(db_config)
        parser = self.get_parser_config(proc_config, source_path)
        extractors = self.get_extractors_config(proc_config)

        # Return in the format the CLI expects
        # The CLI's select_parser_config and select_component_config expect
        # components to be in dictionaries with the component name as key
        return {
            "version": "v1",  # Indicate this is from new schema
            "rag": {
                "parsers": {parser["type"]: parser} if parser else {},
                "embedders": {embedder["type"]: embedder} if embedder else {},
                "vector_stores": {vector_store["type"]: vector_store}
                if vector_store
                else {},
                "retrieval_strategies": {retrieval["type"]: retrieval}
                if retrieval
                else {},
                "extractors": extractors if extractors else [],
                "defaults": {
                    "parser": parser.get("type") if parser else None,
                    "embedder": embedder.get("type") if embedder else None,
                    "vector_store": vector_store.get("type") if vector_store else None,
                    "retrieval_strategy": retrieval.get("type") if retrieval else None,
                },
            },
            "metadata": {
                "strategy_name": strategy_name,
                "database_name": db_config.get("name"),
                "processing_strategy_name": proc_config.get("name"),
            },
        }
