"""Direct handler for new RAG schema - NO LEGACY CONVERSION."""

import sys
from pathlib import Path
from typing import Any, Optional

from _pytest._code import source

from rag.core.base import Parser

# Use the common config module instead of direct YAML loading

# Add the repo root to the path to find the config module
from core.logging import RAGStructLogger
repo_root = Path(__file__).parent.parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from config import load_config
    from config.datamodel import (
        Database,
        DataProcessingStrategy,
        EmbeddingStrategy,
        Extractor,
        Parser,
        RetrievalStrategy,
    )
except ImportError as e:
    raise ImportError(
        f"Could not import config module. Make sure you're running from the repo root. Error: {e}"
    ) from e

logger = RAGStructLogger("rag.core.strategies.handler")


class DbProcessingConfig:
    def __init__(
        self,
        processing_strategy: DataProcessingStrategy,
        database: Database,
        strategy_name: str,
        source_path: Path | None,
    ):
        self.processing_strategy = processing_strategy
        self.database = database
        self.strategy_name = (
            strategy_name or f"{processing_strategy.name}_{database.name}"
        )
        self.source_path = source_path


class SchemaHandler:
    """Handle new RAG schema directly with global config support."""

    def __init__(self, config_source: str):
        """Initialize with global LlamaFarm config file only."""
        self.config_source = Path(config_source)
        self.global_config = None
        self.rag_config = None

        try:
            # Use the common config loader instead of direct YAML loading
            config = load_config(config_path=self.config_source, validate=False)

            # Check if this is a global config (has 'rag' section) or direct RAG config
            if config.rag:
                self.global_config = config
                self.rag_config = config.rag
            else:
                # Direct RAG config format
                raise ValueError("Global config file does not have a rag section")
        except Exception as e:
            raise ValueError(
                f"Global config file not found or invalid: {config_source}. Error: {e}"
            ) from e

    def get_available_strategies(self) -> list[str]:
        """Get list of available combined strategy names."""
        if not self.rag_config:
            return []

        strategies = []
        databases = self.rag_config.databases or []
        processing_strategies = self.rag_config.data_processing_strategies or []

        for proc_strategy in processing_strategies:
            for db in databases:
                strategy_name = f"{proc_strategy.name}_{db.name}"
                strategies.append(strategy_name)

        return strategies

    def get_database_names(self) -> list[str]:
        """Get list of available database names."""
        return [db.name for db in self.rag_config.databases or []]

    def get_data_processing_strategy_names(self) -> list[str]:
        """Get list of available data processing strategy names."""
        return [
            strategy.name
            for strategy in self.rag_config.data_processing_strategies or []
        ]

    def get_database_retrieval_strategies(self, database_name: str) -> list[str]:
        """Get available retrieval strategies for a database."""
        return next(
            (
                [rs.name for rs in db.retrieval_strategies or []]
                for db in self.rag_config.databases or []
                if db.name == database_name
            ),
            [],
        )

    def create_database_config(self, database_name: str) -> Database:
        """Create database configuration for factories."""
        for db in self.rag_config.databases or []:
            if db.name == database_name:
                # Return the database config as-is from the YAML
                return db
        raise ValueError(f"Database '{database_name}' not found")

    def create_processing_config(self, strategy_name: str) -> DataProcessingStrategy:
        """Create data processing strategy configuration."""
        if not self.rag_config or not self.rag_config.data_processing_strategies:
            raise ValueError("No data processing strategies found")

        for strategy in self.rag_config.data_processing_strategies or []:
            if strategy.name == strategy_name:
                return strategy
        raise ValueError(f"Data processing strategy '{strategy_name}' not found")

    def parse_strategy_name(
        self, strategy_name: str
    ) -> tuple[Optional[str], Optional[str]]:
        """Parse combined strategy name into processing and database parts.

        Strategy names are in format: {processing_strategy}_{database_name}
        We need to match against known strategies and databases.
        """
        # Get known strategies and databases
        processing_strategies = [
            str(s.name) for s in self.rag_config.data_processing_strategies or []
        ]
        databases = [db.name for db in self.rag_config.databases or []]

        # Try to find the best match
        for proc in processing_strategies:
            if strategy_name.startswith(proc + "_"):
                # Found processing strategy prefix
                db_part = strategy_name[len(proc) + 1 :]
                if db_part in databases:
                    return proc, db_part

        # Fallback to simple split at last underscore
        parts = strategy_name.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return None, None

    def get_database_config(self, db_name: str) -> Optional[Database]:
        """Get database configuration by name."""
        if not self.rag_config:
            return None

        return next(
            (db for db in self.rag_config.databases or [] if db.name == db_name),
            None,
        )

    def get_processing_strategy_config(
        self, proc_name: str
    ) -> Optional[DataProcessingStrategy]:
        """Get processing strategy configuration by name."""
        if not self.rag_config:
            return None

        return next(
            (
                strategy
                for strategy in (self.rag_config.data_processing_strategies or [])
                if strategy.name == proc_name
            ),
            None,
        )

    def get_combined_config(
        self, strategy_name: str, source_path: Optional[Path] = None
    ) -> DbProcessingConfig:
        """Get combined configuration for a strategy (processing + database).

        Returns the actual new schema config without any conversion.
        """
        proc_name, db_name = self.parse_strategy_name(strategy_name)

        if not proc_name or not db_name:
            # Try using the name directly as processing strategy with first database
            proc_name = strategy_name
            databases = self.rag_config.databases or []
            db_name = databases[0].name if databases else None

        if not proc_name or not db_name:
            raise ValueError(f"Strategy name {strategy_name} not found")

        proc_config = self.get_processing_strategy_config(proc_name)
        db_config = self.get_database_config(db_name)

        if not proc_config:
            logger.error(f"Processing strategy not found: {proc_name}")
            return {}

        if not db_config:
            logger.error(f"Database not found: {db_name}")
            return {}

        # Return the actual new schema configuration
        return DbProcessingConfig(proc_config, db_config, strategy_name, source_path)

    def get_embedder_config(self, database: Database) -> EmbeddingStrategy:
        """Get embedder configuration from database config."""
        default_name = database.default_embedding_strategy
        strategies = database.embedding_strategies or []

        # Find the default strategy
        for strategy in strategies:
            if (
                strategy.name == default_name
                or database.default_embedding_strategy == strategy.name
            ):
                return strategy

        # Fallback to first strategy
        if strategies:
            return strategies[0]

        return EmbeddingStrategy(type="OllamaEmbedder", config={})

    def get_vector_store_config(self, database: Database) -> Database:
        """Get vector store configuration from database config."""
        # Database config has 'type' and 'config' at the top level
        return database

    def get_retrieval_strategy_config(self, database: Database) -> RetrievalStrategy:
        """Get retrieval strategy configuration from database config."""
        default_name = database.default_retrieval_strategy
        strategies = database.retrieval_strategies or []

        # Find the default strategy
        for strategy in strategies:
            if (
                strategy.name == default_name
                or database.default_retrieval_strategy == strategy.name
            ):
                return strategy

        # Fallback to first strategy
        if strategies:
            return strategies[0]

        return RetrievalStrategy(type="BasicSimilarityStrategy", config={})

    def get_parsers_config(self, proc_config: DataProcessingStrategy) -> list[Parser]:
        """Get all parser configurations from processing strategy.

        Returns all parsers configured for the strategy.
        """
        return proc_config.parsers or []

    def get_parser_config(
        self, proc_config: DataProcessingStrategy, source_path: Optional[Path] = None
    ) -> Parser:
        """Get first parser configuration (for backward compatibility).

        DEPRECATED: Use get_parsers_config to get all parsers.
        """
        parsers = self.get_parsers_config(proc_config)
        if parsers:
            return parsers[0]
        return Parser(type="TextParser_Python", config={})

    def get_extractors_config(
        self, proc_config: DataProcessingStrategy
    ) -> list[Extractor]:
        """Get extractors configuration from processing strategy."""
        return proc_config.extractors or []

    def create_component_config(
        self, strategy_name: str, source_path: Optional[Path] = None
    ) -> dict[str, Any]:
        """Create a component configuration that can be used by CLI.

        This creates a structure that matches what the CLI expects,
        with components in arrays as per the schema.
        """
        combined = self.get_combined_config(strategy_name, source_path)

        if not combined:
            return {}

        db_config = combined.database or {}
        proc_config = combined.processing_strategy or {}

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
                "parsers": {parser.type.value: parser} if parser else {},
                "embedders": {embedder.type.value: embedder} if embedder else {},
                "vector_stores": {vector_store.type.value: vector_store}
                if vector_store
                else {},
                "retrieval_strategies": {retrieval.type.value: retrieval}
                if retrieval
                else {},
                "extractors": extractors if extractors else [],
                "defaults": {
                    "parser": parser.type.value if parser else None,
                    "embedder": embedder.type.value if embedder else None,
                    "vector_store": vector_store.type.value if vector_store else None,
                    "retrieval_strategy": retrieval.type.value if retrieval else None,
                },
            },
            "metadata": {
                "strategy_name": strategy_name,
                "database_name": db_config.name,
                "processing_strategy_name": str(proc_config.name),
            },
        }
