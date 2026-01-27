"""
ComponentResolver expands reusable component references and applies defaults.

It resolves references under `components.*` (embedding_strategies, retrieval_strategies,
parsers) into inline definitions on databases and data processing strategies, so
downstream RAG code receives fully inlined configs.

Notes:
- Resolution is pure: we work on a deep copy of the incoming config.
- Validation errors raise ValueError with clear messages.
- This module assumes the datamodel has been regenerated to include the new
  component fields; attribute access is guarded to avoid crashes if fields are
  temporarily missing during migration.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from config.datamodel import (
    Database,
    DatabaseEmbeddingStrategy,
    DatabaseRetrievalStrategy,
    DataProcessingStrategyDefinition,
    LlamaFarmConfig,
    NamedEmbeddingStrategy,
    NamedParserDefinition,
    NamedRetrievalStrategy,
    Parser,
)


class ComponentResolver:
    def __init__(self, config: LlamaFarmConfig):
        self._config = config
        # Safely extract components; fallback to empty dicts when absent to keep resolver tolerant
        components = getattr(config, "components", None) or {}

        def _get_component_collection(name: str) -> list[Any]:
            if isinstance(components, dict):
                return components.get(name, []) or []
            return getattr(components, name, []) or []

        embedding_sources = _get_component_collection("embedding_strategies")
        retrieval_sources = _get_component_collection("retrieval_strategies")
        parser_sources = _get_component_collection("parsers")

        self._embedding_map = self._build_component_map(
            embedding_sources, "embedding_strategies"
        )
        self._retrieval_map = self._build_component_map(
            retrieval_sources, "retrieval_strategies"
        )
        self._parser_map = self._build_component_map(parser_sources, "parsers")

        defaults_source = (
            components.get("defaults", {})
            if isinstance(components, dict)
            else getattr(components, "defaults", {}) or {}
        )
        if isinstance(defaults_source, dict):
            self._defaults = defaults_source
        elif hasattr(defaults_source, "model_dump"):
            # components.defaults may be a Pydantic model; normalize to dict
            self._defaults = defaults_source.model_dump()
        else:
            self._defaults = {}

    @staticmethod
    def _build_component_map(
        components: list[Any], component_type: str
    ) -> dict[str, Any]:
        """
        Validate that each component has a unique, non-empty name and build a map.
        Raises ValueError when validation fails to avoid None keys masking lookups.
        """
        mapping: dict[str, Any] = {}
        for component in components:
            name = (
                component.get("name")
                if isinstance(component, dict)
                else getattr(component, "name", None)
            )
            if not name:
                raise ValueError(
                    f"All {component_type} entries must include a 'name' field"
                )
            if name in mapping:
                raise ValueError(
                    f"Duplicate {component_type} name '{name}' detected; names must be unique"
                )
            mapping[name] = component
        return mapping

    def resolve_config(self, config: LlamaFarmConfig) -> LlamaFarmConfig:
        """Return a deep-copied config with all component references expanded."""
        resolved = config.model_copy(deep=True)

        rag_cfg = getattr(resolved, "rag", None)
        if rag_cfg:
            # Resolve databases
            databases = getattr(rag_cfg, "databases", None) or []
            resolved_dbs: list[Database] = []
            for db in databases:
                resolved_db = self._resolve_database_references(db)
                resolved_db = self._apply_defaults(resolved_db)
                resolved_dbs.append(resolved_db)
            rag_cfg.databases = resolved_dbs

            # Resolve parsers in data processing strategies
            strategies = getattr(rag_cfg, "data_processing_strategies", None) or []
            resolved_strategies: list[DataProcessingStrategyDefinition] = []
            for strategy in strategies:
                resolved_strategies.append(self._resolve_parsers(strategy))
            rag_cfg.data_processing_strategies = resolved_strategies

        return resolved

    def _resolve_database_references(self, db: Database) -> Database:
        """Expand embedding/retrieval references on a database into inline definitions."""
        db_copy: Database = db.model_copy(deep=True)

        # Embedding: if reference field present and strategies missing/empty, inline referenced
        ref_embed = getattr(db, "embedding_strategy", None)
        if ref_embed and (not db_copy.embedding_strategies):
            embed_def = self._resolve_named_component(
                ref_embed, self._embedding_map, "embedding_strategy"
            )
            db_copy.embedding_strategies = [self._to_embedding_strategy(embed_def)]
            if not getattr(db_copy, "default_embedding_strategy", None):
                db_copy.default_embedding_strategy = ref_embed

        # Retrieval: if reference field present and strategies missing/empty, inline referenced
        ref_retrieval = getattr(db, "retrieval_strategy", None)
        if ref_retrieval and (not db_copy.retrieval_strategies):
            retrieval_def = self._resolve_named_component(
                ref_retrieval, self._retrieval_map, "retrieval_strategy"
            )
            db_copy.retrieval_strategies = [self._to_retrieval_strategy(retrieval_def)]
            if not getattr(db_copy, "default_retrieval_strategy", None):
                db_copy.default_retrieval_strategy = ref_retrieval

        return db_copy

    def _apply_defaults(self, db: Database) -> Database:
        """Apply global component defaults when fields are missing."""
        db_copy: Database = db.model_copy(deep=True)

        # Default embedding strategy
        if not getattr(db_copy, "default_embedding_strategy", None):
            default_embed = self._defaults.get("embedding_strategy")
            if default_embed:
                inline_embeddings = db_copy.embedding_strategies or []
                if not inline_embeddings:
                    # No inline strategies: inline the default and mark it as default
                    embed_def = self._resolve_named_component(
                        default_embed, self._embedding_map, "embedding_strategy"
                    )
                    db_copy.embedding_strategies = [
                        self._to_embedding_strategy(embed_def)
                    ]
                    db_copy.default_embedding_strategy = default_embed
                else:
                    # Inline strategies present: only set default if it exists in the list
                    inline_names = set()
                    for strategy in inline_embeddings:
                        name = getattr(strategy, "name", None)
                        if name is None and isinstance(strategy, dict):
                            name = strategy.get("name")
                        if name:
                            inline_names.add(name)
                    if default_embed in inline_names:
                        db_copy.default_embedding_strategy = default_embed

        # Default retrieval strategy
        if not getattr(db_copy, "default_retrieval_strategy", None):
            default_retrieval = self._defaults.get("retrieval_strategy")
            if default_retrieval:
                inline_retrievals = db_copy.retrieval_strategies or []
                if not inline_retrievals:
                    retrieval_def = self._resolve_named_component(
                        default_retrieval, self._retrieval_map, "retrieval_strategy"
                    )
                    db_copy.retrieval_strategies = [
                        self._to_retrieval_strategy(retrieval_def)
                    ]
                    db_copy.default_retrieval_strategy = default_retrieval
                else:
                    inline_names = set()
                    for strategy in inline_retrievals:
                        name = getattr(strategy, "name", None)
                        if name is None and isinstance(strategy, dict):
                            name = strategy.get("name")
                        if name:
                            inline_names.add(name)
                    if default_retrieval in inline_names:
                        db_copy.default_retrieval_strategy = default_retrieval

        return db_copy

    def _resolve_parsers(
        self, strategy: DataProcessingStrategyDefinition
    ) -> DataProcessingStrategyDefinition:
        """Expand parser references within a data processing strategy."""
        strat_copy: DataProcessingStrategyDefinition = strategy.model_copy(deep=True)
        parsers: list[Any] = getattr(strategy, "parsers", None) or []
        resolved_parsers: list[Parser] = []

        for parser in parsers:
            if isinstance(parser, str):
                parser_def = self._resolve_named_component(
                    parser, self._parser_map, "parser"
                )
                resolved_parsers.append(self._to_parser(parser_def))
            else:
                # Already inline
                resolved_parsers.append(self._to_parser(parser))

        strat_copy.parsers = resolved_parsers
        return strat_copy

    def _validate_component_exists(self, ref: str, component_type: str) -> None:
        """Validate that a named component exists; raises ValueError if not found."""
        mapping = {
            "embedding_strategy": self._embedding_map,
            "retrieval_strategy": self._retrieval_map,
            "parser": self._parser_map,
        }.get(component_type, {})
        plural_suffix = (
            "embedding_strategies"
            if component_type == "embedding_strategy"
            else "retrieval_strategies"
            if component_type == "retrieval_strategy"
            else "parsers"
        )
        if ref not in mapping:
            raise ValueError(
                f"{component_type} '{ref}' not found in components.{plural_suffix}"
            )

    def _resolve_named_component(
        self, ref: str, collection: dict[str, Any], component_type: str
    ) -> Any:
        self._validate_component_exists(ref, component_type)
        return collection[ref]

    @staticmethod
    def _to_embedding_strategy(defn: Any) -> DatabaseEmbeddingStrategy:
        if isinstance(defn, DatabaseEmbeddingStrategy):
            return defn.model_copy(deep=True)
        try:
            payload = (
                defn.model_dump()
                if isinstance(defn, NamedEmbeddingStrategy)
                else deepcopy(defn)
            )
            if isinstance(payload, dict):
                type_value = payload.get("type")
                if hasattr(type_value, "value"):
                    payload["type"] = type_value.value
            return DatabaseEmbeddingStrategy(**payload)
        except Exception as e:  # pragma: no cover - defensive guardrails
            raise ValueError(f"Invalid embedding strategy definition: {e}") from e

    @staticmethod
    def _to_retrieval_strategy(defn: Any) -> DatabaseRetrievalStrategy:
        if isinstance(defn, DatabaseRetrievalStrategy):
            return defn.model_copy(deep=True)
        try:
            payload = (
                defn.model_dump()
                if isinstance(defn, NamedRetrievalStrategy)
                else deepcopy(defn)
            )
            if isinstance(payload, dict):
                type_value = payload.get("type")
                if hasattr(type_value, "value"):
                    payload["type"] = type_value.value
            return DatabaseRetrievalStrategy(**payload)
        except Exception as e:  # pragma: no cover - defensive guardrails
            raise ValueError(f"Invalid retrieval strategy definition: {e}") from e

    @staticmethod
    def _to_parser(defn: Any) -> Parser:
        if isinstance(defn, Parser):
            return defn.model_copy(deep=True)
        try:
            payload = (
                defn.model_dump()
                if isinstance(defn, NamedParserDefinition)
                else deepcopy(defn)
            )
            if isinstance(payload, dict):
                payload.pop("name", None)
            return Parser(**payload)
        except Exception as e:  # pragma: no cover - defensive guardrails
            raise ValueError(f"Invalid parser definition: {e}") from e
