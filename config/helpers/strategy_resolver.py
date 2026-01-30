"""Resolve parser configurations using cascading defaults."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from config.datamodel import DataProcessingStrategyDefinition, LlamaFarmConfig
from config.defaults.parser_defaults import get_parser_defaults

logger = logging.getLogger(__name__)


class StrategyResolver:
    """Resolve parser configurations using a 3-level cascade."""

    def __init__(self, config: LlamaFarmConfig):
        self._config = config

    def get_strategy(self, strategy_name: str) -> DataProcessingStrategyDefinition:
        strategies = list(self._iter_strategies())
        for strategy in strategies:
            if strategy.name == strategy_name:
                return strategy
        available = [strategy.name for strategy in strategies]
        raise ValueError(
            f"Strategy '{strategy_name}' not found in configuration. "
            f"Available strategies: {available}"
        )

    def resolve_processing_strategy(
        self,
        strategy_name: str,
        api_overrides: dict[str, dict[str, Any]] | None = None,
    ) -> DataProcessingStrategyDefinition:
        """Return a strategy with parser configs merged with defaults and overrides."""
        strategy = self.get_strategy(strategy_name)
        resolved = strategy.model_copy(deep=True)

        for parser in resolved.parsers or []:
            parser_type = getattr(parser, "type", None)
            if not parser_type:
                logger.warning("Parser missing type; skipping defaults merge")
                continue

            defaults = get_parser_defaults(parser_type)
            if not defaults:
                logger.warning(
                    "No built-in defaults for parser", extra={"type": parser_type}
                )

            merged = defaults
            parser_config = getattr(parser, "config", None)
            if isinstance(parser_config, dict):
                merged = self._deep_merge(merged, parser_config)
            elif parser_config is not None:
                logger.warning(
                    "Parser config is not a dict; skipping merge",
                    extra={"type": parser_type},
                )

            if api_overrides and parser_type in api_overrides:
                override_config = api_overrides[parser_type]
                if isinstance(override_config, dict):
                    merged = self._deep_merge(merged, override_config)
                else:
                    logger.warning(
                        "API override is not a dict; skipping merge",
                        extra={"type": parser_type},
                    )

            parser.config = merged
            logger.debug(
                "Resolved parser config", extra={"type": parser_type, "config": merged}
            )

        return resolved

    def _iter_strategies(self) -> Iterable[DataProcessingStrategyDefinition]:
        rag_config = getattr(self._config, "rag", None)
        if not rag_config:
            return []

        strategies = getattr(rag_config, "data_processing_strategies", None)
        if not strategies:
            return []

        if isinstance(strategies, list):
            return strategies

        if hasattr(strategies, "strategies"):
            return getattr(strategies, "strategies", []) or []

        return []

    def _deep_merge(
        self, base: dict[str, Any], override: dict[str, Any]
    ) -> dict[str, Any]:
        result = deepcopy(base)
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result
