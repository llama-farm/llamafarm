from __future__ import annotations

from pathlib import Path

import pytest

from config import load_config
from config.datamodel import DataProcessingStrategyDefinition, Parser
from config.helpers.strategy_resolver import StrategyResolver


def _load_base_config():
    config_path = Path(__file__).parent / "minimal_config.yaml"
    return load_config(config_path)


def _build_strategy(name: str, parser_type: str, config: dict | None = None):
    return DataProcessingStrategyDefinition(
        name=name,
        description="Strategy used for resolver tests.",
        parsers=[
            Parser(
                type=parser_type,
                config=config or {},
                file_extensions=None,
                file_include_patterns=None,
                priority=50,
                mime_types=None,
                fallback_parser=None,
            )
        ],
        extractors=[],
    )


def test_built_in_defaults_only():
    config = _load_base_config()
    config.rag.data_processing_strategies = [
        _build_strategy("default", "PDFParser_PyPDF2", config={})
    ]

    resolver = StrategyResolver(config)
    resolved = resolver.resolve_processing_strategy("default")

    parser = resolved.parsers[0]
    assert parser.config["chunk_size"] == 512
    assert parser.config["chunk_overlap"] == 50


def test_strategy_override():
    config = _load_base_config()
    config.rag.data_processing_strategies = [
        _build_strategy("default", "PDFParser_PyPDF2", config={"chunk_size": 1024})
    ]

    resolver = StrategyResolver(config)
    resolved = resolver.resolve_processing_strategy("default")

    parser = resolved.parsers[0]
    assert parser.config["chunk_size"] == 1024
    assert parser.config["chunk_overlap"] == 50


def test_api_override():
    config = _load_base_config()
    config.rag.data_processing_strategies = [
        _build_strategy("default", "PDFParser_PyPDF2", config={"chunk_size": 1024})
    ]

    resolver = StrategyResolver(config)
    resolved = resolver.resolve_processing_strategy(
        "default", api_overrides={"PDFParser_PyPDF2": {"chunk_size": 2048}}
    )

    parser = resolved.parsers[0]
    assert parser.config["chunk_size"] == 2048
    assert parser.config["chunk_overlap"] == 50


def test_deep_merge():
    config = _load_base_config()
    config.rag.data_processing_strategies = [
        _build_strategy(
            "default",
            "PDFParser_LlamaIndex",
            config={"nested": {"a": 1}},
        )
    ]

    resolver = StrategyResolver(config)
    resolved = resolver.resolve_processing_strategy(
        "default", api_overrides={"PDFParser_LlamaIndex": {"nested": {"b": 2}}}
    )

    parser = resolved.parsers[0]
    assert parser.config["nested"] == {"a": 1, "b": 2}


def test_strategy_not_found():
    config = _load_base_config()
    config.rag.data_processing_strategies = [
        _build_strategy("default", "PDFParser_PyPDF2", config={})
    ]
    resolver = StrategyResolver(config)

    with pytest.raises(ValueError, match="not found"):
        resolver.resolve_processing_strategy("missing")


def test_unknown_parser_type():
    config = _load_base_config()
    config.rag.data_processing_strategies = [
        _build_strategy("default", "UnknownParser_Test", config={})
    ]

    resolver = StrategyResolver(config)
    resolved = resolver.resolve_processing_strategy("default")

    parser = resolved.parsers[0]
    assert parser.config == {}
