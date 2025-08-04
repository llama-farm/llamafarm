"""Factory classes for creating RAG system components."""

from typing import Dict, Any, Type
from core.base import Parser, Embedder, VectorStore
from parsers.csv_parser import CSVParser, CustomerSupportCSVParser
from parsers.pdf_parser import PDFParser
from embedders.ollama_embedder import OllamaEmbedder
from stores.chroma_store import ChromaStore


class ComponentFactory:
    """Base factory for creating RAG components."""

    _registry: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, component_class: Type):
        """Register a component class with a name."""
        cls._registry[name] = component_class

    @classmethod
    def create(cls, component_type: str, config: Dict[str, Any] = None):
        """Create a component instance by type name."""
        if component_type not in cls._registry:
            raise ValueError(f"Unknown component type: {component_type}")

        component_class = cls._registry[component_type]
        return component_class(config=config)

    @classmethod
    def list_available(cls):
        """List all available component types."""
        return list(cls._registry.keys())


class ParserFactory(ComponentFactory):
    """Factory for creating parser instances."""

    _registry = {
        "CSVParser": CSVParser,
        "CustomerSupportCSVParser": CustomerSupportCSVParser,
        "PDFParser": PDFParser,
    }


class EmbedderFactory(ComponentFactory):
    """Factory for creating embedder instances."""

    _registry = {
        "OllamaEmbedder": OllamaEmbedder,
    }


class VectorStoreFactory(ComponentFactory):
    """Factory for creating vector store instances."""

    _registry = {
        "ChromaStore": ChromaStore,
    }


def create_component_from_config(
    component_config: Dict[str, Any], factory_class: Type[ComponentFactory]
):
    """Create a component from configuration using the appropriate factory."""
    component_type = component_config.get("type")
    if not component_type:
        raise ValueError("Component configuration must specify a 'type'")

    config = component_config.get("config", {})
    return factory_class.create(component_type, config)


def create_embedder_from_config(embedder_config: Dict[str, Any]) -> Embedder:
    """Create an embedder from configuration."""
    return create_component_from_config(embedder_config, EmbedderFactory)


def create_parser_from_config(parser_config: Dict[str, Any]) -> Parser:
    """Create a parser from configuration."""
    return create_component_from_config(parser_config, ParserFactory)


def create_vector_store_from_config(store_config: Dict[str, Any]) -> VectorStore:
    """Create a vector store from configuration."""
    return create_component_from_config(store_config, VectorStoreFactory)
