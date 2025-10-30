"""Factory classes for creating RAG system components."""

from pathlib import Path
from typing import Any

# Import the unified parser system
from components.parsers import (
    ToolAwareParserFactory,
    DirectoryParser,
)
from core.base import Embedder, Parser, VectorStore
from core.logging import RAGStructLogger


# Create parser factory functions for backward compatibility
def _create_parser_factory(parser_name: str):
    """Create a parser factory function for a specific parser type."""

    def parser_factory(*args, **kwargs):
        return ToolAwareParserFactory.create_parser(
            parser_name=parser_name, config=kwargs.get("config")
        )

    return parser_factory


# Legacy parser factory functions
PlainTextParser = _create_parser_factory("TextParser_Python")
PDFParser = _create_parser_factory("PDFParser_LlamaIndex")
CSVParser = _create_parser_factory("CSVParser_Pandas")
DocxParser = _create_parser_factory("DocxParser_LlamaIndex")
MarkdownParser = _create_parser_factory("MarkdownParser_LlamaIndex")
HTMLParser = _create_parser_factory("TextParser_LlamaIndex")  # Web fallback to text
ExcelParser = _create_parser_factory("ExcelParser_Pandas")
CustomerSupportCSVParser = _create_parser_factory("CSVParser_Pandas")

PDF_AVAILABLE = True  # Always available through fallback

# Import embedders
from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder
from components.embedders.universal_embedder.universal_embedder import UniversalEmbedder

# Conditional imports for embedders with dependencies
try:
    from components.embedders.openai_embedder.openai_embedder import OpenAIEmbedder

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from components.embedders.huggingface_embedder.huggingface_embedder import (
        HuggingFaceEmbedder,
    )

    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from components.embedders.sentence_transformer_embedder.sentence_transformer_embedder import (
        SentenceTransformerEmbedder,
    )

    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False

# Conditional imports for vector stores
try:
    from components.stores.chroma_store.chroma_store import ChromaStore

    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

try:
    from components.stores.faiss_store.faiss_store import FAISSStore

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from components.stores.pinecone_store.pinecone_store import PineconeStore

    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    from components.stores.qdrant_store.qdrant_store import QdrantStore

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

# Import extractors
from components.extractors.date_time_extractor.date_time_extractor import (
    DateTimeExtractor,
)
from components.extractors.entity_extractor.entity_extractor import EntityExtractor
from components.extractors.heading_extractor.heading_extractor import HeadingExtractor
from components.extractors.keyword_extractor.keyword_extractor import (
    RAKEExtractor,
    TFIDFExtractor,
    YAKEExtractor,
)
from components.extractors.link_extractor.link_extractor import LinkExtractor
from components.extractors.pattern_extractor.pattern_extractor import PatternExtractor
from components.extractors.statistics_extractor.statistics_extractor import (
    ContentStatisticsExtractor,
)
from components.extractors.summary_extractor.summary_extractor import SummaryExtractor
from components.extractors.table_extractor.table_extractor import TableExtractor

# Import retrieval strategies
from components.retrievers.basic_similarity.basic_similarity import (
    BasicSimilarityStrategy,
)
from components.retrievers.hybrid_universal.hybrid_universal import (
    HybridUniversalStrategy,
)
from components.retrievers.metadata_filtered.metadata_filtered import (
    MetadataFilteredStrategy,
)
from components.retrievers.multi_query.multi_query import MultiQueryStrategy
from components.retrievers.reranked.reranked import RerankedStrategy


class ComponentFactory:
    """Base factory for creating RAG components."""

    _registry: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, component_class: type):
        """Register a component class with a name."""
        cls._registry[name] = component_class

    @classmethod
    def create(
        cls,
        component_type: str,
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        """Create a component instance by type name."""
        if component_type not in cls._registry:
            raise ValueError(
                f"Unknown component type: {component_type}. Registry: {cls._registry}"
            )

        component_class = cls._registry[component_type]
        return component_class(config=config, project_dir=project_dir)

    @classmethod
    def list_available(cls):
        """List all available component types."""
        return list(cls._registry.keys())


class ParserFactoryWrapper(ComponentFactory):
    """Factory for creating parser instances using the new modular system."""

    @classmethod
    def create(
        cls,
        component_type: str,
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        """Create a parser instance using the new ParserFactory.

        Note: project_dir is accepted for API compatibility but not used by parsers.
        """
        # Use the new ParserFactory from components.parsers
        return ToolAwareParserFactory.create_parser(
            parser_name=component_type, config=config
        )

    @classmethod
    def list_available(cls):
        """List available parsers."""
        return [
            "CSVParser",
            "CustomerSupportCSVParser",
            "MarkdownParser",
            "DocxParser",
            "PlainTextParser",
            "HTMLParser",
            "ExcelParser",
            "DirectoryParser",
            "PDFParser",
            # New names
            "text",
            "pdf",
            "csv_excel",
            "docx",
            "markdown",
            "web",
        ]


# Keep the name ParserFactory for backward compatibility but use the wrapper
ParserFactory = ParserFactoryWrapper


class EmbedderFactory(ComponentFactory):
    """Factory for creating embedder instances."""

    _registry = {
        "OllamaEmbedder": OllamaEmbedder,
        "UniversalEmbedder": UniversalEmbedder,
    }

    # Add embedders conditionally based on availability
    if OPENAI_AVAILABLE:
        _registry["OpenAIEmbedder"] = OpenAIEmbedder
    if HUGGINGFACE_AVAILABLE:
        _registry["HuggingFaceEmbedder"] = HuggingFaceEmbedder
    if SENTENCE_TRANSFORMER_AVAILABLE:
        _registry["SentenceTransformerEmbedder"] = SentenceTransformerEmbedder


class VectorStoreFactory(ComponentFactory):
    """Factory for creating vector store instances."""

    _registry = {}

    # Add vector stores conditionally based on availability
    if CHROMA_AVAILABLE:
        _registry["ChromaStore"] = ChromaStore
    if FAISS_AVAILABLE:
        _registry["FAISSStore"] = FAISSStore
    if PINECONE_AVAILABLE:
        _registry["PineconeStore"] = PineconeStore
    if QDRANT_AVAILABLE:
        _registry["QdrantStore"] = QdrantStore


class ExtractorFactory(ComponentFactory):
    """Factory for creating extractor instances."""

    _registry = {
        "YAKEExtractor": YAKEExtractor,
        "RAKEExtractor": RAKEExtractor,
        "TFIDFExtractor": TFIDFExtractor,
        "EntityExtractor": EntityExtractor,
        "DateTimeExtractor": DateTimeExtractor,
        "ContentStatisticsExtractor": ContentStatisticsExtractor,
        "SummaryExtractor": SummaryExtractor,
        "PatternExtractor": PatternExtractor,
        "TableExtractor": TableExtractor,
        "LinkExtractor": LinkExtractor,
        "HeadingExtractor": HeadingExtractor,
    }


class RetrievalStrategyFactory(ComponentFactory):
    """Factory for creating retrieval strategy instances."""

    _registry = {
        "BasicSimilarityStrategy": BasicSimilarityStrategy,
        "HybridUniversalStrategy": HybridUniversalStrategy,
        "MetadataFilteredStrategy": MetadataFilteredStrategy,
        "MultiQueryStrategy": MultiQueryStrategy,
        "RerankedStrategy": RerankedStrategy,
    }


logger = RAGStructLogger("rag.core.factories")


def create_component_from_config(
    component_config: dict[str, Any],
    factory_class: type[ComponentFactory],
    project_dir: Path | None = None,
):
    """Create a component from configuration using the appropriate factory."""
    logger.info(f"Creating component from config: {component_config}")
    component_type = component_config.get("type")
    if not component_type:
        raise ValueError("Component configuration must specify a 'type'")

    config = component_config.get("config", {})
    return factory_class.create(component_type, config, project_dir)


def create_embedder_from_config(embedder_config: dict[str, Any]) -> Embedder:
    """Create an embedder from configuration."""
    return create_component_from_config(embedder_config, EmbedderFactory)


def create_parser_from_config(parser_config: dict[str, Any]) -> Parser:
    """Create a parser from configuration."""
    return create_component_from_config(parser_config, ParserFactory)


def create_vector_store_from_config(
    store_config: dict[str, Any], project_dir: Path
) -> VectorStore:
    """Create a vector store from configuration."""
    return create_component_from_config(store_config, VectorStoreFactory, project_dir)


def create_extractor_from_config(extractor_config: dict[str, Any]):
    """Create an extractor from configuration."""
    return create_component_from_config(extractor_config, ExtractorFactory)


def create_retrieval_strategy_from_config(strategy_config: dict[str, Any]):
    """Create a retrieval strategy from configuration."""
    return create_component_from_config(strategy_config, RetrievalStrategyFactory)
