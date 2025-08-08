"""Factory classes for creating RAG system components."""

from typing import Dict, Any, Type
from core.base import Parser, Embedder, VectorStore

# Import parsers
from components.parsers.csv_parser.csv_parser import CSVParser, CustomerSupportCSVParser
from components.parsers.markdown_parser.markdown_parser import MarkdownParser
from components.parsers.docx_parser.docx_parser import DocxParser
from components.parsers.text_parser.text_parser import PlainTextParser
from components.parsers.html_parser.html_parser import HtmlParser as HTMLParser
from components.parsers.excel_parser.excel_parser import ExcelParser
from components.parsers.directory_parser.directory_parser import DirectoryParser

# Conditional import for PDF parser
try:
    from components.parsers.pdf_parser.pdf_parser import PDFParser
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Import embedders
from components.embedders.ollama_embedder.ollama_embedder import OllamaEmbedder

# Conditional imports for embedders with dependencies
try:
    from components.embedders.openai_embedder.openai_embedder import OpenAIEmbedder
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from components.embedders.huggingface_embedder.huggingface_embedder import HuggingFaceEmbedder
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

try:
    from components.embedders.sentence_transformer_embedder.sentence_transformer_embedder import SentenceTransformerEmbedder
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
from components.extractors.keyword_extractor.keyword_extractor import YAKEExtractor, RAKEExtractor, TFIDFExtractor
from components.extractors.entity_extractor.entity_extractor import EntityExtractor
from components.extractors.datetime_extractor.datetime_extractor import DateTimeExtractor
from components.extractors.statistics_extractor.statistics_extractor import ContentStatisticsExtractor
from components.extractors.summary_extractor.summary_extractor import SummaryExtractor
from components.extractors.pattern_extractor.pattern_extractor import PatternExtractor
from components.extractors.table_extractor.table_extractor import TableExtractor
from components.extractors.link_extractor.link_extractor import LinkExtractor
from components.extractors.heading_extractor.heading_extractor import HeadingExtractor

# Import retrieval strategies
from components.retrievers.basic_similarity.basic_similarity import BasicSimilarityStrategy

from components.retrievers.hybrid_universal.hybrid_universal import HybridUniversalStrategy
from components.retrievers.metadata_filtered.metadata_filtered import MetadataFilteredStrategy
from components.retrievers.multi_query.multi_query import MultiQueryStrategy
from components.retrievers.reranked.reranked import RerankedStrategy


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
        "MarkdownParser": MarkdownParser,
        "DocxParser": DocxParser,
        "PlainTextParser": PlainTextParser,
        "HTMLParser": HTMLParser,
        "ExcelParser": ExcelParser,
        "DirectoryParser": DirectoryParser,
    }
    
    # Add PDF parser conditionally
    if PDF_AVAILABLE:
        _registry["PDFParser"] = PDFParser


class EmbedderFactory(ComponentFactory):
    """Factory for creating embedder instances."""

    _registry = {
        "OllamaEmbedder": OllamaEmbedder,
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


def create_extractor_from_config(extractor_config: Dict[str, Any]):
    """Create an extractor from configuration."""
    return create_component_from_config(extractor_config, ExtractorFactory)


def create_retrieval_strategy_from_config(strategy_config: Dict[str, Any]):
    """Create a retrieval strategy from configuration."""
    return create_component_from_config(strategy_config, RetrievalStrategyFactory)
