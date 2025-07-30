"""
Type definitions for LlamaFarm configuration based on the JSON schema.
This file is auto-generated from schema.yaml - DO NOT EDIT MANUALLY.
"""

from typing import Literal, TypedDict, Union


class PromptConfig(TypedDict):
    """Configuration for a single prompt."""
    name: str | None
    prompt: str
    description: str | None

class ModelConfig(TypedDict):
    """Configuration for a single model."""
    provider: Literal["openai", "anthropic", "google", "local", "custom"]
    model: str

class ParserConfig(TypedDict):
    """Parser configuration within RAG."""
    content_fields: list[str]
    metadata_fields: list[str]

class EmbedderConfig(TypedDict):
    """Embedder configuration within RAG."""
    model: str
    batch_size: int

class VectorStoreConfig(TypedDict):
    """Vector store configuration within RAG."""
    collection_name: str
    persist_directory: str

class Parser(TypedDict):
    """Parser definition in RAG configuration."""
    type: Literal["CustomerSupportCSVParser"]
    config: ParserConfig

class Embedder(TypedDict):
    """Embedder definition in RAG configuration."""
    type: Literal["OllamaEmbedder"]
    config: EmbedderConfig

class VectorStore(TypedDict):
    """Vector store definition in RAG configuration."""
    type: Literal["ChromaStore"]
    config: VectorStoreConfig

class RAGConfig(TypedDict):
    """RAG (Retrieval-Augmented Generation) configuration."""
    parser: Parser | None
    embedder: Embedder | None
    vector_store: VectorStore | None

class LlamaFarmConfig(TypedDict):
    """Complete LlamaFarm configuration."""
    version: Literal["v1"]
    prompts: list[PromptConfig] | None
    rag: RAGConfig
    models: list[ModelConfig]

# Type alias for the configuration dictionary

ConfigDict = Union[LlamaFarmConfig, dict]
