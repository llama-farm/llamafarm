"""
Type definitions for LlamaFarm configuration based on the JSON schema.
This file is auto-generated from schema.yaml - DO NOT EDIT MANUALLY.
"""

from typing import TypedDict, List, Literal, Optional, Union


class PromptConfig(TypedDict):
    """Configuration for a single prompt."""
    name: Optional[str]
    prompt: str
    description: Optional[str]

class ModelConfig(TypedDict):
    """Configuration for a single model."""
    provider: Literal["openai", "anthropic", "google", "local", "custom"]
    model: str

class ParserConfig(TypedDict):
    """Parser configuration within RAG."""
    content_fields: List[str]
    metadata_fields: List[str]

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
    parser: Optional[Parser]
    embedder: Optional[Embedder]
    vector_store: Optional[VectorStore]

class LlamaFarmConfig(TypedDict):
    """Complete LlamaFarm configuration."""
    version: Literal["v1"]
    prompts: Optional[List[PromptConfig]]
    rag: RAGConfig
    models: List[ModelConfig]

# Type alias for the configuration dictionary

ConfigDict = Union[LlamaFarmConfig, dict]
