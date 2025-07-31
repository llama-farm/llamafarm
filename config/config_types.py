"""
Type definitions for LlamaFarm configuration based on the JSON schema.
This file is auto-generated from schema.yaml - DO NOT EDIT MANUALLY.
"""

from typing import TypedDict, List, Literal, Optional, Union, Dict, Any


class PromptConfig(TypedDict):
    """Configuration for a single prompt."""
    name: Optional[str]
    prompt: str
    description: Optional[str]

class ModelConfig(TypedDict):
    """Configuration for a single model."""
    provider: Literal["openai", "anthropic", "google", "local", "custom"]
    model: str

class DefaultsConfig(TypedDict):
    """Default component selections for RAG."""
    parser: str
    embedder: str
    vector_store: str
    retrieval_strategy: str

class DatasetsConfig(TypedDict):
    """Configuration for a single dataset."""
    name: str
    parser: Optional[str]
    files: List[str]

class RAGConfig(TypedDict):
    """RAG (Retrieval-Augmented Generation) configuration."""
    description: Optional[str]
    parsers: Dict[str, Any]
    embedders: Dict[str, Any]
    vector_stores: Dict[str, Any]
    retrieval_strategies: Dict[str, Any]
    defaults: DefaultsConfig

class LlamaFarmConfig(TypedDict):
    """Complete LlamaFarm configuration."""
    version: Literal["v1"]
    name: str
    prompts: List[PromptConfig]
    rag: RAGConfig
    datasets: List[DatasetsConfig]
    models: List[ModelConfig]

# Type alias for the configuration dictionary

ConfigDict = Union[LlamaFarmConfig, dict]
