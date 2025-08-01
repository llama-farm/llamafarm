"""Settings and configuration for Llama Brain."""

import os
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Server settings
    host: str = Field(default="127.0.0.1", env="HOST")
    port: int = Field(default=8080, env="PORT")
    debug: bool = Field(default=True, env="DEBUG")
    
    # Model settings
    ollama_host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
    chat_model: str = Field(default="llama3.2:3b", env="CHAT_MODEL")
    embedding_model: str = Field(default="nomic-embed-text", env="EMBEDDING_MODEL")
    
    # RAG settings
    rag_collection: str = Field(default="llamafarm_docs", env="RAG_COLLECTION")
    rag_top_k: int = Field(default=5, env="RAG_TOP_K")
    
    # Prompts settings
    prompts_config_path: str = Field(
        default="../../prompts/config/default_prompts.yaml",
        env="PROMPTS_CONFIG_PATH"
    )
    
    # Agent settings
    max_iterations: int = Field(default=10, env="MAX_ITERATIONS")
    
    # Data paths (relative to llama-brain directory)
    project_root: Path = Field(default=Path("../.."))
    rag_data_dir: Path = Field(default=Path("./data/rag"))
    chat_history_dir: Path = Field(default=Path("./data/chat"))
    generated_configs_dir: Path = Field(default=Path("./data/configs"))
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure data directories exist
        self.rag_data_dir.mkdir(parents=True, exist_ok=True)
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
        self.generated_configs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def llamafarm_root(self) -> Path:
        """Get the root path of the LlamaFarm project."""
        return Path(__file__).parent.parent.parent.parent.parent
    
    @property
    def rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return {
            "name": "LlamaFarm Documentation RAG",
            "version": "1.0",
            "parser": {
                "type": "MarkdownParser",
                "config": {
                    "preserve_structure": True,
                    "extract_frontmatter": True,
                    "extract_headers": True,
                    "chunk_by_sections": True,
                    "min_section_length": 100
                }
            },
            "embedder": {
                "type": "OllamaEmbedder",
                "config": {
                    "model": self.embedding_model,
                    "timeout": 30
                }
            },
            "vector_store": {
                "type": "ChromaStore",
                "config": {
                    "collection_name": self.rag_collection,
                    "persist_directory": str(self.rag_data_dir / "chroma_db")
                }
            },
            "retrieval": {
                "strategy": "reranked",
                "top_k": self.rag_top_k
            }
        }


# Global settings instance
_settings: Settings = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings