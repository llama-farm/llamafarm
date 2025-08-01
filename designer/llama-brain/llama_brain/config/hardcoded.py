"""Hardcoded configuration for Llama Brain - no dynamic settings needed."""

from pathlib import Path

# =============================================================================
# HARDCODED SYSTEM CONFIGURATION
# =============================================================================

class HardcodedSettings:
    """Hardcoded settings - no dynamic configuration needed."""
    
    def __init__(self):
        # Server settings
        self.host = "127.0.0.1"
        self.port = 8080
        self.debug = True
        
        # Model settings (hardcoded for Ollama + llama3.2:3b)
        self.ollama_host = "http://localhost:11434"
        self.chat_model = "llama3.2:3b"
        self.embedding_model = "nomic-embed-text"
        
        # RAG settings
        self.rag_collection = "llamafarm_knowledge"
        self.rag_top_k = 5
        
        # Paths (relative to llama-brain directory)
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.rag_data_dir = Path(__file__).parent.parent.parent / "data" / "rag"
        self.chat_history_dir = Path(__file__).parent.parent.parent / "data" / "chat"
        self.generated_configs_dir = Path(__file__).parent.parent.parent / "data" / "configs"
        
        # Ensure directories exist
        self.rag_data_dir.mkdir(parents=True, exist_ok=True)
        self.chat_history_dir.mkdir(parents=True, exist_ok=True)
        self.generated_configs_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def llamafarm_root(self) -> Path:
        """Get the root path of the LlamaFarm project."""
        return self.project_root
    
    @property
    def rag_config(self) -> dict:
        """Hardcoded RAG configuration for local ChromaDB."""
        return {
            "name": "LlamaFarm Knowledge Base",
            "version": "1.0",
            "parser": {
                "type": "MarkdownParser",
                "config": {
                    "preserve_structure": True,
                    "extract_frontmatter": True,
                    "extract_headers": True,
                    "chunk_by_sections": True,
                    "min_section_length": 150
                }
            },
            "embedder": {
                "type": "OllamaEmbedder",
                "config": {
                    "model": self.embedding_model,
                    "base_url": self.ollama_host,
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
_settings = None

def get_settings() -> HardcodedSettings:
    """Get the global hardcoded settings."""
    global _settings
    if _settings is None:
        _settings = HardcodedSettings()
    return _settings


# =============================================================================
# KNOWLEDGE BASE PATHS - Default configs to ingest
# =============================================================================

def get_knowledge_sources() -> list:
    """Get list of default config files to ingest into knowledge base."""
    settings = get_settings()
    
    knowledge_sources = [
        # Default configuration files
        settings.llamafarm_root / "rag" / "config" / "default.yaml",
        settings.llamafarm_root / "models" / "config" / "default.yaml", 
        settings.llamafarm_root / "prompts" / "config" / "default.yaml",
        
        # README files for context
        settings.llamafarm_root / "rag" / "README.md",
        settings.llamafarm_root / "models" / "README.md",
        settings.llamafarm_root / "prompts" / "README.md",
        settings.llamafarm_root / "README.md",
        
        # CLAUDE.md files with comprehensive documentation
        settings.llamafarm_root / "rag" / "CLAUDE.md",
    ]
    
    # Return only files that exist
    return [str(path) for path in knowledge_sources if path.exists()]