"""RAG configuration agent."""

from typing import Dict, Any, List
from .base_agent import BaseAgent


class RAGAgent(BaseAgent):
    """Agent specialized in RAG configurations."""
    
    def __init__(self):
        super().__init__("rag")
    
    async def create_config(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Create a RAG configuration using the LlamaFarm RAG system."""
        output_path = self.settings.generated_configs_dir / self.component_name / f"generated_{requirements.get('use_case', 'custom')}.json"
        
        # Use the LlamaFarm client to create the config
        result = await self.client.create_rag_config(requirements, str(output_path))
        
        if result["success"]:
            return {
                "config": result["config"],
                "file_path": result["config_path"],
                "validation_output": result.get("validation_output", ""),
                "success": True
            }
        else:
            raise ValueError(f"Failed to create RAG config: {result.get('error')}")
    
    async def _configure_parser(self, document_types: List[str]) -> Dict[str, Any]:
        """Configure parser based on document types."""
        if len(document_types) == 1:
            doc_type = document_types[0].lower()
            
            if doc_type == "pdf":
                return {
                    "type": "PDFParser",
                    "config": {
                        "extract_images": False,
                        "extract_tables": True,
                        "ocr_enabled": False
                    }
                }
            elif doc_type == "csv":
                return {
                    "type": "CustomerSupportCSVParser",
                    "config": {
                        "content_fields": ["subject", "body"],
                        "metadata_fields": ["type", "priority", "language"],
                        "combine_content": True
                    }
                }
            elif doc_type in ["markdown", "md"]:
                return {
                    "type": "MarkdownParser",
                    "config": {
                        "preserve_structure": True,
                        "extract_frontmatter": True,
                        "extract_headers": True,
                        "chunk_by_sections": True,
                        "min_section_length": 100
                    }
                }
        
        # Default to auto-parser for mixed document types
        return {
            "type": "AutoParser",
            "config": {
                "supported_types": document_types,
                "fallback_to_text": True
            }
        }
    
    async def _configure_embedder(self, use_case: str) -> Dict[str, Any]:
        """Configure embedder based on use case."""
        if use_case == "high_accuracy":
            return {
                "type": "OllamaEmbedder",
                "config": {
                    "model": "nomic-embed-text",
                    "timeout": 60,
                    "batch_size": 8  # Smaller batches for quality
                }
            }
        elif use_case == "fast":
            return {
                "type": "OllamaEmbedder", 
                "config": {
                    "model": "nomic-embed-text",
                    "timeout": 15,
                    "batch_size": 32  # Larger batches for speed
                }
            }
        else:  # basic
            return {
                "type": "OllamaEmbedder",
                "config": {
                    "model": "nomic-embed-text",
                    "timeout": 30,
                    "batch_size": 16
                }
            }
    
    async def _configure_vector_store(self, vector_db: str, use_case: str) -> Dict[str, Any]:
        """Configure vector store based on database and use case."""
        collection_name = f"documents_{use_case}"
        
        if vector_db.lower() == "chromadb":
            persist_dir = f"./chroma_db_{use_case}"
            if use_case == "production":
                persist_dir = "/data/chroma_db"
            
            return {
                "type": "ChromaStore",
                "config": {
                    "collection_name": collection_name,
                    "persist_directory": persist_dir,
                    "distance_metric": "cosine"
                }
            }
        elif vector_db.lower() == "qdrant":
            return {
                "type": "QdrantStore",
                "config": {
                    "collection_name": collection_name,
                    "host": "localhost",
                    "port": 6333,
                    "vector_size": 768,
                    "distance": "Cosine"
                }
            }
        else:
            # Default to ChromaDB
            return {
                "type": "ChromaStore",
                "config": {
                    "collection_name": collection_name,
                    "persist_directory": f"./chroma_db_{use_case}"
                }
            }
    
    async def _configure_retrieval(self, use_case: str) -> Dict[str, Any]:
        """Configure retrieval strategy based on use case."""
        if use_case == "high_accuracy":
            return {
                "strategy": "reranked",
                "top_k": 10,
                "rerank_top_k": 5,
                "metadata_boost": 0.1,
                "recency_boost": 0.05
            }
        elif use_case == "fast":
            return {
                "strategy": "basic_similarity",
                "top_k": 5
            }
        elif use_case == "hybrid":
            return {
                "strategy": "hybrid_universal",
                "top_k": 8,
                "strategies": [
                    {"name": "basic_similarity", "weight": 0.6},
                    {"name": "metadata_filtered", "weight": 0.4}
                ],
                "fusion_method": "reciprocal_rank"
            }
        else:  # basic
            return {
                "strategy": "metadata_filtered",
                "top_k": 5,
                "metadata_filters": {}
            }

    async def edit_config(self, config: Dict[str, Any], changes: Dict[str, Any]) -> Dict[str, Any]:
        """Edit an existing RAG configuration."""
        result = self._merge_configs(config, changes)
        
        # Validate the merged configuration
        validation = await self.validate_config(result)
        if not validation["valid"]:
            raise ValueError(f"Invalid configuration after edit: {validation['errors']}")
        
        return result
    
    async def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a RAG configuration."""
        errors = []
        warnings = []
        
        # Check required sections
        required_sections = ["parser", "embedder", "vector_store"]
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing required section: {section}")
            else:
                section_config = config[section]
                if "type" not in section_config:
                    errors.append(f"Section '{section}' missing 'type' field")
        
        # Validate parser configuration
        if "parser" in config:
            parser_config = config["parser"]
            parser_type = parser_config.get("type")
            
            if parser_type == "PDFParser":
                # Check if PyPDF2 is mentioned (would need to be installed)
                warnings.append("PDFParser requires PyPDF2 to be installed")
            elif parser_type == "MarkdownParser":
                # Validate markdown-specific config
                if "config" in parser_config:
                    md_config = parser_config["config"]
                    if md_config.get("chunk_by_sections") and not md_config.get("min_section_length"):
                        warnings.append("Consider setting min_section_length when chunking by sections")
        
        # Validate embedder configuration
        if "embedder" in config:
            embedder_config = config["embedder"]
            if embedder_config.get("type") == "OllamaEmbedder":
                model = embedder_config.get("config", {}).get("model")
                if not model:
                    errors.append("OllamaEmbedder missing model specification")
        
        # Validate vector store configuration
        if "vector_store" in config:
            vs_config = config["vector_store"]
            vs_type = vs_config.get("type")
            
            if vs_type == "ChromaStore":
                if "collection_name" not in vs_config.get("config", {}):
                    errors.append("ChromaStore missing collection_name")
        
        # Validate retrieval configuration
        if "retrieval" in config:
            retrieval_config = config["retrieval"]
            strategy = retrieval_config.get("strategy")
            
            if strategy == "reranked" and "rerank_top_k" not in retrieval_config:
                warnings.append("Reranked strategy should specify rerank_top_k")
            
            top_k = retrieval_config.get("top_k", 5)
            if top_k > 20:
                warnings.append("Large top_k values may impact performance")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": self._generate_suggestions(config)
        }
    
    def _generate_suggestions(self, config: Dict[str, Any]) -> List[str]:
        """Generate suggestions for improving the RAG configuration."""
        suggestions = []
        
        # Check if using basic similarity only
        retrieval = config.get("retrieval", {})
        if retrieval.get("strategy") == "basic_similarity":
            suggestions.append("Consider using 'reranked' strategy for better accuracy")
        
        # Check parser configuration
        parser = config.get("parser", {})
        if parser.get("type") == "MarkdownParser":
            if not parser.get("config", {}).get("chunk_by_sections"):
                suggestions.append("Enable chunk_by_sections for better document structure preservation")
        
        # Check embedder batch size
        embedder = config.get("embedder", {})
        if embedder.get("type") == "OllamaEmbedder":
            batch_size = embedder.get("config", {}).get("batch_size", 1)
            if batch_size < 8:
                suggestions.append("Consider increasing batch_size for better performance")
        
        # Suggest hybrid approach for complex use cases
        if len(suggestions) == 0:
            suggestions.append("For complex documents, consider hybrid retrieval strategies")
        
        return suggestions