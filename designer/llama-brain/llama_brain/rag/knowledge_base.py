"""Knowledge base for RAG-powered assistance."""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import requests

# Add the RAG system to path
RAG_PATH = Path(__file__).parent.parent.parent.parent.parent / "rag"
sys.path.insert(0, str(RAG_PATH))

from llama_brain.config import get_settings


class KnowledgeBase:
    """RAG-powered knowledge base for LlamaFarm documentation."""
    
    def __init__(self):
        self.settings = get_settings()
        self.rag_api_url = None  # Will be set when RAG system is available
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the knowledge base by ingesting documentation."""
        try:
            # Check if we can use the RAG system
            if not await self._check_rag_system():
                print("⚠️ RAG system not available, using fallback knowledge")
                await self._load_fallback_knowledge()
                return True
                
            # Ingest all README files from the project
            await self._ingest_documentation()
            self.is_initialized = True
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize knowledge base: {e}")
            return False
    
    async def _check_rag_system(self) -> bool:
        """Check if the RAG system is available."""
        try:
            # Try to import RAG components
            from core.enhanced_pipeline import EnhancedPipeline
            from core.factories import create_parser_from_config
            return True
        except ImportError as e:
            print(f"RAG system not available: {e}")
            return False
    
    async def _ingest_documentation(self):
        """Ingest all documentation into the RAG system."""
        try:
            from core.enhanced_pipeline import EnhancedPipeline
            
            # Create pipeline from config
            pipeline = EnhancedPipeline.from_config(
                config=self.settings.rag_config,
                base_dir=str(self.settings.llamafarm_root)
            )
            
            # Find all README files
            readme_files = []
            for component in ['rag', 'models', 'prompts', 'config']:
                readme_path = self.settings.llamafarm_root / component / "README.md"
                if readme_path.exists():
                    readme_files.append(str(readme_path))
            
            # Add main README
            main_readme = self.settings.llamafarm_root / "README.md"
            if main_readme.exists():
                readme_files.append(str(main_readme))
            
            if readme_files:
                print(f"📚 Ingesting {len(readme_files)} documentation files...")
                result = pipeline.run(readme_files)
                print(f"✅ Ingested {len(result.documents)} documents")
            else:
                print("⚠️ No documentation files found")
                
        except Exception as e:
            print(f"❌ Failed to ingest documentation: {e}")
            raise
    
    async def _load_fallback_knowledge(self):
        """Load fallback knowledge when RAG system is not available."""
        # Create a simple knowledge base from README content
        self.fallback_knowledge = {
            "models": {
                "description": "Model management system with support for multiple providers",
                "config_structure": {
                    "providers": "Dictionary of model providers (openai, anthropic, ollama, etc.)",
                    "routing": "Request routing and fallback configuration",
                    "monitoring": "Usage tracking and performance monitoring"
                },
                "examples": [
                    "ollama_local.yaml - Local Ollama configuration",
                    "production.yaml - Production multi-provider setup"
                ]
            },
            "rag": {
                "description": "Document processing and retrieval system",
                "config_structure": {
                    "parser": "Document parser configuration (CSV, PDF, Markdown)",
                    "embedder": "Embedding model configuration",
                    "vector_store": "Vector database configuration",
                    "retrieval": "Retrieval strategy configuration"
                },
                "examples": [
                    "basic_config.json - Simple RAG setup",
                    "advanced_retrieval_config.json - Advanced retrieval strategies"
                ]
            },
            "prompts": {
                "description": "Prompt engineering and template management system",
                "config_structure": {
                    "global_prompts": "System-wide prompt settings",
                    "templates": "Prompt template library",
                    "strategies": "Template selection strategies"
                },
                "examples": [
                    "default_prompts.yaml - Default prompt configuration",
                    "domain_specific configs - Medical, legal, code analysis"
                ]
            }
        }
    
    async def search(self, query: str, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the knowledge base for relevant information."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Use the LlamaFarm client for RAG search
            client = LlamaFarmClient()
            result = await client.search_documentation(query)
            
            if result["success"]:
                return result["results"]
            else:
                print(f"RAG search failed: {result.get('error')}")
                return await self._search_fallback(query, component)
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return await self._search_fallback(query, component)
    
    async def _search_rag(self, query: str, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search using the RAG system."""
        try:
            from api import SearchAPI
            
            # Use the RAG search API
            api = SearchAPI(config_path=None, config_dict=self.settings.rag_config)
            results = api.search(
                query=query,
                top_k=self.settings.rag_top_k,
                metadata_filters={"component": component} if component else None
            )
            
            return [
                {
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "relevance_score": getattr(doc, 'score', 0.0)
                }
                for doc in results
            ]
            
        except Exception as e:
            print(f"❌ RAG search failed: {e}")
            return []
    
    async def _search_fallback(self, query: str, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search using fallback knowledge."""
        results = []
        query_lower = query.lower()
        
        # Search through fallback knowledge
        for comp_name, comp_data in self.fallback_knowledge.items():
            if component and comp_name != component:
                continue
                
            # Simple keyword matching
            content = f"{comp_data['description']} {str(comp_data)}"
            if any(keyword in content.lower() for keyword in query_lower.split()):
                results.append({
                    "content": comp_data['description'],
                    "metadata": {
                        "component": comp_name,
                        "type": "fallback_knowledge",
                        "config_structure": comp_data['config_structure'],
                        "examples": comp_data['examples']
                    },
                    "relevance_score": 0.8
                })
        
        return results
    
    async def get_component_examples(self, component: str) -> List[str]:
        """Get example configurations for a component."""
        examples_dir = self.settings.llamafarm_root / component / "config_examples"
        if not examples_dir.exists():
            # Try alternative locations
            alt_locations = [
                self.settings.llamafarm_root / component / "config",
                self.settings.llamafarm_root / component / "examples"
            ]
            for alt_dir in alt_locations:
                if alt_dir.exists():
                    examples_dir = alt_dir
                    break
        
        if examples_dir.exists():
            return [
                str(f.relative_to(self.settings.llamafarm_root))
                for f in examples_dir.glob("*.{yaml,json}")
            ]
        return []
    
    async def get_component_schema(self, component: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a component configuration."""
        # Try to find schema files
        schema_locations = [
            self.settings.llamafarm_root / "config" / "schema.yaml",
            self.settings.llamafarm_root / component / "schema.yaml",
            self.settings.llamafarm_root / component / "config" / "schema.yaml"
        ]
        
        for schema_path in schema_locations:
            if schema_path.exists():
                try:
                    import yaml
                    with open(schema_path) as f:
                        schema = yaml.safe_load(f)
                    return schema
                except Exception as e:
                    print(f"Failed to load schema from {schema_path}: {e}")
        
        return None