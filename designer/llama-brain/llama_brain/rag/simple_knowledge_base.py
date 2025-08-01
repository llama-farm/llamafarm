"""Simple knowledge base using hardcoded config and default configs as knowledge."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional

from llama_brain.config.hardcoded import get_settings, get_knowledge_sources


class SimpleKnowledgeBase:
    """Simple RAG knowledge base with hardcoded configuration."""
    
    def __init__(self):
        self.settings = get_settings()
        self.is_initialized = False
        self.knowledge_sources = get_knowledge_sources()
    
    async def initialize(self) -> bool:
        """Initialize the knowledge base by ingesting default configs."""
        if self.is_initialized:
            return True
            
        try:
            print("🧠 Initializing Llama Brain knowledge base...")
            
            # Check if we have knowledge sources
            if not self.knowledge_sources:
                print("⚠️ No knowledge sources found")
                return await self._load_fallback_knowledge()
            
            # Try to use RAG system to ingest knowledge
            success = await self._ingest_with_rag()
            
            if success:
                print(f"✅ Knowledge base initialized with {len(self.knowledge_sources)} sources")
                self.is_initialized = True
                return True
            else:
                print("⚠️ RAG ingestion failed, using fallback knowledge")
                return await self._load_fallback_knowledge()
                
        except Exception as e:
            print(f"❌ Failed to initialize knowledge base: {e}")
            return await self._load_fallback_knowledge()
    
    async def _ingest_with_rag(self) -> bool:
        """Use the actual RAG CLI to ingest knowledge sources."""
        try:
            rag_dir = self.settings.llamafarm_root / "rag"
            
            # Create temporary config file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(self.settings.rag_config, f, indent=2)
                config_path = f.name
            
            try:
                # Use RAG CLI to ingest documents
                cmd = [
                    "python", "cli.py", "ingest"
                ] + self.knowledge_sources + [
                    "--config", config_path,
                    "--batch-size", "10"
                ]
                
                print(f"🔄 Running: {' '.join(cmd)}")
                
                result = subprocess.run(
                    cmd,
                    cwd=rag_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes
                )
                
                if result.returncode == 0:
                    print(f"✅ RAG ingestion successful")
                    print(f"Output: {result.stdout[:200]}...")
                    return True
                else:
                    print(f"❌ RAG ingestion failed: {result.stderr}")
                    return False
                    
            finally:
                # Clean up temp file
                Path(config_path).unlink(missing_ok=True)
                
        except Exception as e:
            print(f"❌ RAG ingestion error: {e}")
            return False
    
    async def _load_fallback_knowledge(self) -> bool:
        """Load fallback knowledge when RAG system fails."""
        print("📚 Loading fallback knowledge base...")
        
        # Read actual content from knowledge sources for fallback
        self.fallback_knowledge = {
            "models": {
                "description": "LlamaFarm Models system with multi-provider support",
                "key_features": [
                    "OpenAI, Anthropic, Ollama, Together AI, Groq, Cohere support",
                    "Automatic fallback chains",
                    "Rate limiting and cost tracking",
                    "Local and cloud model support"
                ],
                "config_examples": {
                    "development": "ollama_llama3_2_3b for local development",
                    "production": "openai_gpt4o_mini for cost-effective production",
                    "high_quality": "anthropic_claude_3_sonnet for complex tasks"
                }
            },
            "rag": {
                "description": "Document processing and retrieval system",
                "key_features": [
                    "Multiple parsers: PDF, CSV, Markdown, JSON",
                    "Multiple vector stores: ChromaDB, Qdrant, Weaviate, Milvus",
                    "Advanced retrieval strategies",
                    "Comprehensive embedding model support"
                ],
                "config_examples": {
                    "basic": "ChromaDB with MarkdownParser for simple setups",
                    "production": "Qdrant with advanced retrieval strategies",
                    "multi_modal": "Weaviate with image and text processing"
                }
            },
            "prompts": {
                "description": "Prompt engineering and template management",
                "key_features": [
                    "Template system with Jinja2 support",
                    "Global prompts and context injection",
                    "Rule-based and context-aware strategies",
                    "Domain-specific templates"
                ],
                "config_examples": {
                    "basic_qa": "Simple question answering template",
                    "medical_qa": "Medical domain-specific prompts",
                    "agent_coordination": "Multi-agent orchestration templates"
                }
            }
        }
        
        self.is_initialized = True
        return True
    
    async def search(self, query: str, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search the knowledge base."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            # Try RAG search first
            return await self._search_with_rag(query, component)
        except Exception as e:
            print(f"RAG search failed: {e}, using fallback")
            return await self._search_fallback(query, component)
    
    async def _search_with_rag(self, query: str, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search using the RAG system."""
        try:
            rag_dir = self.settings.llamafarm_root / "rag"
            
            # Use the actual working YAML config instead of generating JSON
            config_path = self.settings.llamafarm_root / "designer" / "llama-brain" / "configs" / "llama_brain_rag.yaml"
            
            if not config_path.exists():
                print(f"❌ RAG config not found at {config_path}")
                return []
            
            print(f"🔍 [RAG] Using config: {config_path}")
            
            # Build search command using uv run for proper environment
            cmd = [
                "uv", "run", "--active", "python", "cli.py", 
                "--config", str(config_path),
                "search", query,
                "--top-k", str(self.settings.rag_top_k)
            ]
            
            print(f"🔍 [RAG CLI] Running: cd {rag_dir} && {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=rag_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            print(f"🔍 [RAG CLI] Exit code: {result.returncode}")
            if result.stdout:
                print(f"🔍 [RAG CLI] Output:")
                print(result.stdout)
            if result.stderr:
                print(f"🔍 [RAG CLI] Error:")
                print(result.stderr)
            
            if result.returncode == 0:
                # Parse RAG output into structured results
                parsed_results = self._parse_rag_output(result.stdout)
                print(f"🔍 [RAG CLI] Parsed {len(parsed_results)} results")
                return parsed_results
            else:
                print(f"❌ RAG search failed: {result.stderr}")
                return []
                
        except Exception as e:
            print(f"RAG search error: {e}")
            return []
    
    def _parse_rag_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse RAG CLI output into structured results."""
        results = []
        lines = output.split('\n')
        
        current_result = {}
        in_content = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Look for result headers
            if "Result #" in line and "Document:" in line:
                if current_result:
                    results.append(current_result)
                current_result = {"content": "", "metadata": {}}
                # Extract document name
                if "Document:" in line:
                    doc_name = line.split("Document:")[-1].strip()
                    current_result["metadata"]["document"] = doc_name
                in_content = False
                continue
            
            # Look for similarity scores
            if "🎯 Similarity:" in line:
                try:
                    score_part = line.split("🎯 Similarity:")[-1].strip()
                    score_str = score_part.split("(")[0].strip()
                    score = float(score_str)
                    current_result["relevance_score"] = abs(score)  # Make positive for easier interpretation
                except:
                    pass
                continue
            
            # Look for content section
            if "📝 Content Preview:" in line:
                in_content = True
                continue
            
            # Skip separator lines and empty lines
            if line_stripped.startswith("=") or not line_stripped:
                continue
            
            # Collect content lines
            if in_content and current_result:
                if current_result["content"]:
                    current_result["content"] += "\n" + line_stripped
                else:
                    current_result["content"] = line_stripped
        
        # Add final result
        if current_result:
            results.append(current_result)
        
        # Clean up results and add component detection
        for result in results:
            if "metadata" not in result:
                result["metadata"] = {}
            
            # Extract component from document name
            doc_name = result["metadata"].get("document", "")
            if "models" in doc_name.lower():
                result["metadata"]["component"] = "models"
            elif "rag" in doc_name.lower():
                result["metadata"]["component"] = "rag"
            elif "prompts" in doc_name.lower():
                result["metadata"]["component"] = "prompts"
            elif "config" in doc_name.lower():
                result["metadata"]["component"] = "config"
            else:
                result["metadata"]["component"] = "general"
        
        return results[:self.settings.rag_top_k]
    
    async def _search_fallback(self, query: str, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fallback search using simple keyword matching."""
        results = []
        query_lower = query.lower()
        query_words = query_lower.split()
        
        for comp_name, comp_data in self.fallback_knowledge.items():
            if component and comp_name != component:
                continue
            
            # Simple scoring based on keyword matches
            content = f"{comp_data['description']} {' '.join(comp_data['key_features'])} {str(comp_data['config_examples'])}"
            content_lower = content.lower()
            
            score = sum(1 for word in query_words if word in content_lower) / len(query_words)
            
            if score > 0.2:  # Threshold for relevance
                results.append({
                    "content": comp_data['description'],
                    "metadata": {
                        "component": comp_name,
                        "type": "fallback_knowledge",
                        "features": comp_data['key_features'],
                        "examples": comp_data['config_examples']
                    },
                    "relevance_score": score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:self.settings.RAG_TOP_K]
    
    def get_component_info(self, component: str) -> Dict[str, Any]:
        """Get comprehensive information about a component."""
        if not self.is_initialized:
            return {}
            
        if hasattr(self, 'fallback_knowledge') and component in self.fallback_knowledge:
            return self.fallback_knowledge[component]
        
        return {
            "description": f"Information about {component} component",
            "key_features": [],
            "config_examples": {}
        }