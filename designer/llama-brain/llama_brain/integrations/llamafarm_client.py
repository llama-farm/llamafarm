"""Client for interacting with LlamaFarm systems."""

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml

from llama_brain.config.hardcoded import get_settings


class LlamaFarmClient:
    """Client for interacting with external LlamaFarm systems."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llamafarm_root = self.settings.llamafarm_root
        # Paths to our hardcoded configs - THIS IS THE META PART!
        self.brain_configs = {
            'models': self.settings.llamafarm_root / 'designer' / 'llama-brain' / 'configs' / 'llama_brain_models.yaml',
            'rag': self.settings.llamafarm_root / 'designer' / 'llama-brain' / 'configs' / 'llama_brain_rag.yaml',
            'prompts': self.settings.llamafarm_root / 'prompts' / 'config' / 'default_prompts.yaml'
        }
    
    async def test_models_cli(self) -> Dict[str, Any]:
        """Test the models CLI system."""
        try:
            models_dir = self.llamafarm_root / "models"
            result = subprocess.run(
                ["python", "cli.py", "--help"],
                cwd=models_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "available": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    async def test_rag_cli(self) -> Dict[str, Any]:
        """Test the RAG CLI system."""
        try:
            rag_dir = self.llamafarm_root / "rag"
            result = subprocess.run(
                ["python", "cli.py", "--help"],
                cwd=rag_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "available": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    async def test_prompts_cli(self) -> Dict[str, Any]:
        """Test the prompts CLI system."""
        try:
            prompts_dir = self.llamafarm_root / "prompts"
            result = subprocess.run(
                ["python", "-m", "prompts.cli", "--help"],
                cwd=prompts_dir,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return {
                "available": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e)
            }
    
    async def create_model_config(self, requirements: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Create a model configuration using the models system."""
        try:
            models_dir = self.llamafarm_root / "models"
            
            # Create a temporary config based on requirements
            config_template = await self._select_model_template(requirements)
            
            # Save template to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(config_template, f, default_flow_style=False)
                temp_config_path = f.name
            
            try:
                # Use the models CLI with OUR hardcoded config (META!)
                result = subprocess.run(
                    ["python", "cli.py", "validate-config", temp_config_path, "--config", str(self.brain_configs['models'])],
                    cwd=models_dir,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    # Copy the validated config to the output path
                    output_file = Path(output_path)
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_config_path, 'r') as src, open(output_file, 'w') as dst:
                        dst.write(src.read())
                    
                    return {
                        "success": True,
                        "config_path": str(output_file),
                        "config": config_template,
                        "validation_output": result.stdout
                    }
                else:
                    return {
                        "success": False,
                        "error": f"Validation failed: {result.stderr}",
                        "config": config_template
                    }
                    
            finally:
                # Clean up temp file
                Path(temp_config_path).unlink(missing_ok=True)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_rag_config(self, requirements: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Create a RAG configuration using the RAG system."""
        try:
            rag_dir = self.llamafarm_root / "rag"
            
            # Create config based on requirements
            config_template = await self._select_rag_template(requirements)
            
            # Save to output path
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                json.dump(config_template, f, indent=2)
            
            # Test the config with RAG CLI using OUR hardcoded config (META!)
            result = subprocess.run(
                ["python", "cli.py", "info", "--config", str(self.brain_configs['rag'])],
                cwd=rag_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "config_path": str(output_file),
                "config": config_template,
                "validation_output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def create_prompt_config(self, requirements: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Create a prompt configuration using the prompts system."""
        try:
            prompts_dir = self.llamafarm_root / "prompts"
            
            # Create config based on requirements
            config_template = await self._select_prompt_template(requirements)
            
            # Save to output path
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w') as f:
                yaml.dump(config_template, f, default_flow_style=False)
            
            # Test the config with prompts CLI using OUR hardcoded config (META!)
            result = subprocess.run(
                ["python", "-m", "prompts.cli", "validate", "templates", "--config", str(self.brain_configs['prompts'])],
                cwd=self.llamafarm_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "config_path": str(output_file),
                "config": config_template,
                "validation_output": result.stdout,
                "error": result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def ingest_documentation(self) -> Dict[str, Any]:
        """Ingest documentation using the RAG system."""
        try:
            rag_dir = self.llamafarm_root / "rag"
            
            # Create a config for documentation ingestion
            docs_config = {
                "name": "LlamaFarm Documentation",
                "parser": {
                    "type": "MarkdownParser",
                    "config": {
                        "preserve_structure": True,
                        "extract_frontmatter": True,
                        "chunk_by_sections": True,
                        "min_section_length": 100
                    }
                },
                "embedder": {
                    "type": "OllamaEmbedder",
                    "config": {
                        "model": self.settings.EMBEDDING_MODEL,
                        "timeout": 30
                    }
                },
                "vector_store": {
                    "type": "ChromaStore",
                    "config": {
                        "collection_name": self.settings.RAG_COLLECTION,
                        "persist_directory": str(self.settings.RAG_DATA_DIR / "chroma_db")
                    }
                }
            }
            
            # Save config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(docs_config, f, indent=2)
                config_path = f.name
            
            try:
                # Find README files
                readme_files = []
                for component in ['rag', 'models', 'prompts', 'config']:
                    readme_path = self.llamafarm_root / component / "README.md"
                    if readme_path.exists():
                        readme_files.append(str(readme_path))
                
                # Add main README
                main_readme = self.llamafarm_root / "README.md"
                if main_readme.exists():
                    readme_files.append(str(main_readme))
                
                # Ingest BOTH the default configs AND readme files (META KNOWLEDGE!)
                all_files = readme_files + [
                    str(self.settings.llamafarm_root / "rag" / "config" / "default.yaml"),
                    str(self.settings.llamafarm_root / "models" / "config" / "default.yaml"),
                    str(self.settings.llamafarm_root / "prompts" / "config" / "default.yaml")
                ]
                
                if all_files:
                    # Use RAG CLI with OUR hardcoded config to ingest default configs (META!)
                    result = subprocess.run(
                        ["python", "cli.py", "ingest"] + all_files + ["--config", str(self.brain_configs['rag'])],
                        cwd=rag_dir,
                        capture_output=True,
                        text=True,
                        timeout=300  # 5 minutes for ingestion
                    )
                    
                    return {
                        "success": result.returncode == 0,
                        "files_ingested": all_files,
                        "output": result.stdout,
                        "error": result.stderr if result.returncode != 0 else None
                    }
                else:
                    return {
                        "success": False,
                        "error": "No README files found to ingest"
                    }
                    
            finally:
                Path(config_path).unlink(missing_ok=True)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    async def search_documentation(self, query: str) -> Dict[str, Any]:
        """Search documentation using the RAG system."""
        try:
            rag_dir = self.llamafarm_root / "rag"
            
            # Create search config
            search_config = {
                "name": "Documentation Search",
                "embedder": {
                    "type": "OllamaEmbedder",
                    "config": {
                        "model": self.settings.EMBEDDING_MODEL,
                        "timeout": 30
                    }
                },
                "vector_store": {
                    "type": "ChromaStore",
                    "config": {
                        "collection_name": self.settings.RAG_COLLECTION,
                        "persist_directory": str(self.settings.RAG_DATA_DIR / "chroma_db")
                    }
                },
                "retrieval": {
                    "strategy": "reranked",
                    "top_k": self.settings.RAG_TOP_K
                }
            }
            
            # Save config to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(search_config, f, indent=2)
                config_path = f.name
            
            try:
                # Use RAG CLI to search with OUR hardcoded config (META!)
                result = subprocess.run(
                    ["python", "cli.py", "--config", str(self.brain_configs['rag']), "search", query, "--top-k", str(self.settings.RAG_TOP_K)],
                    cwd=rag_dir,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode == 0:
                    # Parse the search results
                    search_results = []
                    output_lines = result.stdout.split('\n')
                    
                    current_result = {}
                    for line in output_lines:
                        if line.startswith("Document"):
                            if current_result:
                                search_results.append(current_result)
                            current_result = {"content": "", "metadata": {}}
                        elif line.strip():
                            if "content" not in current_result:
                                current_result["content"] = line
                            else:
                                current_result["content"] += "\n" + line
                    
                    if current_result:
                        search_results.append(current_result)
                    
                    return {
                        "success": True,
                        "results": search_results,
                        "query": query,
                        "raw_output": result.stdout
                    }
                else:
                    return {
                        "success": False,
                        "error": result.stderr,
                        "query": query
                    }
                    
            finally:
                Path(config_path).unlink(missing_ok=True)
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _select_model_template(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate model template based on requirements."""
        use_case = requirements.get("use_case", "development")
        
        # Use existing example configs as templates
        if use_case == "production":
            return await self._load_model_example("production")
        elif use_case == "multi_provider":
            return await self._load_model_example("demo_multi_config")
        elif use_case == "ollama":
            return await self._load_model_example("demo_ollama_config")
        else:
            return await self._load_model_example("development")
    
    async def _select_rag_template(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate RAG template based on requirements."""
        document_types = requirements.get("document_types", ["markdown"])
        use_case = requirements.get("use_case", "basic")
        
        # Use existing example configs as templates
        if "markdown" in document_types:
            return await self._load_rag_example("markdown_config")
        elif "pdf" in document_types:
            return await self._load_rag_example("pdf_config")
        elif use_case == "advanced":
            return await self._load_rag_example("advanced_retrieval_config")
        else:
            return await self._load_rag_example("basic_config")
    
    async def _select_prompt_template(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Select appropriate prompt template based on requirements."""
        domain = requirements.get("domain", "general")
        use_case = requirements.get("use_case", "basic")
        
        # Use existing example configs as templates
        if domain == "medical":
            return await self._load_prompt_example("medical_config")
        elif domain == "legal":
            return await self._load_prompt_example("legal_config")
        elif use_case == "advanced":
            return await self._load_prompt_example("context_aware_config")
        else:
            return await self._load_prompt_example("simple_qa_config")
    
    async def _load_model_example(self, example_name: str) -> Dict[str, Any]:
        """Load a model example configuration."""
        try:
            models_dir = self.llamafarm_root / "models"
            config_paths = [
                models_dir / "config" / f"{example_name}.yaml",
                models_dir / f"{example_name}.yaml",
                models_dir / "config_examples" / f"{example_name}.yaml"
            ]
            
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        return yaml.safe_load(f)
            
            # Fallback to basic config
            return {
                "name": "Basic Model Config",
                "providers": {
                    "primary": {
                        "provider": "ollama",
                        "model": "llama3.2:3b",
                        "base_url": "http://localhost:11434"
                    }
                }
            }
            
        except Exception:
            return {
                "name": "Fallback Model Config",
                "providers": {
                    "primary": {
                        "provider": "ollama",
                        "model": "llama3.2:3b"
                    }
                }
            }
    
    async def _load_rag_example(self, example_name: str) -> Dict[str, Any]:
        """Load a RAG example configuration."""
        try:
            rag_dir = self.llamafarm_root / "rag"
            config_paths = [
                rag_dir / "config_examples" / f"{example_name}.json",
                rag_dir / "config_examples" / f"{example_name}.yaml"
            ]
            
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        if config_path.suffix == '.json':
                            return json.load(f)
                        else:
                            return yaml.safe_load(f)
            
            # Fallback to basic config
            return {
                "name": "Basic RAG Config",
                "parser": {"type": "MarkdownParser"},
                "embedder": {"type": "OllamaEmbedder"},
                "vector_store": {"type": "ChromaStore"}
            }
            
        except Exception:
            return {
                "name": "Fallback RAG Config",
                "parser": {"type": "MarkdownParser"},
                "embedder": {"type": "OllamaEmbedder"},
                "vector_store": {"type": "ChromaStore"}
            }
    
    async def _load_prompt_example(self, example_name: str) -> Dict[str, Any]:
        """Load a prompt example configuration."""
        try:
            prompts_dir = self.llamafarm_root / "prompts"
            config_paths = [
                prompts_dir / "config_examples" / "domain_specific" / f"{example_name}.yaml",
                prompts_dir / "config_examples" / "basic" / f"{example_name}.yaml",
                prompts_dir / "config" / f"{example_name}.yaml"
            ]
            
            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path) as f:
                        return yaml.safe_load(f)
            
            # Fallback to basic config
            return {
                "name": "Basic Prompt Config",
                "templates": {
                    "qa_basic": {
                        "template_id": "qa_basic",
                        "name": "Basic QA",
                        "template": "Question: {{ query }}\nAnswer:"
                    }
                }
            }
            
        except Exception:
            return {
                "name": "Fallback Prompt Config",
                "templates": {
                    "basic": {
                        "template_id": "basic",
                        "template": "{{ query }}"
                    }
                }
            }