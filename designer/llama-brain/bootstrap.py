#!/usr/bin/env python3
"""Bootstrap script for Llama Brain - uses LlamaFarm systems to configure itself."""

import asyncio
import json
import subprocess
import tempfile
import yaml
from pathlib import Path

from llama_brain.config import get_settings
from llama_brain.integrations import LlamaFarmClient


class LlamaBrainBootstrap:
    """Bootstrap Llama Brain using LlamaFarm systems."""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = LlamaFarmClient()
        self.brain_root = Path(__file__).parent
        self.configs_dir = self.brain_root / "configs"
        self.configs_dir.mkdir(exist_ok=True)
    
    async def bootstrap(self):
        """Bootstrap the entire Llama Brain system."""
        print("🧠 Bootstrapping Llama Brain using LlamaFarm systems...")
        
        # Step 1: Test system availability
        print("\n1️⃣ Testing LlamaFarm system availability...")
        await self._test_systems()
        
        # Step 2: Create Llama Brain's own model configuration
        print("\n2️⃣ Creating Llama Brain's model configuration...")
        await self._create_model_config()
        
        # Step 3: Create Llama Brain's RAG configuration
        print("\n3️⃣ Creating Llama Brain's RAG configuration...")
        await self._create_rag_config()
        
        # Step 4: Create Llama Brain's prompt configuration
        print("\n4️⃣ Creating Llama Brain's prompt configuration...")
        await self._create_prompt_config()
        
        # Step 5: Initialize RAG system with documentation
        print("\n5️⃣ Ingesting LlamaFarm documentation...")
        await self._ingest_documentation()
        
        # Step 6: Test the complete system
        print("\n6️⃣ Testing complete system...")
        await self._test_system()
        
        print("\n✅ Llama Brain bootstrap complete!")
        print(f"📁 Configurations saved to: {self.configs_dir}")
        print(f"🚀 Start the server with: python -m llama_brain.server.main")
    
    async def _test_systems(self):
        """Test availability of all LlamaFarm systems."""
        models_status = await self.client.test_models_cli()
        rag_status = await self.client.test_rag_cli()
        prompts_status = await self.client.test_prompts_cli()
        
        print(f"  📊 Models CLI: {'✅' if models_status['available'] else '❌'}")
        if not models_status['available']:
            print(f"     Error: {models_status.get('error', 'Unknown error')}")
        
        print(f"  🔍 RAG CLI: {'✅' if rag_status['available'] else '❌'}")
        if not rag_status['available']:
            print(f"     Error: {rag_status.get('error', 'Unknown error')}")
        
        print(f"  📝 Prompts CLI: {'✅' if prompts_status['available'] else '❌'}")
        if not prompts_status['available']:
            print(f"     Error: {prompts_status.get('error', 'Unknown error')}")
    
    async def _create_model_config(self):
        """Create Llama Brain's model configuration using the models system."""
        print("  Creating model config for Llama Brain...")
        
        # Define requirements for Llama Brain's model config
        requirements = {
            "use_case": "ai_assistant",
            "providers": ["ollama"],
            "primary_model": self.settings.chat_model,
            "embedding_model": self.settings.embedding_model
        }
        
        # Create config using models system
        models_dir = self.settings.llamafarm_root / "models"
        
        # Create custom Llama Brain model config
        config = {
            "name": "Llama Brain Model Configuration",
            "version": "1.0",
            "description": "Model configuration for Llama Brain AI assistant",
            "providers": {
                "chat_model": {
                    "provider": "ollama",
                    "model": self.settings.chat_model,
                    "base_url": self.settings.ollama_host,
                    "timeout": 60,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "use_cases": ["chat", "reasoning", "config_generation"]
                },
                "embedding_model": {
                    "provider": "ollama", 
                    "model": self.settings.embedding_model,
                    "base_url": self.settings.ollama_host,
                    "timeout": 30,
                    "use_cases": ["embeddings", "similarity"]
                }
            },
            "routing": {
                "default_provider": "chat_model",
                "strategy": "use_case_based",
                "rules": [
                    {
                        "condition": "task_type == 'embedding'",
                        "provider": "embedding_model"
                    },
                    {
                        "condition": "task_type == 'chat'",
                        "provider": "chat_model"
                    }
                ]
            },
            "monitoring": {
                "enabled": True,
                "track_usage": True,
                "log_level": "INFO"
            },
            "optimization": {
                "caching": {
                    "enabled": True,
                    "ttl_seconds": 1800,
                    "max_entries": 1000
                },
                "context_management": {
                    "max_context_length": 8192,
                    "context_truncation_strategy": "sliding_window"
                }
            }
        }
        
        # Save config
        config_path = self.configs_dir / "llama_brain_models.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Validate using models CLI
        try:
            result = subprocess.run(
                ["python", "cli.py", "validate", str(config_path)],
                cwd=models_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"  ✅ Model config created and validated: {config_path}")
            else:
                print(f"  ⚠️  Model config created but validation warnings: {result.stderr}")
                
        except Exception as e:
            print(f"  ⚠️  Model config created but couldn't validate: {e}")
    
    async def _create_rag_config(self):
        """Create Llama Brain's RAG configuration using the RAG system."""
        print("  Creating RAG config for Llama Brain...")
        
        rag_dir = self.settings.llamafarm_root / "rag"
        
        # Create Llama Brain's specialized RAG config
        config = {
            "name": "Llama Brain Knowledge Base",
            "version": "1.0",
            "description": "RAG configuration for Llama Brain's knowledge base of LlamaFarm documentation",
            "parser": {
                "type": "MarkdownParser",
                "config": {
                    "preserve_structure": True,
                    "extract_frontmatter": True,
                    "extract_headers": True,
                    "extract_links": True,
                    "extract_code_blocks": True,
                    "chunk_by_sections": True,
                    "min_section_length": 150,
                    "include_code_in_content": True
                }
            },
            "embedder": {
                "type": "OllamaEmbedder",
                "config": {
                    "model": self.settings.embedding_model,
                    "timeout": 45,
                    "batch_size": 8  # Conservative for accuracy
                }
            },
            "vector_store": {
                "type": "ChromaStore",
                "config": {
                    "collection_name": self.settings.rag_collection,
                    "persist_directory": str(self.settings.rag_data_dir / "chroma_db"),
                    "distance_metric": "cosine"
                }
            },
            "retrieval": {
                "strategy": "reranked",
                "top_k": 8,
                "rerank_top_k": self.settings.rag_top_k,
                "metadata_boost": 0.15,
                "recency_boost": 0.05,
                "length_preference": "medium"
            },
            "metadata_enrichment": {
                "extract_component": True,
                "extract_use_case": True,
                "extract_difficulty": True
            }
        }
        
        # Save config
        config_path = self.configs_dir / "llama_brain_rag.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Test config using RAG CLI
        try:
            result = subprocess.run(
                ["python", "cli.py", "info", "--config", str(config_path)],
                cwd=rag_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"  ✅ RAG config created and tested: {config_path}")
            else:
                print(f"  ⚠️  RAG config created but test warnings: {result.stderr}")
                
        except Exception as e:
            print(f"  ⚠️  RAG config created but couldn't test: {e}")
    
    async def _create_prompt_config(self):
        """Create Llama Brain's prompt configuration using the prompts system."""
        print("  Creating prompt config for Llama Brain...")
        
        prompts_dir = self.settings.llamafarm_root / "prompts"
        
        # Create Llama Brain's specialized prompt config
        config = {
            "name": "Llama Brain Prompt Configuration",
            "version": "1.0",
            "description": "Prompt configuration for Llama Brain AI assistant specialized in LlamaFarm configuration",
            "enabled": True,
            "default_strategy": "context_aware_config_assistant",
            
            "global_prompts": {
                "system_identity": {
                    "prompt": "You are Llama Brain, an AI assistant specialized in helping users create and configure LlamaFarm components. You have deep knowledge of Models, RAG systems, and Prompts configuration.",
                    "enabled": True,
                    "apply_to": ["all"]
                },
                "config_expertise": {
                    "prompt": "When helping with configurations, always:\n1. Ask clarifying questions about use case and requirements\n2. Explain the reasoning behind configuration choices\n3. Suggest best practices and optimizations\n4. Warn about common pitfalls",
                    "enabled": True,
                    "apply_to": ["config_generation", "config_assistance"]
                },
                "llamafarm_knowledge": {
                    "prompt": "Base your responses on the official LlamaFarm documentation and examples. Reference specific configuration files and CLI commands when appropriate.",
                    "enabled": True,
                    "apply_to": ["all"]
                }
            },
            
            "templates": {
                "config_assistant": {
                    "template_id": "config_assistant",
                    "name": "Configuration Assistant",
                    "type": "agentic",
                    "template": "# LlamaFarm Configuration Assistant\n\n## Context\n{{ context | format_documents }}\n\n## User Request\n{{ query }}\n\n## Analysis\nBased on the documentation and your request, I'll help you create the appropriate configuration.\n\n## Response\nLet me analyze your requirements:\n\n1. **Component Type**: {{ component_type | default('To be determined') }}\n2. **Use Case**: {{ use_case | default('To be determined') }}\n3. **Requirements**: {{ requirements | default('Need more details') }}\n\n## Recommendation",
                    "variables": ["context", "query", "component_type", "use_case", "requirements"],
                    "metadata": {
                        "category": "agentic",
                        "use_cases": ["config_generation", "assistance"],
                        "output_format": "structured_response"
                    }
                },
                
                "model_config_specialist": {
                    "template_id": "model_config_specialist",
                    "name": "Model Configuration Specialist",
                    "type": "domain_specific",
                    "template": "# Model Configuration Specialist\n\n## Documentation Context\n{{ context | format_documents }}\n\n## Request\n{{ query }}\n\n## Model Configuration Analysis\n\nFor your {{ use_case }} use case, I recommend:\n\n### Provider Selection\n- **Primary**: {{ primary_provider | default('To be determined based on requirements') }}\n- **Reasoning**: {{ provider_reasoning | default('Need to understand your infrastructure and requirements') }}\n\n### Configuration Structure\n```yaml\nname: {{ config_name | default('Custom Model Config') }}\nversion: '1.0'\nproviders:\n  # Configuration will be generated based on your specific needs\n```\n\n### Next Steps",
                    "variables": ["context", "query", "use_case", "primary_provider", "provider_reasoning", "config_name"],
                    "metadata": {
                        "category": "domain_specific",
                        "domain": "models",
                        "specialization": "model_configuration"
                    }
                },
                
                "rag_config_specialist": {
                    "template_id": "rag_config_specialist", 
                    "name": "RAG Configuration Specialist",
                    "type": "domain_specific",
                    "template": "# RAG Configuration Specialist\n\n## Documentation Context\n{{ context | format_documents }}\n\n## Request\n{{ query }}\n\n## RAG Configuration Analysis\n\n### Document Analysis\n- **Document Types**: {{ document_types | default('To be determined') }}\n- **Volume**: {{ document_volume | default('Unknown') }}\n- **Use Case**: {{ use_case | default('General RAG') }}\n\n### Recommended Pipeline\n1. **Parser**: {{ recommended_parser | default('Will determine based on document types') }}\n2. **Embedder**: {{ recommended_embedder | default('Standard Ollama embedder') }}\n3. **Vector Store**: {{ recommended_vector_store | default('ChromaDB for most use cases') }}\n4. **Retrieval Strategy**: {{ recommended_strategy | default('Context-dependent') }}\n\n### Configuration Preview\n```json\n{\n  \"name\": \"{{ config_name | default('Custom RAG Config') }}\",\n  \"parser\": {\n    \"type\": \"{{ recommended_parser }}\"\n  }\n}\n```\n\n### Optimization Suggestions",
                    "variables": ["context", "query", "document_types", "document_volume", "use_case", "recommended_parser", "recommended_embedder", "recommended_vector_store", "recommended_strategy", "config_name"],
                    "metadata": {
                        "category": "domain_specific",
                        "domain": "rag",
                        "specialization": "rag_configuration"
                    }
                },
                
                "prompt_config_specialist": {
                    "template_id": "prompt_config_specialist",
                    "name": "Prompt Configuration Specialist", 
                    "type": "domain_specific",
                    "template": "# Prompt Configuration Specialist\n\n## Documentation Context\n{{ context | format_documents }}\n\n## Request\n{{ query }}\n\n## Prompt Configuration Analysis\n\n### Domain Analysis\n- **Domain**: {{ domain | default('General') }}\n- **Use Case**: {{ use_case | default('Basic prompt management') }}\n- **Complexity**: {{ complexity | default('Standard') }}\n\n### Template Recommendations\n{{ template_recommendations | default('Will suggest based on your specific needs') }}\n\n### Strategy Selection\n- **Recommended Strategy**: {{ recommended_strategy | default('rule_based for simple cases') }}\n- **Reasoning**: {{ strategy_reasoning | default('Based on complexity and requirements') }}\n\n### Configuration Structure\n```yaml\nname: {{ config_name | default('Custom Prompt Config') }}\nversion: '1.0'\ntemplates:\n  # Templates will be customized for your domain\nstrategies:\n  # Strategy configuration based on your needs\n```\n\n### Best Practices",
                    "variables": ["context", "query", "domain", "use_case", "complexity", "template_recommendations", "recommended_strategy", "strategy_reasoning", "config_name"],
                    "metadata": {
                        "category": "domain_specific",
                        "domain": "prompts",
                        "specialization": "prompt_configuration"
                    }
                }
            },
            
            "strategies": {
                "context_aware_config_assistant": {
                    "type": "context_aware",
                    "description": "Context-aware template selection for configuration assistance",
                    "factors": [
                        {"name": "component_detection", "weight": 0.4},
                        {"name": "complexity_analysis", "weight": 0.3},
                        {"name": "use_case_matching", "weight": 0.3}
                    ],
                    "rules": [
                        {
                            "condition": "contains(query, ['model', 'llm', 'provider'])",
                            "template": "model_config_specialist",
                            "confidence": 0.9
                        },
                        {
                            "condition": "contains(query, ['rag', 'retrieval', 'document', 'search'])",
                            "template": "rag_config_specialist",
                            "confidence": 0.9
                        },
                        {
                            "condition": "contains(query, ['prompt', 'template', 'strategy'])",
                            "template": "prompt_config_specialist",
                            "confidence": 0.9
                        }
                    ],
                    "fallback_template": "config_assistant"
                }
            }
        }
        
        # Save config
        config_path = self.configs_dir / "llama_brain_prompts.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        # Test config using prompts CLI
        try:
            result = subprocess.run(
                ["python", "-m", "prompts.cli", "template", "list", "--config", str(config_path)],
                cwd=prompts_dir,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print(f"  ✅ Prompt config created and tested: {config_path}")
                print(f"     Templates available: {result.stdout.count('template_id')}")
            else:
                print(f"  ⚠️  Prompt config created but test warnings: {result.stderr}")
                
        except Exception as e:
            print(f"  ⚠️  Prompt config created but couldn't test: {e}")
    
    async def _ingest_documentation(self):
        """Ingest LlamaFarm documentation using the RAG system."""
        print("  Ingesting LlamaFarm documentation...")
        
        rag_dir = self.settings.llamafarm_root / "rag"
        config_path = self.configs_dir / "llama_brain_rag.json"
        
        # Find all documentation files
        doc_files = []
        
        # README files from each component
        for component in ['rag', 'models', 'prompts', 'config', 'server']:
            readme_path = self.settings.llamafarm_root / component / "README.md"
            if readme_path.exists():
                doc_files.append(str(readme_path))
                print(f"    📄 Found: {component}/README.md")
        
        # Main project README
        main_readme = self.settings.llamafarm_root / "README.md"
        if main_readme.exists():
            doc_files.append(str(main_readme))
            print(f"    📄 Found: README.md")
        
        # CLAUDE.md in RAG (comprehensive documentation)
        claude_md = self.settings.llamafarm_root / "rag" / "CLAUDE.md"
        if claude_md.exists():
            doc_files.append(str(claude_md))
            print(f"    📄 Found: rag/CLAUDE.md")
        
        # Structure documentation
        for component in ['models', 'prompts']:
            structure_path = self.settings.llamafarm_root / component / "STRUCTURE.md"
            if structure_path.exists():
                doc_files.append(str(structure_path))
                print(f"    📄 Found: {component}/STRUCTURE.md")
        
        if doc_files:
            try:
                print(f"    🔄 Ingesting {len(doc_files)} documentation files...")
                
                # Use RAG CLI to ingest documents
                cmd = ["python", "cli.py", "ingest"] + doc_files + ["--config", str(config_path)]
                result = subprocess.run(
                    cmd,
                    cwd=rag_dir,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minutes for ingestion
                )
                
                if result.returncode == 0:
                    print(f"  ✅ Successfully ingested {len(doc_files)} documentation files")
                    # Count documents from output
                    if "documents" in result.stdout:
                        print(f"     {result.stdout}")
                else:
                    print(f"  ❌ Failed to ingest documentation: {result.stderr}")
                    print(f"     Command: {' '.join(cmd)}")
                    
            except Exception as e:
                print(f"  ❌ Exception during ingestion: {e}")
        else:
            print("  ⚠️  No documentation files found to ingest")
    
    async def _test_system(self):
        """Test the complete Llama Brain system."""
        print("  Testing Llama Brain system integration...")
        
        # Test RAG search
        try:
            results = await self.client.search_documentation("how to create a model configuration")
            if results["success"]:
                print(f"  ✅ RAG search working - found {len(results.get('results', []))} results")
            else:
                print(f"  ❌ RAG search failed: {results.get('error')}")
        except Exception as e:
            print(f"  ❌ RAG search test failed: {e}")
        
        # Test Ollama connectivity
        try:
            import requests
            response = requests.get(f"{self.settings.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                print(f"  ✅ Ollama connected - {len(models)} models available")
                
                # Check if required models are available
                model_names = [m["name"] for m in models]
                if self.settings.chat_model in model_names:
                    print(f"    ✅ Chat model '{self.settings.chat_model}' available")
                else:
                    print(f"    ⚠️  Chat model '{self.settings.chat_model}' not found")
                    
                if self.settings.embedding_model in model_names:
                    print(f"    ✅ Embedding model '{self.settings.embedding_model}' available")
                else:
                    print(f"    ⚠️  Embedding model '{self.settings.embedding_model}' not found")
            else:
                print(f"  ❌ Ollama not responding properly")
        except Exception as e:
            print(f"  ❌ Ollama connectivity test failed: {e}")


async def main():
    """Main bootstrap function."""
    bootstrap = LlamaBrainBootstrap()
    await bootstrap.bootstrap()


if __name__ == "__main__":
    asyncio.run(main())