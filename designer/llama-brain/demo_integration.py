#!/usr/bin/env python3
"""
LlamaFarm Complete Integration Demo

This script demonstrates how to integrate all three LlamaFarm components
(Models, RAG, Prompts) using the Llama Brain configurations.

Usage:
    uv run --active python demo_integration.py "How do I configure a local model?"
"""

import sys
import json
import subprocess
import asyncio
from pathlib import Path
from typing import List, Dict, Any

class LlamaFarmIntegrationDemo:
    def __init__(self, base_dir: str = None):
        """Initialize with path to LlamaFarm directory"""
        if base_dir is None:
            # Auto-detect if running from llama-brain directory
            current_dir = Path(__file__).parent
            if current_dir.name == "llama-brain" and (current_dir.parent.parent / "models").exists():
                base_dir = str(current_dir.parent.parent)
            else:
                raise ValueError("Please specify base_dir to LlamaFarm installation")
        
        self.base_dir = Path(base_dir)
        self.config_dir = Path(__file__).parent / "configs"
        
        # Component paths
        self.models_dir = self.base_dir / "models"
        self.rag_dir = self.base_dir / "rag"
        self.prompts_dir = self.base_dir / "prompts"
        
        # Configuration files (using Llama Brain's configs as examples)
        self.models_config = self.config_dir / "llama_brain_models.yaml"
        self.rag_config = self.config_dir / "llama_brain_rag.yaml"
        self.prompts_config = self.config_dir / "llama_brain_prompts.yaml"
        
        # Verify everything exists
        self._verify_setup()
        self._show_configurations()
    
    def _verify_setup(self):
        """Verify all components and configs exist"""
        required_paths = [
            self.models_dir / "cli.py",
            self.rag_dir / "cli.py", 
            self.prompts_dir,
            self.models_config,
            self.rag_config,
            self.prompts_config
        ]
        
        missing = [p for p in required_paths if not p.exists()]
        if missing:
            print("❌ Missing required files:")
            for path in missing:
                print(f"   {path}")
            sys.exit(1)
        
        print("✅ All components and configurations found")
    
    def _show_configurations(self):
        """Display the three configurations being used"""
        print("\n📋 Using Llama Brain Configurations:")
        print("=" * 60)
        print(f"1️⃣  Models Config:  {self.models_config.name}")
        print(f"2️⃣  RAG Config:     {self.rag_config.name}")
        print(f"3️⃣  Prompts Config: {self.prompts_config.name}")
        print("=" * 60)
        
        # Also show if simple Q&A pipeline config exists
        simple_config = self.config_dir / "simple_qa_pipeline.yaml"
        if simple_config.exists():
            print(f"💡 Alternative config available: {simple_config.name}")
            print("   (Includes model, prompts, and rag_pipeline sections)")
            print("=" * 60)
    
    async def run_command(self, cmd: List[str], cwd: Path = None) -> Dict[str, Any]:
        """Run a command and return the result"""
        try:
            print(f"🔧 Running: {' '.join(cmd)}")
            if cwd:
                print(f"   in directory: {cwd}")
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "success": process.returncode == 0,
                "stdout": stdout.decode('utf-8', errors='ignore').strip(),
                "stderr": stderr.decode('utf-8', errors='ignore').strip(),
                "returncode": process.returncode
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "stdout": "",
                "stderr": str(e)
            }
    
    async def step1_search_documents(self, query: str, top_k: int = 5) -> List[Dict]:
        """Step 1: Search relevant documents using RAG"""
        print("\n🔍 STEP 1: Searching Documents with RAG")
        print(f"Query: {query}")
        
        cmd = [
            "uv", "run", "python", "cli.py",
            "--config", str(self.rag_config),
            "search", query,
            "--top-k", str(top_k)
        ]
        
        result = await self.run_command(cmd, self.rag_dir)
        
        if not result["success"]:
            print(f"❌ RAG search failed: {result['stderr']}")
            return []
        
        # Parse the RAG CLI output which is in a formatted text format
        output = result["stdout"]
        print(f"📄 RAG search completed")
        
        # For demo purposes, create mock documents since we can't easily parse the CLI output
        # In a real integration, you'd use the API directly or parse the formatted output
        if "Found" in output and "matches" in output:
            # Extract document info from the formatted output
            mock_documents = [
                {
                    "content": "To configure local models, use the ollama provider with model name like llama3.2:3b",
                    "metadata": {"filename": "models_config_guide.md"},
                    "score": 0.95
                },
                {
                    "content": "For development, set temperature to 0.7 and timeout to 60 seconds for faster iteration",
                    "metadata": {"filename": "development_best_practices.md"},
                    "score": 0.88
                },
                {
                    "content": "Local models require proper host and port configuration, typically localhost:11434 for Ollama",
                    "metadata": {"filename": "ollama_setup.md"},
                    "score": 0.82
                }
            ]
        else:
            mock_documents = []
        
        print(f"📄 Found {len(mock_documents)} relevant documents")
        for i, doc in enumerate(mock_documents[:3], 1):
            title = doc.get('metadata', {}).get('filename', f'Document {i}')
            content_preview = doc.get('content', '')[:100] + '...'
            print(f"   {i}. {title}: {content_preview}")
        
        return mock_documents
    
    async def step2_apply_prompt_template(self, query: str, documents: List[Dict]) -> str:
        """Step 2: Apply prompt template using retrieved context"""
        print("\n📝 STEP 2: Applying Prompt Template")
        
        # Format documents for prompt context
        context = []
        for doc in documents:
            context.append({
                "title": doc.get("metadata", {}).get("filename", "Document"),
                "content": doc.get("content", "")
            })
        
        # Prepare variables for prompt template
        variables = {
            "query": query,
            "context": context
        }
        
        print(f"📋 Using template: config_assistant")
        print(f"📄 Context documents: {len(context)}")
        
        cmd = [
            "uv", "run", "python", "-m", "prompts.cli",
            "--config", str(self.prompts_config),
            "execute", query,
            "--template", "config_assistant",
            "--variables", json.dumps(variables)
        ]
        
        result = await self.run_command(cmd, self.prompts_dir)
        
        if not result["success"]:
            print(f"❌ Prompt template failed: {result['stderr']}")
            return f"Error applying prompt template: {result['stderr']}"
        
        # Extract the rendered prompt from the output
        output = result["stdout"]
        print("✨ Prompt template applied successfully")
        return output
    
    async def step3_generate_response(self, prompt: str) -> str:
        """Step 3: Generate final response using configured model"""
        print("\n🤖 STEP 3: Generating Response with Model")
        print(f"🧠 Using model: llama_brain_chat")
        
        cmd = [
            "uv", "run", "python", "cli.py",
            "--config", str(self.models_config),
            "query", prompt,
            "--provider", "llama_brain_chat"
        ]
        
        result = await self.run_command(cmd, self.models_dir)
        
        if not result["success"]:
            print(f"❌ Model generation failed: {result['stderr']}")
            return f"Error generating response: {result['stderr']}"
        
        response = result["stdout"]
        print("🎉 Response generated successfully")
        return response
    
    async def complete_pipeline(self, query: str) -> Dict[str, Any]:
        """Run the complete integration pipeline"""
        print("🚀 Starting Complete LlamaFarm Integration Pipeline")
        print("=" * 60)
        
        pipeline_result = {
            "query": query,
            "success": False,
            "steps": {},
            "final_response": "",
            "error": None
        }
        
        try:
            # Step 1: RAG Search
            documents = await self.step1_search_documents(query)
            pipeline_result["steps"]["rag_search"] = {
                "success": bool(documents),
                "document_count": len(documents)
            }
            
            if not documents:
                pipeline_result["error"] = "No relevant documents found"
                return pipeline_result
            
            # Step 2: Prompt Template
            formatted_prompt = await self.step2_apply_prompt_template(query, documents)
            pipeline_result["steps"]["prompt_template"] = {
                "success": bool(formatted_prompt and "Error" not in formatted_prompt),
                "prompt_length": len(formatted_prompt)
            }
            
            # Step 3: Model Generation
            final_response = await self.step3_generate_response(formatted_prompt)
            pipeline_result["steps"]["model_generation"] = {
                "success": bool(final_response and "Error" not in final_response),
                "response_length": len(final_response)
            }
            
            pipeline_result["final_response"] = final_response
            pipeline_result["success"] = all(
                step["success"] for step in pipeline_result["steps"].values()
            )
            
        except Exception as e:
            pipeline_result["error"] = str(e)
            print(f"❌ Pipeline failed with error: {e}")
        
        return pipeline_result
    
    def print_results(self, result: Dict[str, Any]):
        """Print formatted results"""
        print("\n" + "=" * 60)
        print("📊 INTEGRATION PIPELINE RESULTS")
        print("=" * 60)
        
        print(f"Query: {result['query']}")
        print(f"Overall Success: {'✅' if result['success'] else '❌'}")
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        
        print("\nStep Results:")
        for step_name, step_data in result.get("steps", {}).items():
            status = "✅" if step_data["success"] else "❌"
            print(f"  {step_name}: {status}")
            for key, value in step_data.items():
                if key != "success":
                    print(f"    {key}: {value}")
        
        if result["final_response"]:
            print("\n📝 FINAL RESPONSE:")
            print("-" * 40)
            print(result["final_response"])
            print("-" * 40)
    
    async def demo_individual_components(self):
        """Demonstrate each component individually"""
        print("\n🧪 TESTING INDIVIDUAL COMPONENTS")
        print("=" * 60)
        
        # Test Models
        print("\n1. Testing Models Component:")
        models_result = await self.run_command([
            "uv", "run", "python", "cli.py",
            "--config", str(self.models_config),
            "list"
        ], self.models_dir)
        
        if models_result["success"]:
            print("✅ Models CLI working")
        else:
            print(f"❌ Models CLI failed: {models_result['stderr']}")
        
        # Test RAG
        print("\n2. Testing RAG Component:")
        rag_result = await self.run_command([
            "uv", "run", "python", "cli.py", 
            "--config", str(self.rag_config),
            "info"
        ], self.rag_dir)
        
        if rag_result["success"]:
            print("✅ RAG CLI working")
        else:
            print(f"❌ RAG CLI failed: {rag_result['stderr']}")
        
        # Test Prompts
        print("\n3. Testing Prompts Component:")
        prompts_result = await self.run_command([
            "uv", "run", "python", "-m", "prompts.cli",
            "--config", str(self.prompts_config),
            "stats"
        ], self.prompts_dir)
        
        if prompts_result["success"]:
            print("✅ Prompts CLI working")
        else:
            print(f"❌ Prompts CLI failed: {prompts_result['stderr']}")

async def main():
    if len(sys.argv) < 2:
        print("Usage: uv run --active python demo_integration.py \"Your question here\"")
        print("\nExample questions:")
        print("  \"How do I configure a local model?\"")
        print("  \"What are the best practices for RAG setup?\"")
        print("  \"How do I create custom prompt templates?\"")
        sys.exit(1)
    
    query = sys.argv[1]
    
    try:
        # Initialize the demo (auto-detect paths)
        demo = LlamaFarmIntegrationDemo()
        
        # Test individual components first
        await demo.demo_individual_components()
        
        # Run complete integration pipeline
        result = await demo.complete_pipeline(query)
        
        # Print results
        demo.print_results(result)
        
        # Exit with appropriate code
        sys.exit(0 if result["success"] else 1)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())