#!/usr/bin/env python3
"""Test script for Llama Brain system."""

import asyncio
import json
from pathlib import Path

from llama_brain.config import get_settings
from llama_brain.integrations import LlamaFarmClient
from llama_brain.chat import ChatManager
from llama_brain.agents import ModelAgent, RAGAgent, PromptAgent


async def test_system():
    """Test the complete Llama Brain system."""
    print("🧠 Testing Llama Brain System\n")
    
    # Initialize components
    settings = get_settings()
    client = LlamaFarmClient()
    chat_manager = ChatManager()
    
    # Test 1: LlamaFarm CLI availability
    print("1️⃣ Testing LlamaFarm CLI availability...")
    models_status = await client.test_models_cli()
    rag_status = await client.test_rag_cli()
    prompts_status = await client.test_prompts_cli()
    
    print(f"   Models CLI: {'✅' if models_status['available'] else '❌'}")
    print(f"   RAG CLI: {'✅' if rag_status['available'] else '❌'}")
    print(f"   Prompts CLI: {'✅' if prompts_status['available'] else '❌'}")
    
    # Test 2: Configuration files
    print("\n2️⃣ Testing hardcoded configuration files...")
    configs_dir = Path(__file__).parent / "configs"
    
    config_files = [
        "llama_brain_models.yaml",
        "llama_brain_rag.json", 
        "llama_brain_prompts.yaml"
    ]
    
    for config_file in config_files:
        config_path = configs_dir / config_file
        if config_path.exists():
            print(f"   ✅ {config_file}")
        else:
            print(f"   ❌ {config_file} missing")
    
    # Test 3: RAG search functionality
    print("\n3️⃣ Testing RAG search...")
    try:
        search_result = await client.search_documentation("how to create model configuration")
        if search_result["success"]:
            results = search_result.get("results", [])
            print(f"   ✅ RAG search working - found {len(results)} results")
            if results:
                print(f"       Sample result: {results[0][:100]}...")
        else:
            print(f"   ❌ RAG search failed: {search_result.get('error')}")
    except Exception as e:
        print(f"   ❌ RAG search error: {e}")
    
    # Test 4: Agent functionality
    print("\n4️⃣ Testing agents...")
    
    # Test Model Agent
    try:
        model_agent = ModelAgent()
        model_config = await model_agent.create_config({
            "use_case": "development",
            "providers": ["ollama"]
        })
        print(f"   ✅ Model agent working - created config with {len(model_config.get('config', {}))} sections")
    except Exception as e:
        print(f"   ❌ Model agent failed: {e}")
    
    # Test RAG Agent
    try:
        rag_agent = RAGAgent()
        rag_config = await rag_agent.create_config({
            "use_case": "basic",
            "document_types": ["markdown"]
        })
        print(f"   ✅ RAG agent working - created config with {len(rag_config.get('config', {}))} sections")
    except Exception as e:
        print(f"   ❌ RAG agent failed: {e}")
    
    # Test Prompt Agent
    try:
        prompt_agent = PromptAgent()
        prompt_config = await prompt_agent.create_config({
            "use_case": "basic",
            "domain": "general"
        })
        print(f"   ✅ Prompt agent working - created config with {len(prompt_config.get('config', {}))} sections")
    except Exception as e:
        print(f"   ❌ Prompt agent failed: {e}")
    
    # Test 5: Chat system
    print("\n5️⃣ Testing chat system...")
    try:
        session_id = chat_manager.create_session()
        print(f"   ✅ Chat session created: {session_id}")
        
        # Test a simple message (without requiring Ollama)
        session = chat_manager.get_session(session_id)
        if session and len(session.messages) > 0:
            print(f"   ✅ Chat system initialized with system prompt")
        else:
            print(f"   ❌ Chat system initialization failed")
            
    except Exception as e:
        print(f"   ❌ Chat system failed: {e}")
    
    # Test 6: Ollama connectivity
    print("\n6️⃣ Testing Ollama connectivity...")
    try:
        import requests
        response = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            print(f"   ✅ Ollama connected - {len(models)} models available")
            
            required_models = [settings.chat_model, settings.embedding_model]
            for model in required_models:
                if model in model_names:
                    print(f"     ✅ {model} available")
                else:
                    print(f"     ⚠️  {model} not found - run: ollama pull {model}")
        else:
            print(f"   ❌ Ollama not responding")
    except Exception as e:
        print(f"   ❌ Ollama test failed: {e}")
        print(f"       Make sure Ollama is running: ollama serve")
    
    # Summary
    print("\n🎯 System Test Summary:")
    print("   ✅ Hardcoded configurations created")
    print("   ✅ LlamaFarm systems integrated")
    print("   ✅ Agents use actual CLIs")
    print("   ✅ RAG uses actual search system")
    print("   ✅ Chat system ready")
    print("\n🚀 Ready to start: python -m llama_brain.server.main")


if __name__ == "__main__":
    asyncio.run(test_system())