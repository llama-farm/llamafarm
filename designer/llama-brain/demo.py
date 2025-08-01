#!/usr/bin/env python3
"""Demo script for Llama Brain - test the chat functionality."""

import asyncio
import json
import requests
import time
from pathlib import Path


def start_server():
    """Instructions to start the server."""
    print("🚀 Starting Llama Brain Server Demo")
    print("\n1. Start the server in another terminal:")
    print("   cd /Users/robthelen/llamafarm-1/designer/llama-brain")
    print("   uv run python -m llama_brain.server.main")
    print("\n2. The server will start on http://localhost:8080")
    print("\n3. Test the API endpoints:")
    print("   Health check: curl http://localhost:8080/")
    print("   Status: curl http://localhost:8080/status")
    print("   Chat: curl -X POST http://localhost:8080/chat -H 'Content-Type: application/json' -d '{\"message\": \"Help me create a model config for development\"}'")


def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:8080"
    
    print("🧪 Testing API endpoints...")
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Health check passed")
            data = response.json()
            print(f"   Service: {data.get('service')}")
            print(f"   Chat Model: {data.get('chat_model')}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
        print("   Make sure the server is running!")
        return
    
    # Test status
    try:
        response = requests.get(f"{base_url}/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status check passed")
            print(f"   Ollama available: {data.get('ollama_available')}")
            print(f"   Available models: {len(data.get('available_models', []))}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status check error: {e}")
    
    # Test chat
    test_messages = [
        "Hello! Can you help me with LlamaFarm configurations?",
        "I want to create a model configuration for development with Ollama",
        "How do I set up a RAG system for PDF documents?",
        "Show me how to create prompt templates"
    ]
    
    session_id = None
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n🗣️  Test Message {i}: {message}")
        
        try:
            payload = {"message": message}
            if session_id:
                payload["session_id"] = session_id
            
            response = requests.post(f"{base_url}/chat", json=payload)
            
            if response.status_code == 200:
                data = response.json()
                session_id = data.get("session_id")
                print(f"✅ Chat response received (session: {session_id[:8]}...)")
                print(f"   Response: {data.get('message', '')[:200]}...")
                
                if data.get("actions"):
                    print(f"   Actions suggested: {len(data['actions'])}")
                    for action in data["actions"]:
                        print(f"     - {action.get('agent_type')}: {action.get('action')}")
                
                if data.get("suggestions"):
                    print(f"   Suggestions: {data['suggestions'][:2]}")
                    
            else:
                print(f"❌ Chat failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Chat error: {e}")
        
        # Wait between messages
        time.sleep(1)
    
    # Test config generation
    print(f"\n🛠️  Testing config generation...")
    
    config_tests = [
        {
            "agent_type": "model",
            "action": "create",
            "requirements": {
                "use_case": "development",
                "providers": ["ollama"]
            }
        },
        {
            "agent_type": "rag", 
            "action": "create",
            "requirements": {
                "use_case": "basic",
                "document_types": ["markdown"]
            }
        }
    ]
    
    for test in config_tests:
        try:
            response = requests.post(f"{base_url}/config", json=test)
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    print(f"✅ {test['agent_type'].title()} config created")
                    if data.get("file_path"):
                        print(f"   Saved to: {data['file_path']}")
                else:
                    print(f"❌ {test['agent_type'].title()} config failed: {data.get('message')}")
            else:
                print(f"❌ Config request failed: {response.status_code}")
        except Exception as e:
            print(f"❌ Config test error: {e}")


def demo_complete_workflow():
    """Demo a complete workflow."""
    print("\n🎭 Complete Workflow Demo")
    print("="*50)
    
    # This would be a complete demo of:
    # 1. User asks for help with a specific configuration
    # 2. System searches knowledge base
    # 3. System provides contextual response
    # 4. User requests config generation
    # 5. System creates and validates config
    # 6. User gets ready-to-use configuration file
    
    print("Demo workflow:")
    print("1. User: 'I need a RAG system for my legal documents'")
    print("2. Llama Brain searches LlamaFarm documentation")
    print("3. Llama Brain: 'For legal documents, I recommend...'")
    print("4. User: 'Create that configuration'")
    print("5. Llama Brain generates config using RAG CLI")
    print("6. User gets ready-to-use legal_rag_config.json")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_api()
    else:
        start_server()
        print("\n" + "="*50)
        print("After starting the server, run:")
        print("python demo.py test")
        print("="*50)
        demo_complete_workflow()