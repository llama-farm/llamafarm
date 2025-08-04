#!/usr/bin/env python3
"""
Test script to verify chat functionality works correctly
"""

import requests
import json

def test_chat_endpoint():
    """Test the chat endpoint with project requests"""
    
    url = "http://localhost:8000/v1/inference/chat"
    headers = {
        "Content-Type": "application/json",
        "X-Session-ID": "test-session-123"
    }
    
    # Test messages - try both create and list
    test_messages = [
        "Create project testproject in rmo namespace",
        "how many projects are in the rmo namespace?"
    ]
    
    all_tests_passed = True
    
    for i, message in enumerate(test_messages):
        print(f"\n🧪 Test {i+1}: {message}")
        data = {"message": message}
        
        try:
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Chat endpoint responded successfully")
                print(f"📝 Message: {result.get('message', 'No message')[:100]}...")
                print(f"🆔 Session ID: {result.get('session_id', 'No session ID')}")
                
                # Check if tool was executed
                tool_results = result.get('tool_results', [])
                if tool_results:
                    for tool_result in tool_results:
                        if tool_result.get('tool_used') == 'projects':
                            print("✅ Projects tool was used")
                            print(f"🔧 Integration type: {tool_result.get('integration_type')}")
                            break
                    else:
                        print("⚠️ No projects tool found in results")
                        all_tests_passed = False
                else:
                    print("⚠️ No tool results found")
                    all_tests_passed = False
                    
            else:
                print(f"❌ Chat endpoint failed with status {response.status_code}")
                print(f"📝 Response: {response.text}")
                all_tests_passed = False
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server. Make sure it's running on localhost:8000")
            return False
        except Exception as e:
            print(f"❌ Error testing chat endpoint: {e}")
            all_tests_passed = False
    
    return all_tests_passed

def test_agent_status():
    """Test the agent status endpoint"""
    
    url = "http://localhost:8000/v1/inference/agent-status"
    
    try:
        print("🧪 Testing agent status endpoint...")
        response = requests.get(url)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Agent status endpoint responded successfully")
            print(f"🔧 Atomic agents available: {result.get('atomic_agents_available', False)}")
            print(f"🌍 Environment status: {result.get('environment_status', 'unknown')}")
            print(f"🤖 Current model: {result.get('current_model', 'unknown')}")
            print(f"🛠️ Model supports tools: {result.get('model_supports_tools', False)}")
            return True
        else:
            print(f"❌ Agent status endpoint failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Error testing agent status: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting LlamaFarm chat tests...")
    print("=" * 50)
    
    # Test agent status first
    status_ok = test_agent_status()
    print()
    
    # Test chat endpoint
    chat_ok = test_chat_endpoint()
    print()
    
    if status_ok and chat_ok:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")
        print("\n💡 Make sure to:")
        print("   1. Start the server with: ./start_server.sh")
        print("   2. Ensure the virtual environment is activated")
        print("   3. Check that all dependencies are installed") 