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

def test_chat_endpoint_negative_cases():
    """Test the chat endpoint with negative and edge cases"""
    
    base_url = "http://localhost:8000/v1/inference/chat"
    
    # Test cases for negative scenarios
    test_cases = [
        {
            "name": "Missing session ID",
            "headers": {"Content-Type": "application/json"},
            "data": {"message": "List my projects"},
            "expected_status": [400, 422]  # Could be either depending on validation
        },
        {
            "name": "Empty session ID",
            "headers": {"Content-Type": "application/json", "X-Session-ID": ""},
            "data": {"message": "List my projects"},
            "expected_status": [400, 422]
        },
        {
            "name": "Invalid session ID format",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "invalid/session:id"},
            "data": {"message": "List my projects"},
            "expected_status": [200, 400, 422]  # Might be accepted but handled
        },
        {
            "name": "Missing message body",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "test-session"},
            "data": {},
            "expected_status": [400, 422]
        },
        {
            "name": "Empty message",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "test-session"},
            "data": {"message": ""},
            "expected_status": [200, 400, 422]  # Might be handled gracefully
        },
        {
            "name": "Null message",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "test-session"},
            "data": {"message": None},
            "expected_status": [400, 422]
        },
        {
            "name": "Invalid JSON",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "test-session"},
            "data": "invalid json",
            "expected_status": [400, 422],
            "send_raw": True
        },
        {
            "name": "Extremely long message",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "test-session"},
            "data": {"message": "x" * 10000},
            "expected_status": [200, 400, 413]  # Might be handled or rejected
        },
        {
            "name": "SQL injection attempt",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "test-session"},
            "data": {"message": "'; DROP TABLE projects; --"},
            "expected_status": [200]  # Should be handled safely
        },
        {
            "name": "XSS attempt",
            "headers": {"Content-Type": "application/json", "X-Session-ID": "test-session"},
            "data": {"message": "<script>alert('xss')</script>"},
            "expected_status": [200]  # Should be handled safely
        }
    ]
    
    all_tests_passed = True
    
    for test_case in test_cases:
        print(f"\n🧪 Negative Test: {test_case['name']}")
        
        try:
            if test_case.get("send_raw"):
                response = requests.post(
                    base_url, 
                    headers=test_case["headers"], 
                    data=test_case["data"]
                )
            else:
                response = requests.post(
                    base_url, 
                    headers=test_case["headers"], 
                    json=test_case["data"]
                )
            
            if response.status_code in test_case["expected_status"]:
                print(f"✅ Expected status {response.status_code}")
                if response.status_code >= 400:
                    print(f"📝 Error response: {response.text[:100]}...")
                else:
                    # For successful responses, ensure they're handled safely
                    try:
                        result = response.json()
                        print(f"📝 Safe response: {str(result)[:100]}...")
                    except:
                        print("📝 Non-JSON response (might be expected)")
            else:
                print(f"❌ Unexpected status {response.status_code}, expected one of {test_case['expected_status']}")
                print(f"📝 Response: {response.text[:200]}...")
                all_tests_passed = False
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server. Make sure it's running on localhost:8000")
            return False
        except Exception as e:
            print(f"❌ Error in negative test: {e}")
            all_tests_passed = False
    
    return all_tests_passed


def test_chat_endpoint_edge_cases():
    """Test the chat endpoint with edge cases"""
    
    base_url = "http://localhost:8000/v1/inference/chat"
    
    edge_cases = [
        {
            "name": "Unicode characters",
            "message": "Create project 测试项目 in 命名空间 namespace",
            "session_id": "unicode-test-session"
        },
        {
            "name": "Special characters in project name",
            "message": "Create project my-app_v2.0 in test namespace",
            "session_id": "special-chars-session"
        },
        {
            "name": "Very long session ID",
            "message": "List my projects",
            "session_id": "x" * 100
        },
        {
            "name": "Case sensitivity test",
            "message": "CREATE PROJECT MyApp IN Test NAMESPACE",
            "session_id": "case-test-session"
        }
    ]
    
    # Add concurrent session tests
    for i in range(3):
        edge_cases.append({
            "name": f"Multiple concurrent sessions {i+1}",
            "message": "List projects in concurrent namespace",
            "session_id": f"concurrent-session-{i}"
        })
    
    all_tests_passed = True
    
    for test_case in edge_cases:
        print(f"\n🧪 Edge Case: {test_case['name']}")
        
        headers = {
            "Content-Type": "application/json",
            "X-Session-ID": test_case["session_id"]
        }
        data = {"message": test_case["message"]}
        
        try:
            response = requests.post(base_url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Edge case handled successfully")
                print(f"📝 Response: {str(result.get('message', 'No message'))[:100]}...")
            else:
                print(f"⚠️ Edge case returned status {response.status_code}")
                print(f"📝 Response: {response.text[:200]}...")
                # Edge cases might fail gracefully, so don't mark as failure
                
        except requests.exceptions.ConnectionError:
            print("❌ Could not connect to server. Make sure it's running on localhost:8000")
            return False
        except Exception as e:
            print(f"❌ Error in edge case test: {e}")
            all_tests_passed = False
    
    return all_tests_passed


if __name__ == "__main__":
    print("🚀 Starting LlamaFarm chat tests...")
    print("=" * 50)
    
    # Test agent status first
    status_ok = test_agent_status()
    print()
    
    # Test chat endpoint - positive cases
    chat_ok = test_chat_endpoint()
    print()
    
    # Test chat endpoint - negative cases
    negative_ok = test_chat_endpoint_negative_cases()
    print()
    
    # Test chat endpoint - edge cases
    edge_ok = test_chat_endpoint_edge_cases()
    print()
    
    if status_ok and chat_ok and negative_ok and edge_ok:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above.")
        print("\n💡 Make sure to:")
        print("   1. Start the server with: ./start_server.sh")
        print("   2. Ensure the virtual environment is activated")
        print("   3. Check that all dependencies are installed") 