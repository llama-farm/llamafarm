import uuid
import json
import re
from typing import Dict, Optional, List
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from atomic_agents.context import ChatHistory
from atomic_agents.agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.base import BaseIOSchema
import instructor
from openai import OpenAI
from core.config import settings
from services.tool_service import ToolService

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    session_id: str
    tool_results: Optional[List[Dict]] = None

# Store agent instances to maintain conversation context
# In production, use Redis, database, or other persistent storage
agent_sessions: Dict[str, AtomicAgent] = {}

def get_tools_system_prompt() -> str:
    """Generate a system prompt that includes information about available tools."""
    tools = ToolService.get_available_tools()
    
    if not tools:
        return "You are a helpful assistant for LlamaFarm. You can help with various tasks."
    
    tools_info = []
    for tool in tools:
        tools_info.append(f"- {tool['name']}: {tool['description']}")
        # Add more detailed information about the tool
        if tool['name'] == 'projects':
            tools_info.append("  Actions: list, create")
            tools_info.append("  Parameters:")
            tools_info.append("    - action: 'list' or 'create'")
            tools_info.append("    - namespace: string (required)")
            tools_info.append("    - project_id: string (required for create action)")
    
    tools_text = "\n".join(tools_info)
    
    return f"""You are a helpful assistant for LlamaFarm. You have access to the following tools:

{tools_text}

CRITICAL INSTRUCTIONS:
1. When a user asks you to list projects, you MUST respond with: [TOOL_CALL:projects:{{"action": "list", "namespace": "test"}}]
2. When a user asks you to create a project, you MUST respond with: [TOOL_CALL:projects:{{"action": "create", "namespace": "test", "project_id": "project_name"}}]
3. You MUST use the exact format: [TOOL_CALL:tool_name:{{"param": "value"}}]
4. You MUST include the tool call in your response when the user asks for project-related tasks
5. After the tool call, provide a brief explanation of what you're doing

Examples of correct responses:
- User: "List my projects" → You: "[TOOL_CALL:projects:{{"action": "list", "namespace": "test"}}] I'll list your projects for you."
- User: "Create a new project called my_project" → You: "[TOOL_CALL:projects:{{"action": "create", "namespace": "test", "project_id": "my_project"}}] I'll create a new project called my_project for you."

Remember: ALWAYS use the tool call format when handling project requests!"""

def create_agent() -> AtomicAgent:
    """Create a new agent instance"""
    # Initialize memory
    memory = ChatHistory()
    
    # Create system prompt with tool information
    system_prompt = get_tools_system_prompt()
    
    # Initialize memory with an initial message from the assistant
    initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today? I can help you manage your LlamaFarm configurations and projects.")
    memory.add_message("assistant", initial_message)
    
    # Create OpenAI-compatible client pointing to Ollama
    ollama_client = OpenAI(
        base_url=settings.ollama_host,
        api_key=settings.ollama_api_key,  # Ollama doesn't require a real API key, but instructor needs something
    )
    
    client = instructor.from_openai(ollama_client)

    # Agent setup with specified configuration
    agent = AtomicAgent(
        config=AgentConfig(
            client=client,
            model=settings.ollama_model,  # Using Ollama model name (make sure this model is installed)
            history=memory,
            system_role="system",
            system_prompt_generator=None,  # We'll handle system prompt manually
        )
    )
    
    # Add system prompt to memory
    memory.add_message("system", BasicChatOutputSchema(chat_message=system_prompt))
    
    return agent

def extract_tool_calls(message: str) -> list:
    """
    Extract tool calls from the agent's response message.
    Looks for patterns like:
    - [TOOL_CALL:update_config_name:{"project_id": "test", "config_type": "prompts", "old_name": "old", "new_name": "new"}]
    - [TOOL_CALL:projects:{"action": "list", "namespace": "test"}]
    """
    print(f"🔍 [Inference] extract_tool_calls called with message: {message[:200]}...")
    tool_calls = []
    
    # More flexible pattern that handles nested JSON
    pattern = r'\[TOOL_CALL:([^:]+):(\{[^}]*(?:\{[^}]*\}[^}]*)*\})\]'
    
    matches = re.findall(pattern, message)
    print(f"🔍 [Inference] Found {len(matches)} tool call matches: {matches}")
    
    for tool_name, params_str in matches:
        try:
            # Clean up the parameters string
            params_str = params_str.strip()
            
            # Handle potential JSON parsing issues
            if not params_str.startswith('{'):
                print(f"❌ [Inference] Invalid JSON format: {params_str}")
                continue
                
            params = json.loads(params_str)
            tool_call = {
                "tool_name": tool_name,
                "parameters": params
            }
            tool_calls.append(tool_call)
            print(f"✅ [Inference] Parsed tool call: {tool_call}")
        except json.JSONDecodeError as e:
            print(f"❌ [Inference] Failed to parse tool call '{params_str}': {e}")
            # Try to fix common JSON issues
            try:
                # Remove trailing commas and fix quotes
                fixed_params = re.sub(r',\s*}', '}', params_str)
                fixed_params = re.sub(r',\s*]', ']', fixed_params)
                params = json.loads(fixed_params)
                tool_call = {
                    "tool_name": tool_name,
                    "parameters": params
                }
                tool_calls.append(tool_call)
                print(f"✅ [Inference] Fixed and parsed tool call: {tool_call}")
            except json.JSONDecodeError as e2:
                print(f"❌ [Inference] Still failed to parse after fixes: {e2}")
                continue
        except Exception as e:
            print(f"❌ [Inference] Unexpected error parsing tool call: {e}")
            continue
    
    print(f"📋 [Inference] Returning {len(tool_calls)} valid tool calls")
    return tool_calls

def execute_tool_calls(tool_calls: list) -> list:
    """
    Execute tool calls and return results.
    """
    print(f"🚀 [Inference] execute_tool_calls called with {len(tool_calls)} tool calls")
    results = []
    
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]
        
        print(f"🔧 [Inference] Executing tool call {i+1}/{len(tool_calls)}: {tool_name} with params {parameters}")
        
        result = ToolService.execute_tool(tool_name, **parameters)
        
        tool_result = {
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result
        }
        results.append(tool_result)
        print(f"✅ [Inference] Tool call {i+1} completed: {result}")
    
    print(f"📊 [Inference] All tool calls completed. Returning {len(results)} results")
    return results

def remove_tool_calls_from_message(message: str) -> str:
    """
    Remove tool call patterns from the message to get clean text.
    """
    pattern = r'\[TOOL_CALL:[^:]+:\{[^}]+\}\]'
    return re.sub(pattern, '', message).strip()

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Send a message to the chat agent with tool calling capabilities"""
    print(f"💬 [Inference] /chat endpoint called with message: '{request.message[:100]}...'")
    print(f"🆔 [Inference] Session ID: {session_id}")
    
    try:
        # If no session ID provided, create a new one
        if not session_id or session_id not in agent_sessions:
            if not session_id:
                session_id = str(uuid.uuid4())
            print(f"🆕 [Inference] Creating new agent session: {session_id}")
            agent = create_agent()
            agent_sessions[session_id] = agent
        else:
            # Use existing agent to maintain conversation context
            print(f"🔄 [Inference] Using existing agent session: {session_id}")
            agent = agent_sessions[session_id]

        # Process the user's input through the agent and get the response
        print(f"🤖 [Inference] Running agent with message: '{request.message[:100]}...'")
        input_schema = BasicChatInputSchema(chat_message=request.message)
        response = agent.run(input_schema)
        
        # Extract the message from the response
        if hasattr(response, 'chat_message'):
            response_message = response.chat_message
        else:
            # Fallback if the response structure is different
            response_message = str(response)
        
        print(f"📤 [Inference] Agent response: {response_message[:500]}...")
        
        # Check for tool calls in the response
        print(f"🔍 [Inference] Checking for tool calls in response message...")
        tool_calls = extract_tool_calls(response_message)
        tool_results = None
        
        if tool_calls:
            print(f"🔧 [Inference] Found {len(tool_calls)} tool calls, executing...")
            # Execute the tool calls
            tool_results = execute_tool_calls(tool_calls)
            
            # Remove tool call patterns from the message
            response_message = remove_tool_calls_from_message(response_message)
            print(f"🧹 [Inference] Removed tool calls from message, new length: {len(response_message)}")
            
            # Add tool results to the response message
            if tool_results:
                result_summary = []
                for result in tool_results:
                    if result["result"]["success"]:
                        result_summary.append(f"✅ {result['result']['message']}")
                    else:
                        result_summary.append(f"❌ {result['result']['message']}")
                
                if result_summary:
                    response_message += "\n\n" + "\n".join(result_summary)
                    print(f"📝 [Inference] Added {len(result_summary)} result summaries to response")
        else:
            print(f"📭 [Inference] No tool calls found in response")
            print(f"🔍 [Inference] Full response for debugging: {response_message}")
        
        # Debug the response structure
        print(f"📤 [Inference] Final response message length: {len(response_message)}")
        print(f"🔧 [Inference] Tool results type: {type(tool_results)}")
        if tool_results:
            print(f"🔧 [Inference] Tool results count: {len(tool_results)}")
            for i, result in enumerate(tool_results):
                print(f"🔧 [Inference] Tool result {i+1}: {result}")
        
        return ChatResponse(
            message=response_message, 
            session_id=session_id,
            tool_results=tool_results
        )
        
    except Exception as e:
        print(f"💥 [Inference] Error in chat endpoint: {str(e)}")
        import traceback
        print(f"📚 [Inference] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    if session_id in agent_sessions:
        del agent_sessions[session_id]
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")

@router.get("/tools")
async def get_available_tools():
    """Get information about available tools"""
    print(f"📋 [Inference] /tools endpoint called")
    tools = ToolService.get_available_tools()
    print(f"📊 [Inference] Returning {len(tools)} available tools")
    return tools

@router.post("/test-tool-calling")
async def test_tool_calling():
    """Test tool calling functionality"""
    print(f"🧪 [Inference] /test-tool-calling endpoint called")
    
    # Test the system prompt generation
    system_prompt = get_tools_system_prompt()
    print(f"📝 [Inference] Generated system prompt: {system_prompt[:200]}...")
    
    # Test tool extraction
    test_message = 'Here is a test: [TOOL_CALL:projects:{"action": "list", "namespace": "test"}]'
    tool_calls = extract_tool_calls(test_message)
    print(f"🔧 [Inference] Extracted tool calls from test: {tool_calls}")
    
    # Test tool execution
    if tool_calls:
        results = execute_tool_calls(tool_calls)
        print(f"📊 [Inference] Tool execution results: {results}")
    
    return {
        "system_prompt_length": len(system_prompt),
        "available_tools": ToolService.get_available_tools(),
        "test_tool_calls": tool_calls,
        "test_results": results if tool_calls else None
    }

@router.post("/test-agent-response")
async def test_agent_response(request: ChatRequest):
    """Test the agent's response to a specific message"""
    print(f"🧪 [Inference] /test-agent-response endpoint called with: {request.message}")
    
    try:
        # Create a new agent for testing
        agent = create_agent()
        
        # Process the user's input through the agent and get the response
        print(f"🤖 [Inference] Running agent with message: '{request.message[:100]}...'")
        input_schema = BasicChatInputSchema(chat_message=request.message)
        response = agent.run(input_schema)
        
        # Extract the message from the response
        if hasattr(response, 'chat_message'):
            response_message = response.chat_message
        else:
            # Fallback if the response structure is different
            response_message = str(response)
        
        print(f"📤 [Inference] Agent response: {response_message}")
        
        # Check for tool calls in the response
        tool_calls = extract_tool_calls(response_message)
        
        return {
            "user_message": request.message,
            "agent_response": response_message,
            "tool_calls_found": len(tool_calls),
            "tool_calls": tool_calls,
            "system_prompt": get_tools_system_prompt()[:500] + "..." if len(get_tools_system_prompt()) > 500 else get_tools_system_prompt()
        }
        
    except Exception as e:
        print(f"💥 [Inference] Error in test_agent_response: {str(e)}")
        import traceback
        print(f"📚 [Inference] Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error testing agent response: {str(e)}")