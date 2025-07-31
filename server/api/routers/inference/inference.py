import uuid
import json
import re
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema
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
    tool_results: Optional[Dict] = None

# Store agent instances to maintain conversation context
# In production, use Redis, database, or other persistent storage
agent_sessions: Dict[str, BaseAgent] = {}

def create_agent() -> BaseAgent:
    """Create a new agent instance"""
    # Initialize memory
    memory = AgentMemory()
    
    # Initialize memory with an initial message from the assistant
    initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today? I can help you manage your LlamaFarm configurations, including updating config names.")
    memory.add_message("assistant", initial_message)
    
    # Create OpenAI-compatible client pointing to Ollama
    ollama_client = OpenAI(
        base_url=settings.ollama_host,
        api_key=settings.ollama_api_key,  # Ollama doesn't require a real API key, but instructor needs something
    )
    
    client = instructor.from_openai(ollama_client)

    # Agent setup with specified configuration
    agent = BaseAgent(
        config=BaseAgentConfig(
            client=client,
            model=settings.ollama_model,  # Using Ollama model name (make sure this model is installed)
            memory=memory,
        )
    )
    
    return agent

def extract_tool_calls(message: str) -> list:
    """
    Extract tool calls from the agent's response message.
    Looks for patterns like:
    - [TOOL_CALL:update_config_name:{"project_id": "test", "config_type": "prompts", "old_name": "old", "new_name": "new"}]
    """
    tool_calls = []
    pattern = r'\[TOOL_CALL:([^:]+):(\{[^}]+\})\]'
    
    matches = re.findall(pattern, message)
    for tool_name, params_str in matches:
        try:
            params = json.loads(params_str)
            tool_calls.append({
                "tool_name": tool_name,
                "parameters": params
            })
        except json.JSONDecodeError:
            # Skip malformed tool calls
            continue
    
    return tool_calls

def execute_tool_calls(tool_calls: list) -> list:
    """
    Execute tool calls and return results.
    """
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call["tool_name"]
        parameters = tool_call["parameters"]
        
        result = ToolService.execute_tool(tool_name, **parameters)
        results.append({
            "tool_name": tool_name,
            "parameters": parameters,
            "result": result
        })
    
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
    try:
        # If no session ID provided, create a new one
        if not session_id or session_id not in agent_sessions:
            if not session_id:
                session_id = str(uuid.uuid4())
            agent = create_agent()
            agent_sessions[session_id] = agent
        else:
            # Use existing agent to maintain conversation context
            agent = agent_sessions[session_id]

        # Process the user's input through the agent and get the response
        input_schema = BaseAgentInputSchema(chat_message=request.message)
        response = agent.run(input_schema)
        
        # Extract the message from the response
        if hasattr(response, 'chat_message'):
            response_message = response.chat_message
        else:
            # Fallback if the response structure is different
            response_message = str(response)
        
        # Check for tool calls in the response
        tool_calls = extract_tool_calls(response_message)
        tool_results = None
        
        if tool_calls:
            # Execute the tool calls
            tool_results = execute_tool_calls(tool_calls)
            
            # Remove tool call patterns from the message
            response_message = remove_tool_calls_from_message(response_message)
            
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
        
        return ChatResponse(
            message=response_message, 
            session_id=session_id,
            tool_results=tool_results
        )
        
    except Exception as e:
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
    return ToolService.get_available_tools()