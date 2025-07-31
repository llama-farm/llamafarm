import uuid
from typing import Dict, Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema, BaseAgentOutputSchema
import instructor
from openai import OpenAI
from core.config import settings

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    message: str
    session_id: str

# Store agent instances to maintain conversation context
# In production, use Redis, database, or other persistent storage
agent_sessions: Dict[str, BaseAgent] = {}

def create_agent() -> BaseAgent:
    """Create a new agent instance"""
    # Initialize memory
    memory = AgentMemory()
    
    # Initialize memory with an initial message from the assistant
    initial_message = BaseAgentOutputSchema(chat_message="Hello! How can I assist you today?")
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

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Send a message to the chat agent"""
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
        
        return ChatResponse(message=response_message, session_id=session_id)
        
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