"""FastAPI server for Llama Brain."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn

from llama_brain.config.hardcoded import get_settings
from llama_brain.chat import ChatManager, ChatMessage, MessageRole, ChatResponse
from llama_brain.agents import ModelAgent, RAGAgent, PromptAgent


app = FastAPI(
    title="Llama Brain API",
    description="AI-powered configuration assistant for LlamaFarm",
    version="0.1.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
settings = get_settings()
chat_manager = ChatManager()
model_agent = ModelAgent()
rag_agent = RAGAgent()
prompt_agent = PromptAgent()


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ChatResponseModel(BaseModel):
    message: str
    session_id: str
    actions: List[Dict[str, Any]] = []
    context_used: List[Dict[str, Any]] = []
    confidence: float = 0.8
    suggestions: List[str] = []
    executed_actions: List[Dict[str, Any]] = []


class ConfigRequest(BaseModel):
    agent_type: str  # "model", "rag", "prompt"
    action: str      # "create", "edit", "validate"
    requirements: Dict[str, Any] = {}
    config: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None


class ConfigResponse(BaseModel):
    success: bool
    config: Optional[Dict[str, Any]] = None
    validation: Optional[Dict[str, Any]] = None
    message: str
    file_path: Optional[str] = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Llama Brain",
        "version": "0.1.0",
        "ollama_host": settings.ollama_host,
        "chat_model": settings.chat_model
    }


@app.post("/chat", response_model=ChatResponseModel)
async def chat(request: ChatRequest):
    """Handle chat messages."""
    try:
        # Create session if not provided
        session_id = request.session_id
        if not session_id:
            session_id = chat_manager.create_session()
        
        # Process the message
        response = await chat_manager.process_message(session_id, request.message)
        
        return ChatResponseModel(
            message=response.message,
            session_id=session_id,
            actions=[action.dict() for action in response.actions],
            context_used=response.context_used,
            confidence=response.confidence,
            suggestions=response.suggestions,
            executed_actions=response.executed_actions
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/{session_id}/history")
async def get_chat_history(session_id: str):
    """Get chat history for a session."""
    session = chat_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "messages": [
            {
                "role": msg.role if isinstance(msg.role, str) else msg.role.value, 
                "content": msg.content,
                "timestamp": msg.timestamp.isoformat()
            }
            for msg in session.messages
            if (msg.role != MessageRole.SYSTEM and msg.role != "system")  # Hide system messages
        ]
    }


@app.post("/config", response_model=ConfigResponse)
async def handle_config(request: ConfigRequest):
    """Handle configuration operations."""
    try:
        agent = None
        
        # Select the appropriate agent
        if request.agent_type == "model":
            agent = model_agent
        elif request.agent_type == "rag":
            agent = rag_agent
        elif request.agent_type == "prompt":
            agent = prompt_agent
        else:
            raise ValueError(f"Unknown agent type: {request.agent_type}")
        
        # Handle the requested action
        if request.action == "create":
            config = await agent.create_config(request.requirements)
            
            # Save the configuration
            filename = f"{request.agent_type}_{request.requirements.get('use_case', 'custom')}"
            file_path = await agent.save_config(config, filename)
            
            return ConfigResponse(
                success=True,
                config=config,
                message=f"Successfully created {request.agent_type} configuration",
                file_path=str(file_path)
            )
            
        elif request.action == "edit":
            if not request.config or not request.changes:
                raise ValueError("Edit action requires both config and changes")
            
            config = await agent.edit_config(request.config, request.changes)
            
            # Save the edited configuration
            filename = f"{request.agent_type}_edited"
            file_path = await agent.save_config(config, filename)
            
            return ConfigResponse(
                success=True,
                config=config,
                message=f"Successfully edited {request.agent_type} configuration",
                file_path=str(file_path)
            )
            
        elif request.action == "validate":
            if not request.config:
                raise ValueError("Validate action requires config")
            
            validation = await agent.validate_config(request.config)
            
            return ConfigResponse(
                success=validation["valid"],
                config=request.config,
                validation=validation,
                message=f"Configuration validation {'passed' if validation['valid'] else 'failed'}"
            )
            
        else:
            raise ValueError(f"Unknown action: {request.action}")
            
    except Exception as e:
        return ConfigResponse(
            success=False,
            message=f"Error: {str(e)}"
        )


@app.get("/config/examples/{agent_type}")
async def get_examples(agent_type: str):
    """Get available example configurations for an agent."""
    try:
        agent = None
        if agent_type == "model":
            agent = model_agent
        elif agent_type == "rag":
            agent = rag_agent
        elif agent_type == "prompt":
            agent = prompt_agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        examples = agent.get_available_examples()
        
        return {
            "agent_type": agent_type,
            "examples": examples
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config/examples/{agent_type}/{example_name}")
async def load_example(agent_type: str, example_name: str):
    """Load a specific example configuration."""
    try:
        agent = None
        if agent_type == "model":
            agent = model_agent
        elif agent_type == "rag":
            agent = rag_agent
        elif agent_type == "prompt":
            agent = prompt_agent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        config = await agent.load_example_config(example_name)
        if not config:
            raise HTTPException(status_code=404, detail="Example not found")
        
        return {
            "agent_type": agent_type,
            "example_name": example_name,
            "config": config
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status")
async def get_status():
    """Get system status."""
    try:
        # Check Ollama connection
        import requests
        ollama_response = requests.get(f"{settings.ollama_host}/api/tags", timeout=5)
        ollama_available = ollama_response.status_code == 200
        
        # Check available models
        models = []
        if ollama_available:
            models_data = ollama_response.json()
            models = [model["name"] for model in models_data.get("models", [])]
        
        return {
            "ollama_available": ollama_available,
            "ollama_host": settings.ollama_host,
            "chat_model": settings.chat_model,
            "embedding_model": settings.embedding_model,
            "available_models": models,
            "active_sessions": len(chat_manager.sessions)
        }
        
    except Exception as e:
        return {
            "ollama_available": False,
            "error": str(e)
        }


def main():
    """Run the server."""
    uvicorn.run(
        "llama_brain.server.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )


if __name__ == "__main__":
    main()