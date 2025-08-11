import uuid

from fastapi import APIRouter, Header, HTTPException

from .models import ChatRequest, ChatResponse
from .services import AgentSessionManager, ChatProcessor

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest, 
    session_id: str | None = Header(None, alias="X-Session-ID")
):
    """Send a message to the chat agent with advanced tool execution support"""
    try:
        # If no session ID provided, create a new one
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Use ChatProcessor which includes tool execution logic
        response_message, tool_info = ChatProcessor.process_chat(request, session_id)
        
        return ChatResponse(
            message=response_message, 
            session_id=session_id,
            tool_results=tool_info
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat: {str(e)}"
        ) from e

@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    if success := AgentSessionManager.delete_session(session_id):
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")