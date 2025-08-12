
import uuid

from fastapi import APIRouter, Header, HTTPException, Response

from ..shared.response_utils import (
    build_chat_response,
    create_streaming_response,
    set_session_header,
)
from .models import ChatRequest, ChatResponse
from .services import AgentSessionManager, ChatProcessor

router = APIRouter(
    prefix="/inference",
    tags=["inference"],
)

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    response: Response,
    session_id: str | None = Header(None, alias="X-Session-ID"),
):
    """Send a message to the chat agent with advanced tool execution support"""
    try:
        # If no session ID provided, create a new one
        if not session_id:
            session_id = str(uuid.uuid4())

        # Use ChatProcessor directly with the full OpenAI-style request
        response_message, _tool_info = ChatProcessor.process_chat(
            request, session_id
        )

        # Set session header for client continuity
        set_session_header(response, session_id)

        # If client requested streaming, return Server-Sent Events stream
        if request.stream:
            return create_streaming_response(request, response_message, session_id)

        # Return OpenAI-compatible response
        return build_chat_response(request.model, response_message)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat: {str(e)}"
        ) from e

@router.delete("/chat/session/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    if AgentSessionManager.delete_session(session_id):
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")