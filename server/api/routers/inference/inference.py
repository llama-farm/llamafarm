import uuid

from fastapi import APIRouter, Header, HTTPException, Response

from ..shared.response_utils import (
    build_chat_response,
    create_streaming_response_from_iterator,
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
    x_no_session: str | None = Header(None, alias="X-No-Session"),
):
    """Send a message to the chat agent with advanced tool execution support"""
    try:
        # Support stateless mode via X-No-Session
        if x_no_session is not None or session_id is None:
            stream_iter = ChatProcessor.process_chat(request, None)
        else:
            session_id = str(uuid.uuid4())
            stream_iter = ChatProcessor.process_chat(request, session_id)
            set_session_header(response, session_id)

        # If client requested streaming, return Server-Sent Events stream using agent-native streaming when possible
        if request.stream:
            sid = session_id or ""
            return create_streaming_response_from_iterator(
                request,
                stream_iter,
                sid,
            )

        # Non-streaming path
        return build_chat_response(request.model, stream_iter)

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing chat: {str(e)}"
        ) from e


@router.delete("/chat/sessions/{session_id}")
async def delete_chat_session(session_id: str):
    """Delete a chat session"""
    if AgentSessionManager.delete_session(session_id):
        return {"message": f"Session {session_id} deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")
