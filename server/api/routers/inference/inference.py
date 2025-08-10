from collections.abc import AsyncIterator
import json
import time
import uuid
from types import SimpleNamespace

from fastapi import APIRouter, Header, HTTPException, Response
from starlette.responses import StreamingResponse

from .models import ChatChoice, ChatMessage, ChatRequest, ChatResponse
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

        # Extract the latest user message from OpenAI-style messages
        latest_user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user" and msg.content:
                latest_user_message = msg.content
                break
        if latest_user_message is None and request.messages:
            latest_user_message = request.messages[-1].content

        # Build a small compatibility object expected by ChatProcessor
        compat_request = SimpleNamespace()
        compat_request.message = latest_user_message or ""
        # Pass optional routing context via metadata if provided
        metadata = request.metadata or {}
        compat_request.namespace = metadata.get("namespace")
        compat_request.project_id = metadata.get("project_id")

        # Use ChatProcessor which includes tool execution logic
        response_message, _tool_info = ChatProcessor.process_chat(
            compat_request, session_id
        )

        # Set session header for client continuity if response object is available
        if response is not None:
            response.headers["X-Session-ID"] = session_id

        # If client requested streaming, return Server-Sent Events stream
        if request.stream:
            created_ts = int(time.time())

            async def event_stream() -> AsyncIterator[bytes]:
                # Initial role delta (OpenAI-compatible)
                preface = {
                    "id": f"chat-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(preface)}\n\n".encode()

                # Stream content in chunks
                chunk_size = 40
                for i in range(0, len(response_message), chunk_size):
                    piece = response_message[i : i + chunk_size]
                    payload = {
                        "id": f"chat-{uuid.uuid4()}",
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield f"data: {json.dumps(payload)}\n\n".encode()

                # Final done signal
                done_payload = {
                    "id": f"chat-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(done_payload)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"X-Session-ID": session_id},
            )

        # Return OpenAI-compatible response
        return ChatResponse(
            id=f"chat-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_message),
                    finish_reason="stop",
                )
            ],
            usage=None,
        )

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