import json
import asyncio
import time
import uuid
from collections.abc import AsyncIterator

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

        # Use ChatProcessor directly with the full OpenAI-style request
        response_message, _tool_info = ChatProcessor.process_chat(
            request, session_id
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
                await asyncio.sleep(0)

                # Stream content in unicode- and word-boundary-safe chunks
                max_chars = 80
                def generate_chunks(text: str, limit: int) -> list[str]:
                    # Split on whitespace when possible; otherwise hard split
                    words = text.split()
                    if not words:
                        return []
                    chunks: list[str] = []
                    current: str = ""
                    for word in words:
                        to_add = word if current == "" else f" {word}"
                        if len(current) + len(to_add) <= limit:
                            current += to_add
                        else:
                            if current:
                                chunks.append(current)
                            # If word longer than limit, split that word safely
                            if len(word) > limit:
                                for i in range(0, len(word), limit):
                                    chunks.append(word[i:i+limit])
                                current = ""
                            else:
                                current = word
                    if current:
                        chunks.append(current)
                    return chunks

                for piece in generate_chunks(response_message, max_chars):
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
                    await asyncio.sleep(0)

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
                await asyncio.sleep(0)
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={
                    "X-Session-ID": session_id,
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
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