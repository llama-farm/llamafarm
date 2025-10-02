import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator

from fastapi import Response
from starlette.responses import StreamingResponse

from ..inference.models import ChatChoice, ChatMessage, ChatRequest, ChatResponse


def set_session_header(response: Response | None, session_id: str | None) -> None:
    if response is not None and session_id:
        response.headers["X-Session-ID"] = session_id


def build_chat_response(model: str | None, response_message: str) -> ChatResponse:
    return ChatResponse(
        id=f"chat-{uuid.uuid4()}",
        object="chat.completion",
        created=int(time.time()),
        model=model,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=response_message),
                finish_reason="stop",
            )
        ],
    )


def _generate_chunks(text: str, limit: int) -> list[str]:
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
            if len(word) > limit:
                for i in range(0, len(word), limit):
                    chunks.append(word[i : i + limit])
                current = ""
            else:
                current = word
    if current:
        chunks.append(current)
    return chunks


def create_streaming_response(
    request: ChatRequest, response_message: str, session_id: str
) -> StreamingResponse:
    created_ts = int(time.time())

    async def event_stream() -> AsyncIterator[bytes]:
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

        for piece in _generate_chunks(response_message, 80):
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

        done_payload = {
            "id": f"chat-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
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


def create_streaming_response_from_iterator(
    request: ChatRequest,
    stream_source: AsyncGenerator[str],
    session_id: str,
    *,
    default_message: str | None = None,
) -> StreamingResponse:
    created_ts = int(time.time())

    async def event_stream() -> AsyncIterator[bytes]:
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

        emitted = False

        async for piece in stream_source:
            if not piece:
                continue
            payload = {
                "id": f"chat-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": str(piece)},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
            # Sleep(0) yields control back to event loop, preventing blocking
            # This allows other coroutines to run between stream chunks
            await asyncio.sleep(0)
            emitted = True

        if not emitted and default_message:
            payload = {
                "id": f"chat-{uuid.uuid4()}",
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": default_message},
                        "finish_reason": None,
                    }
                ],
            }
            yield f"data: {json.dumps(payload)}\n\n".encode()
            await asyncio.sleep(0)

        done_payload = {
            "id": f"chat-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": created_ts,
            "model": request.model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
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
