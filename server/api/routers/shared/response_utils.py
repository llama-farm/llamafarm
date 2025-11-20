import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import Response
from starlette.responses import StreamingResponse

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger()


def set_session_header(response: Response | None, session_id: str | None) -> None:
    if response is not None and session_id:
        response.headers["X-Session-ID"] = session_id


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
    request: Any, response_message: str, session_id: str
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
    request,
    stream_source,
    session_id: str,
    *,
    default_message: str | None = None,
) -> StreamingResponse:
    """Convert ChatCompletionChunk stream to OpenAI SSE format.

    Args:
        request: Chat request (for model info)
        stream_source: AsyncGenerator yielding ChatCompletionChunk objects
        session_id: Session ID for header
        default_message: Fallback message if no chunks emitted
    """

    async def event_stream() -> AsyncIterator[bytes]:
        emitted = False

        try:
            async for chunk in stream_source:
                if not chunk:
                    continue
                # Chunk is already a ChatCompletionChunk - serialize it directly
                chunk_dict = chunk.model_dump(exclude_none=True)
                yield f"data: {json.dumps(chunk_dict)}\n\n".encode()
                await asyncio.sleep(0)
                emitted = True
        except asyncio.CancelledError:
            # Re-raise cancellation so it propagates correctly
            raise
        except Exception as e:
            # Log the error but ensure we still send [DONE] marker
            logger.error(
                "Error in stream source",
                exc_info=True,
                error=str(e),
            )
            # If we haven't emitted anything and have a default message, send it
            if not emitted and default_message:
                created_ts = int(time.time())
                payload = {
                    "id": f"chat-{uuid.uuid4()}",
                    "object": "chat.completion.chunk",
                    "created": created_ts,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": default_message},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield f"data: {json.dumps(payload)}\n\n".encode()
                await asyncio.sleep(0)
            # Always send [DONE] even on error to prevent unexpected EOF
            yield b"data: [DONE]\n\n"
            return

        if not emitted and default_message:
            # Fallback: create a simple chunk with default message
            created_ts = int(time.time())
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
