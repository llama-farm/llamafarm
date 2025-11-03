import asyncio
from datetime import datetime
import os
import logging

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from models import LanguageModel
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice as ChoiceChunk,
    ChoiceDelta,
)
from .types import ChatCompletionRequest

logger = logging.getLogger(__name__)


class ChatCompletionsService:
    def __init__(self):
        # import here to avoid circular import
        from server import load_language

        self.load_language = load_language

    async def chat_completions(self, chat_request: ChatCompletionRequest):
        """
        Chat completions service.
        """

        try:
            model = await self.load_language(chat_request.model)

            # Convert messages to prompt
            # ChatCompletionMessageParam is already dict-compatible
            messages_dict = [dict(msg) for msg in chat_request.messages]
            if isinstance(model, LanguageModel):
                prompt = model.format_messages(messages_dict)

            # Handle streaming if requested
            if chat_request.stream:
                logger.info(
                    f"Streaming chat completions for model: {chat_request.model}"
                )

                # Return SSE stream
                async def generate_sse():
                    completion_id = f"chatcmpl-{os.urandom(16).hex()}"
                    created_time = int(datetime.now().timestamp())

                    # Send initial chunk
                    initial_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=chat_request.model,
                        choices=[
                            ChoiceChunk(
                                index=0,
                                delta=ChoiceDelta(role="assistant", content=""),
                                finish_reason=None,
                            )
                        ],
                    )
                    yield f"data: {initial_chunk.model_dump_json(exclude_none=True)}\n\n".encode()

                    # Stream tokens
                    async for token in model.generate_stream(
                        prompt=prompt,
                        max_tokens=chat_request.max_tokens,
                        temperature=chat_request.temperature if chat_request.temperature is not None else 0.7,
                        top_p=chat_request.top_p,
                        stop=chat_request.stop,
                    ):
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            object="chat.completion.chunk",
                            created=created_time,
                            model=chat_request.model,
                            choices=[
                                ChoiceChunk(
                                    index=0,
                                    delta=ChoiceDelta(role="assistant", content=token),
                                    finish_reason=None,
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                        # CRITICAL: This asyncio.sleep(0) forces the event loop to yield,
                        # ensuring the stream flushes immediately for token-by-token delivery.
                        # Without this, tokens would buffer and arrive in large chunks.
                        # See test_streaming_server.py for verification tests.
                        await asyncio.sleep(0)

                    # Send final chunk
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=chat_request.model,
                        choices=[
                            ChoiceChunk(
                                index=0,
                                delta=ChoiceDelta(),
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield f"data: {final_chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                    await asyncio.sleep(0)
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(
                    generate_sse(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "X-Accel-Buffering": "no",
                    },
                )

            # Non-streaming response
            response_text = await model.generate(
                prompt=prompt,
                max_tokens=chat_request.max_tokens,
                temperature=chat_request.temperature,
                top_p=chat_request.top_p,
                stop=chat_request.stop,
            )

            return {
                "id": f"chatcmpl-{os.urandom(16).hex()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": chat_request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": response_text},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # TODO: Implement token counting
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        except Exception as e:
            logger.error(f"Error in chat_completions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))
