import asyncio
import logging
import os
from datetime import datetime

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChoiceChunk,
)

from models import GGUFLanguageModel
from utils.thinking import inject_thinking_control, parse_thinking_response

from .types import ChatCompletionRequest, ThinkingContent

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
            # Import parsing utility
            from utils.model_format import parse_model_with_quantization

            # Get context window size from request
            n_ctx = chat_request.n_ctx

            # Parse model name to extract quantization if present
            model_id, gguf_quantization = parse_model_with_quantization(
                chat_request.model
            )

            model = await self.load_language(
                model_id,
                n_ctx=n_ctx,
                preferred_quantization=gguf_quantization,
            )

            # Convert messages to dict format
            # ChatCompletionMessageParam is already dict-compatible
            messages_dict = [dict(msg) for msg in chat_request.messages]

            # Extract thinking params from extra_body if not set at top level
            # (OpenAI SDK sends custom params via extra_body)
            think_param = chat_request.think
            thinking_budget_param = chat_request.thinking_budget
            if chat_request.extra_body:
                if think_param is None and "think" in chat_request.extra_body:
                    think_param = chat_request.extra_body.get("think")
                if (
                    thinking_budget_param is None
                    and "thinking_budget" in chat_request.extra_body
                ):
                    thinking_budget_param = chat_request.extra_body.get(
                        "thinking_budget"
                    )

            # Check if this is a GGUF model - use native chat completion for proper template
            # GGUF models have create_chat_completion() which uses the embedded chat template
            # This is essential for models like Qwen that use special tokens (<|im_start|>, etc.)
            # and thinking tags (<think>)
            is_gguf = isinstance(model, GGUFLanguageModel)

            # Inject thinking control (Qwen soft switch: /think or /no_think)
            # Default is OFF - inject /no_think unless explicitly enabled with think=true
            if is_gguf:
                # think=True -> enable, think=False or None -> disable
                enable_thinking = think_param is True
                messages_dict = inject_thinking_control(
                    messages_dict, enable_thinking=enable_thinking
                )
                logger.info(
                    f"Thinking mode {'enabled' if enable_thinking else 'disabled'} via soft switch"
                )

            # Calculate total token budget for generation
            # - max_tokens: for the final answer (default: 512)
            # - thinking_budget: for the thinking process (default: 1024 if thinking enabled)
            # Total = thinking_budget + max_tokens (so answer isn't cut short by thinking)
            answer_tokens = chat_request.max_tokens or 512

            # Determine if thinking is enabled (default: OFF for predictable behavior)
            # User must explicitly set think=true to enable thinking mode
            thinking_enabled = think_param is True

            if thinking_enabled and is_gguf:
                # Use provided thinking_budget or default to 1024
                thinking_tokens = thinking_budget_param or 1024
                total_max_tokens = thinking_tokens + answer_tokens
                logger.info(
                    f"Token allocation: {thinking_tokens} for thinking + {answer_tokens} for answer = {total_max_tokens} total"
                )
            else:
                # No thinking, just use answer tokens
                total_max_tokens = answer_tokens
                thinking_tokens = 0

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

                    # Stream tokens using unified generate_stream API
                    token_stream = model.generate_stream(
                        messages=messages_dict,
                        max_tokens=total_max_tokens,
                        temperature=chat_request.temperature
                        if chat_request.temperature is not None
                        else 0.7,
                        top_p=chat_request.top_p,
                        stop=chat_request.stop,
                        thinking_budget=thinking_tokens if is_gguf else None,
                    )

                    async for token in token_stream:
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

            # Non-streaming response using unified generate API
            response_text = await model.generate(
                messages=messages_dict,
                max_tokens=total_max_tokens,
                temperature=chat_request.temperature
                if chat_request.temperature is not None
                else 0.7,
                top_p=chat_request.top_p,
                stop=chat_request.stop,
                thinking_budget=thinking_tokens if is_gguf else None,
            )

            # Parse thinking content from response (like Ollama does)
            # This separates <think>...</think> into a separate field
            parsed = parse_thinking_response(response_text)

            # Build response with optional thinking field (Ollama-compatible)
            response = {
                "id": f"chatcmpl-{os.urandom(16).hex()}",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": chat_request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": parsed.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # TODO: Implement token counting
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

            # Add thinking field if present (Ollama-compatible)
            if parsed.thinking:
                response["thinking"] = ThinkingContent(
                    content=parsed.thinking,
                    tokens=None,  # TODO: count thinking tokens
                ).model_dump()

            return response

        except Exception as e:
            logger.error(f"Error in chat_completions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e
