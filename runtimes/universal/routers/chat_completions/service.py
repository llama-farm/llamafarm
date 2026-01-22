import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChoiceChunk,
)

from models import GGUFLanguageModel
from utils.context_manager import TruncationStrategy
from utils.context_summarizer import ContextSummarizer
from utils.history_compressor import HistoryCompressor
from utils.thinking import inject_thinking_control, parse_thinking_response
from utils.tool_calling import (
    detect_probable_tool_call,
    detect_tool_call_in_content,
    extract_arguments_progress,
    extract_tool_name_from_partial,
    is_tool_call_complete,
    parse_tool_choice,
    strip_tool_call_from_content,
)

from .types import ChatCompletionRequest, ContextUsageInfo, ThinkingContent


class ToolCallStreamState(Enum):
    """State machine states for incremental tool call streaming."""

    NORMAL = "normal"  # Streaming regular content
    BUFFERING_START = "buffering_start"  # Detected <tool_call>, waiting for name
    STREAMING_ARGS = "streaming_args"  # Name emitted, streaming arguments


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

            # Get GGUF-specific parameters from request
            n_ctx = chat_request.n_ctx
            n_batch = chat_request.n_batch
            n_gpu_layers = chat_request.n_gpu_layers
            n_threads = chat_request.n_threads
            flash_attn = chat_request.flash_attn
            use_mmap = chat_request.use_mmap
            use_mlock = chat_request.use_mlock
            cache_type_k = chat_request.cache_type_k
            cache_type_v = chat_request.cache_type_v

            # Also check extra_body for these parameters (OpenAI SDK sends custom params there)
            if chat_request.extra_body:
                if n_ctx is None and "n_ctx" in chat_request.extra_body:
                    n_ctx = chat_request.extra_body.get("n_ctx")
                if n_batch is None and "n_batch" in chat_request.extra_body:
                    n_batch = chat_request.extra_body.get("n_batch")
                if n_gpu_layers is None and "n_gpu_layers" in chat_request.extra_body:
                    n_gpu_layers = chat_request.extra_body.get("n_gpu_layers")
                if n_threads is None and "n_threads" in chat_request.extra_body:
                    n_threads = chat_request.extra_body.get("n_threads")
                if flash_attn is None and "flash_attn" in chat_request.extra_body:
                    flash_attn = chat_request.extra_body.get("flash_attn")
                if use_mmap is None and "use_mmap" in chat_request.extra_body:
                    use_mmap = chat_request.extra_body.get("use_mmap")
                if use_mlock is None and "use_mlock" in chat_request.extra_body:
                    use_mlock = chat_request.extra_body.get("use_mlock")
                if cache_type_k is None and "cache_type_k" in chat_request.extra_body:
                    cache_type_k = chat_request.extra_body.get("cache_type_k")
                if cache_type_v is None and "cache_type_v" in chat_request.extra_body:
                    cache_type_v = chat_request.extra_body.get("cache_type_v")

            # Parse model name to extract quantization if present
            model_id, gguf_quantization = parse_model_with_quantization(
                chat_request.model
            )

            model = await self.load_language(
                model_id,
                n_ctx=n_ctx,
                n_batch=n_batch,
                n_gpu_layers=n_gpu_layers,
                n_threads=n_threads,
                flash_attn=flash_attn,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                cache_type_k=cache_type_k,
                cache_type_v=cache_type_v,
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

            # Convert tools to dict format if provided (for streaming)
            tools_dict = None
            if chat_request.tools:
                tools_dict = [dict(tool) for tool in chat_request.tools]

            # Context management for GGUF models
            context_usage_info = None
            if is_gguf and model.context_manager:
                # Apply history compression to reduce token usage
                compressor = HistoryCompressor(model.token_counter)
                messages_dict = compressor.compress(messages_dict)

                # Validate context and truncate if needed
                usage = model.context_manager.validate_messages(messages_dict)

                if model.context_manager.needs_truncation(messages_dict):
                    auto_truncate = chat_request.auto_truncate
                    if auto_truncate is None:
                        auto_truncate = True  # Default to auto-truncate

                    if not auto_truncate:
                        # Return error instead of truncating
                        raise HTTPException(
                            status_code=400,
                            detail={
                                "error": "context_length_exceeded",
                                "message": (
                                    f"Prompt ({usage.prompt_tokens} tokens) exceeds "
                                    f"context limit ({usage.total_context} tokens). "
                                    "Set auto_truncate=true to automatically truncate."
                                ),
                                "context_usage": {
                                    "total_context": usage.total_context,
                                    "prompt_tokens": usage.prompt_tokens,
                                    "available_for_completion": usage.available_for_completion,
                                },
                            },
                        )

                    # Determine truncation strategy
                    strategy = None
                    if chat_request.truncation_strategy:
                        try:
                            strategy = TruncationStrategy(
                                chat_request.truncation_strategy
                            )
                        except ValueError:
                            logger.warning(
                                f"Unknown truncation strategy: {chat_request.truncation_strategy}, "
                                "using default (summarize)"
                            )
                            strategy = TruncationStrategy.SUMMARIZE
                    else:
                        strategy = TruncationStrategy.SUMMARIZE  # Default

                    # Handle summarization strategy (async, needs special handling)
                    if strategy == TruncationStrategy.SUMMARIZE:
                        try:
                            # Pass the server's load_language for proper caching
                            summarizer = ContextSummarizer(
                                load_language=self.load_language
                            )
                            messages_dict = await summarizer.summarize_messages(
                                messages_dict
                            )
                            # Re-validate after summarization
                            usage = model.context_manager.validate_messages(
                                messages_dict
                            )

                            # Check if we STILL need truncation after summarization
                            # (e.g., if recent messages are still too large)
                            if model.context_manager.needs_truncation(messages_dict):
                                logger.warning(
                                    f"Still over budget after summarization "
                                    f"({usage.prompt_tokens} tokens), applying fallback truncation"
                                )
                                messages_dict, usage = (
                                    model.context_manager.truncate_if_needed(
                                        messages_dict,
                                        TruncationStrategy.KEEP_SYSTEM_SLIDING,
                                    )
                                )
                                usage = type(usage)(
                                    total_context=usage.total_context,
                                    prompt_tokens=usage.prompt_tokens,
                                    available_for_completion=usage.available_for_completion,
                                    truncated=True,
                                    truncated_messages=usage.truncated_messages,
                                    strategy_used="summarize+keep_system",
                                )
                            else:
                                usage = type(usage)(
                                    total_context=usage.total_context,
                                    prompt_tokens=usage.prompt_tokens,
                                    available_for_completion=usage.available_for_completion,
                                    truncated=True,
                                    truncated_messages=0,  # Summarized, not removed
                                    strategy_used="summarize",
                                )
                            logger.info(
                                f"Context summarized: {usage.prompt_tokens} tokens after summarization"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Summarization failed: {e}, falling back to keep_system"
                            )
                            messages_dict, usage = (
                                model.context_manager.truncate_if_needed(
                                    messages_dict,
                                    TruncationStrategy.KEEP_SYSTEM_SLIDING,
                                )
                            )
                    else:
                        # Use regular truncation strategy
                        messages_dict, usage = model.context_manager.truncate_if_needed(
                            messages_dict, strategy
                        )
                        logger.info(
                            f"Context truncated: {usage.truncated_messages} messages removed, "
                            f"strategy={usage.strategy_used}"
                        )

                # Store context usage for response
                context_usage_info = ContextUsageInfo(
                    total_context=usage.total_context,
                    prompt_tokens=usage.prompt_tokens,
                    available_for_completion=usage.available_for_completion,
                    truncated=usage.truncated,
                    truncated_messages=usage.truncated_messages,
                    strategy_used=usage.strategy_used,
                )

                # Final safety check: ensure we're actually under budget
                if model.context_manager.needs_truncation(messages_dict):
                    final_usage = model.context_manager.validate_messages(messages_dict)
                    logger.error(
                        f"CRITICAL: Still over context budget after all truncation: "
                        f"{final_usage.prompt_tokens} tokens > "
                        f"{model.context_manager.budget.max_prompt_tokens} max"
                    )
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "context_truncation_failed",
                            "message": (
                                f"Failed to reduce context to fit within budget. "
                                f"Current: {final_usage.prompt_tokens} tokens, "
                                f"Max: {model.context_manager.budget.max_prompt_tokens} tokens. "
                                "Try sending fewer or shorter messages."
                            ),
                            "context_usage": {
                                "total_context": final_usage.total_context,
                                "prompt_tokens": final_usage.prompt_tokens,
                                "available_for_completion": final_usage.available_for_completion,
                            },
                        },
                    )

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
                        tools=tools_dict,
                        tool_choice=chat_request.tool_choice,
                    )

                    # State machine for incremental tool call streaming
                    accumulated_content = ""
                    tool_state = ToolCallStreamState.NORMAL
                    buffered_tokens = []
                    tool_call_id = None
                    tool_call_index = 0
                    args_emitted_length = 0
                    any_tool_calls_emitted = False  # Track if we emitted any tool calls

                    # Parse tool_choice to determine if we should detect tool calls
                    # When tool_choice="none", we skip tool detection entirely
                    tool_choice_mode, _ = parse_tool_choice(chat_request.tool_choice)
                    should_detect_tools = tools_dict and tool_choice_mode != "none"

                    async for token in token_stream:
                        accumulated_content += token

                        # STATE: NORMAL - streaming regular content
                        if tool_state == ToolCallStreamState.NORMAL:
                            # Check if we're entering a tool call
                            if should_detect_tools and detect_probable_tool_call(
                                accumulated_content
                            ):
                                tool_state = ToolCallStreamState.BUFFERING_START
                                buffered_tokens.append(token)
                                continue

                            # Normal content streaming
                            chunk = ChatCompletionChunk(
                                id=completion_id,
                                object="chat.completion.chunk",
                                created=created_time,
                                model=chat_request.model,
                                choices=[
                                    ChoiceChunk(
                                        index=0,
                                        delta=ChoiceDelta(
                                            role="assistant", content=token
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                            # CRITICAL: This asyncio.sleep(0) forces the event loop
                            # to yield, ensuring token-by-token delivery.
                            await asyncio.sleep(0)

                        # STATE: BUFFERING_START - waiting for tool name
                        elif tool_state == ToolCallStreamState.BUFFERING_START:
                            buffered_tokens.append(token)

                            # Try to extract tool name
                            tool_name = extract_tool_name_from_partial(
                                accumulated_content
                            )
                            if tool_name:
                                # Emit initial tool call chunk with name
                                tool_call_id = f"call_{uuid.uuid4()}"
                                initial_tool_chunk = ChatCompletionChunk(
                                    id=completion_id,
                                    object="chat.completion.chunk",
                                    created=created_time,
                                    model=chat_request.model,
                                    choices=[
                                        ChoiceChunk(
                                            index=0,
                                            delta=ChoiceDelta(
                                                tool_calls=[
                                                    ChoiceDeltaToolCall(
                                                        index=tool_call_index,
                                                        id=tool_call_id,
                                                        type="function",
                                                        function=ChoiceDeltaToolCallFunction(
                                                            name=tool_name,
                                                            arguments="",
                                                        ),
                                                    )
                                                ]
                                            ),
                                            finish_reason=None,
                                        )
                                    ],
                                )
                                yield f"data: {initial_tool_chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                                await asyncio.sleep(0)

                                tool_state = ToolCallStreamState.STREAMING_ARGS
                                args_emitted_length = 0
                                logger.info(
                                    f"Tool call started: {tool_name} (id={tool_call_id})"
                                )

                        # STATE: STREAMING_ARGS - incrementally streaming arguments
                        elif tool_state == ToolCallStreamState.STREAMING_ARGS:
                            # Check if tool call is complete
                            if is_tool_call_complete(accumulated_content):
                                # Parse the complete tool call to get final arguments
                                # We only want the FIRST complete tool call in accumulated_content
                                tool_calls = detect_tool_call_in_content(
                                    accumulated_content
                                )
                                if tool_calls:
                                    _, final_args = tool_calls[0]

                                    # Emit remaining arguments (from where we left off)
                                    if len(final_args) > args_emitted_length:
                                        remaining_args = final_args[
                                            args_emitted_length:
                                        ]
                                        args_chunk = ChatCompletionChunk(
                                            id=completion_id,
                                            object="chat.completion.chunk",
                                            created=created_time,
                                            model=chat_request.model,
                                            choices=[
                                                ChoiceChunk(
                                                    index=0,
                                                    delta=ChoiceDelta(
                                                        tool_calls=[
                                                            ChoiceDeltaToolCall(
                                                                index=tool_call_index,
                                                                function=ChoiceDeltaToolCallFunction(
                                                                    arguments=remaining_args,
                                                                ),
                                                            )
                                                        ]
                                                    ),
                                                    finish_reason=None,
                                                )
                                            ],
                                        )
                                        yield f"data: {args_chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                                        await asyncio.sleep(0)

                                # Log the completed tool call
                                if tool_calls:
                                    tool_name_completed, tool_args = tool_calls[0]
                                    logger.info(
                                        f"Tool call completed: {tool_name_completed} "
                                        f"(id={tool_call_id}, args={tool_args[:100]}{'...' if len(tool_args) > 100 else ''})"
                                    )

                                # Mark that we've emitted at least one tool call
                                any_tool_calls_emitted = True

                                # Reset state machine for potential next tool call
                                # Strip the completed tool call from accumulated_content
                                accumulated_content = strip_tool_call_from_content(
                                    accumulated_content
                                )
                                tool_state = ToolCallStreamState.NORMAL
                                buffered_tokens = []
                                tool_call_id = None
                                tool_call_index += 1
                                args_emitted_length = 0

                                # Check if there's already another tool call starting
                                # in the remaining content
                                if should_detect_tools and detect_probable_tool_call(
                                    accumulated_content
                                ):
                                    tool_state = ToolCallStreamState.BUFFERING_START

                                # Continue processing - don't return yet
                                continue

                            # Try to extract arguments progress
                            args_progress = extract_arguments_progress(
                                accumulated_content
                            )
                            if args_progress:
                                _, current_args = args_progress
                                # Emit new argument characters
                                if len(current_args) > args_emitted_length:
                                    new_args = current_args[args_emitted_length:]
                                    args_chunk = ChatCompletionChunk(
                                        id=completion_id,
                                        object="chat.completion.chunk",
                                        created=created_time,
                                        model=chat_request.model,
                                        choices=[
                                            ChoiceChunk(
                                                index=0,
                                                delta=ChoiceDelta(
                                                    tool_calls=[
                                                        ChoiceDeltaToolCall(
                                                            index=tool_call_index,
                                                            function=ChoiceDeltaToolCallFunction(
                                                                arguments=new_args,
                                                            ),
                                                        )
                                                    ]
                                                ),
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                                    yield f"data: {args_chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                                    await asyncio.sleep(0)
                                    args_emitted_length = len(current_args)

                    # Handle incomplete tool calls at stream end
                    if (
                        tool_state != ToolCallStreamState.NORMAL
                        and buffered_tokens
                        and not is_tool_call_complete(accumulated_content)
                    ):
                        # Emit buffered tokens as regular content
                        for buffered_token in buffered_tokens:
                            chunk = ChatCompletionChunk(
                                id=completion_id,
                                object="chat.completion.chunk",
                                created=created_time,
                                model=chat_request.model,
                                choices=[
                                    ChoiceChunk(
                                        index=0,
                                        delta=ChoiceDelta(content=buffered_token),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                            await asyncio.sleep(0)

                    # Debug log the accumulated streaming response
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            f"Streaming response complete ({len(accumulated_content)} chars):\n"
                            f"{accumulated_content}"
                        )

                    # Send final chunk with appropriate finish_reason
                    # If we emitted any tool calls, use "tool_calls", otherwise "stop"
                    finish_reason = "tool_calls" if any_tool_calls_emitted else "stop"
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        object="chat.completion.chunk",
                        created=created_time,
                        model=chat_request.model,
                        choices=[
                            ChoiceChunk(
                                index=0,
                                delta=ChoiceDelta(),
                                finish_reason=finish_reason,
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
                tools=tools_dict,
                tool_choice=chat_request.tool_choice,
            )

            # Debug log the raw response from the model
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Model raw response ({len(response_text)} chars):\n{response_text}"
                )

            # Parse thinking content from response (like Ollama does)
            # This separates <think>...</think> into a separate field
            parsed = parse_thinking_response(response_text)

            # Check for tool calls in response (only if tools were provided and tool_choice != "none")
            # This is consistent with streaming path which only checks when tools are enabled
            tool_calls = None
            tool_choice_mode, _ = parse_tool_choice(chat_request.tool_choice)
            if tools_dict and tool_choice_mode != "none":
                tool_calls = detect_tool_call_in_content(parsed.content)

            if tool_calls:
                # Log detected tool calls
                for name, args in tool_calls:
                    logger.info(
                        f"Tool call detected: {name} "
                        f"(args={args[:100]}{'...' if len(args) > 100 else ''})"
                    )

                # Build response with tool calls
                prompt_tokens = (
                    context_usage_info.prompt_tokens if context_usage_info else 0
                )
                response = {
                    "id": f"chatcmpl-{os.urandom(16).hex()}",
                    "object": "chat.completion",
                    "created": int(datetime.now().timestamp()),
                    "model": chat_request.model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": f"call_{uuid.uuid4()}",
                                        "type": "function",
                                        "function": {
                                            "name": name,
                                            "arguments": args,
                                        },
                                    }
                                    for name, args in tool_calls
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 0,  # TODO: count completion tokens
                        "total_tokens": prompt_tokens,
                    },
                }
                # Add context usage info if available
                if context_usage_info:
                    response["x_context_usage"] = context_usage_info.model_dump()

                # Debug log the response with tool calls
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(
                        f"Sending response with tool calls:\n"
                        f"{json.dumps(response, indent=2, default=str)}"
                    )

                return response

            # Build response with optional thinking field (Ollama-compatible)
            prompt_tokens = (
                context_usage_info.prompt_tokens if context_usage_info else 0
            )
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
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": 0,  # TODO: count completion tokens
                    "total_tokens": prompt_tokens,
                },
            }

            # Add thinking field if present (Ollama-compatible)
            if parsed.thinking:
                response["thinking"] = ThinkingContent(
                    content=parsed.thinking,
                    tokens=None,  # TODO: count thinking tokens
                ).model_dump()

            # Add context usage info if available
            if context_usage_info:
                response["x_context_usage"] = context_usage_info.model_dump()

            # Debug log the response
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Sending response:\n{json.dumps(response, indent=2, default=str)}"
                )

            return response

        except Exception as e:
            logger.error(f"Error in chat_completions: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e)) from e
