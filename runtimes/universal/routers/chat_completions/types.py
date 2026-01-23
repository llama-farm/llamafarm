from typing import Literal

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel, Field

# ============================================================================
# Audio Content Types (for STT transcription)
# ============================================================================


class InputAudio(BaseModel):
    """Audio data for input_audio content parts.

    Audio content is automatically transcribed via STT before LLM processing.
    """

    data: str = Field(..., description="Base64-encoded audio data")
    format: Literal["wav", "mp3", "pcm"] = Field(
        default="wav", description="Audio format (wav recommended for best compatibility)"
    )


class AudioContentPart(BaseModel):
    """Audio content part for messages with audio.

    Audio is automatically transcribed via STT and the text is passed to the LLM.
    """

    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudio


class TextContentPart(BaseModel):
    """Text content part for messages."""

    type: Literal["text"] = "text"
    text: str


# Union type for content parts in messages (text, audio, etc.)
ContentPart = AudioContentPart | TextContentPart | dict


# ============================================================================
# Tool Calling Types
# ============================================================================


class FunctionCall(BaseModel):
    """Function call details within a tool call."""

    name: str
    arguments: str  # JSON string of arguments


class ToolCall(BaseModel):
    """A tool call made by the assistant."""

    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str
    messages: list[ChatCompletionMessageParam]
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_tokens: int | None = None
    stream: bool | None = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    user: str | None = None
    # GGUF model parameters (llama.cpp specific)
    n_ctx: int | None = None  # Context window size (affects KV cache memory)
    n_batch: int | None = (
        None  # Batch size for prompt processing (affects compute buffer)
    )
    n_gpu_layers: int | None = None  # Number of layers to offload to GPU (-1 = all)
    n_threads: int | None = None  # CPU thread count (None = auto)
    flash_attn: bool | None = None  # Enable flash attention for faster inference
    use_mmap: bool | None = None  # Memory-map model file (True = efficient swapping)
    use_mlock: bool | None = (
        None  # Lock model in RAM (False = allow OS memory management)
    )
    cache_type_k: str | None = None  # KV cache key quantization (q4_0, q8_0, f16)
    cache_type_v: str | None = None  # KV cache value quantization (q4_0, q8_0, f16)
    extra_body: dict | None = None

    # Tool/function calling parameters
    tools: list[ChatCompletionToolParam] | None = None
    tool_choice: str | dict | None = (
        None  # "auto", "none", "required", or specific tool
    )

    # Thinking/reasoning model parameters (Ollama-compatible)
    # Controls whether thinking models show their reasoning process
    think: bool | None = None  # None = model default, True = enable, False = disable
    # Maximum tokens to spend on thinking before forcing answer generation
    # When reached, model is nudged to close </think> and provide answer
    thinking_budget: int | None = None

    # Context management parameters
    # Whether to automatically truncate messages if context is exceeded
    auto_truncate: bool | None = True
    # Truncation strategy: "sliding_window", "keep_system", "middle_out", "summarize"
    truncation_strategy: str | None = None


class ThinkingContent(BaseModel):
    """Thinking/reasoning content from a thinking model."""

    content: str  # The raw thinking content (without <think> tags)
    tokens: int | None = None  # Number of tokens used for thinking


class ContextUsageInfo(BaseModel):
    """Context window usage information."""

    total_context: int  # Total context window size in tokens
    prompt_tokens: int  # Tokens used by the prompt (input)
    available_for_completion: int  # Remaining tokens for output
    truncated: bool = False  # Whether truncation was applied
    truncated_messages: int = 0  # Number of messages removed
    strategy_used: str | None = None  # Truncation strategy used (if any)


class ChatCompletionResponse(BaseModel):
    """Extended chat completion response with thinking support."""

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict
    # Ollama-compatible: separate thinking from content
    thinking: ThinkingContent | None = None
    # Context usage information (extension field)
    x_context_usage: ContextUsageInfo | None = None


# ============================================================================
# Audio Content Extraction Utilities
# ============================================================================


def extract_audio_from_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[tuple[int, InputAudio]]:
    """Extract audio content parts from chat messages.

    Scans messages for input_audio content parts and returns them with
    their message index for later replacement if STT fallback is needed.

    Args:
        messages: List of chat completion messages

    Returns:
        List of (message_index, InputAudio) tuples for each audio part found
    """
    audio_parts: list[tuple[int, InputAudio]] = []

    for idx, message in enumerate(messages):
        # Skip if message is a string or has no content
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if content is None:
            continue

        # Handle list of content parts (multimodal message)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    audio_data = part.get("input_audio", {})
                    if isinstance(audio_data, dict) and "data" in audio_data:
                        audio_parts.append(
                            (
                                idx,
                                InputAudio(
                                    data=audio_data["data"],
                                    format=audio_data.get("format", "wav"),
                                ),
                            )
                        )

    return audio_parts


def has_audio_content(messages: list[ChatCompletionMessageParam]) -> bool:
    """Check if any messages contain audio content.

    Fast check without extracting the actual audio data.

    Args:
        messages: List of chat completion messages

    Returns:
        True if any message contains input_audio content
    """
    for message in messages:
        if not isinstance(message, dict):
            continue

        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "input_audio":
                    return True

    return False


def replace_audio_with_text(
    messages: list[ChatCompletionMessageParam],
    transcriptions: dict[int, str],
) -> list[dict]:
    """Replace audio content parts with transcribed text.

    Used when falling back to STT for models that don't support direct audio.

    Args:
        messages: Original messages with audio content
        transcriptions: Map of message_index -> transcribed text

    Returns:
        New messages list with audio replaced by text
    """
    result = []

    for idx, message in enumerate(messages):
        if not isinstance(message, dict):
            result.append(message)
            continue

        content = message.get("content")

        # If this message had audio and we have a transcription
        if idx in transcriptions:
            if isinstance(content, list):
                # Build new content parts, replacing audio with text
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "input_audio":
                        # Replace with transcribed text
                        new_parts.append({"type": "text", "text": transcriptions[idx]})
                    else:
                        new_parts.append(part)

                # Consolidate text parts
                consolidated = _consolidate_text_parts(new_parts)
                result.append({**message, "content": consolidated})
            else:
                # Simple string content - shouldn't happen but handle it
                result.append(message)
        else:
            result.append(dict(message) if isinstance(message, dict) else message)

    return result


def _consolidate_text_parts(parts: list[dict]) -> str | list[dict]:
    """Consolidate adjacent text parts into a single string if possible.

    If the result is all text parts, returns a simple string.
    Otherwise returns the list with adjacent text parts merged.
    """
    if not parts:
        return ""

    # Check if all parts are text
    all_text = all(
        isinstance(p, dict) and p.get("type") == "text" for p in parts
    )

    if all_text:
        # Return simple string
        return " ".join(p.get("text", "") for p in parts if isinstance(p, dict))

    # Otherwise, merge adjacent text parts
    result = []
    current_text = []

    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            current_text.append(part.get("text", ""))
        else:
            if current_text:
                result.append({"type": "text", "text": " ".join(current_text)})
                current_text = []
            result.append(part)

    if current_text:
        result.append({"type": "text", "text": " ".join(current_text)})

    return result
