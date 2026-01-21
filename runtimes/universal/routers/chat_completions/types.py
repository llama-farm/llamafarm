from typing import Literal

from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam
from pydantic import BaseModel


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
