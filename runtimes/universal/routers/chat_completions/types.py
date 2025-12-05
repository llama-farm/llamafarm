from typing import Literal

from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel


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
    n_ctx: int | None = None  # Context window size for GGUF models
    extra_body: dict | None = None

    # Thinking/reasoning model parameters (Ollama-compatible)
    # Controls whether thinking models show their reasoning process
    think: bool | None = None  # None = model default, True = enable, False = disable
    # Maximum tokens to spend on thinking before forcing answer generation
    # When reached, model is nudged to close </think> and provide answer
    thinking_budget: int | None = None


class ThinkingContent(BaseModel):
    """Thinking/reasoning content from a thinking model."""

    content: str  # The raw thinking content (without <think> tags)
    tokens: int | None = None  # Number of tokens used for thinking


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
