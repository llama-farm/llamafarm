from pydantic import BaseModel
from openai.types.chat import ChatCompletionMessageParam


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
