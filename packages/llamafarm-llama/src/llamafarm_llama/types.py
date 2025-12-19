"""
Type definitions for llamafarm-llama.

Provides TypedDict definitions compatible with llama-cpp-python's response formats.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Literal, Optional, TypedDict, Union


class ChatMessage(TypedDict, total=False):
    """A chat message."""

    role: str
    content: str
    name: Optional[str]
    tool_calls: Optional[List[Any]]
    tool_call_id: Optional[str]


class ChatCompletionChoice(TypedDict):
    """A single completion choice."""

    index: int
    message: ChatMessage
    finish_reason: Optional[str]


class ChatCompletionUsage(TypedDict):
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(TypedDict):
    """Full chat completion response."""

    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


class ChatCompletionChunkDelta(TypedDict, total=False):
    """Delta in a streaming chunk."""

    role: Optional[str]
    content: Optional[str]


class ChatCompletionChunkChoice(TypedDict):
    """A single choice in a streaming chunk."""

    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str]


class ChatCompletionChunk(TypedDict):
    """A streaming chunk."""

    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]


class EmbeddingData(TypedDict):
    """A single embedding result."""

    index: int
    embedding: List[float]
    object: Literal["embedding"]


class EmbeddingUsage(TypedDict):
    """Embedding token usage."""

    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(TypedDict):
    """Full embedding response."""

    object: Literal["list"]
    data: List[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# Type aliases
ChatCompletionStreamResponse = Iterator[ChatCompletionChunk]
CompletionResponse = Union[ChatCompletionResponse, ChatCompletionStreamResponse]
