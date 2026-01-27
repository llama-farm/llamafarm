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


# Multimodal types for audio/vision input


class AudioData(TypedDict):
    """Audio data for multimodal input."""

    data: str  # base64-encoded audio or file path
    format: Literal["wav", "pcm", "mp3"]


class AudioContentPart(TypedDict):
    """Audio content part for multimodal messages."""

    type: Literal["input_audio"]
    input_audio: AudioData


class TextContentPart(TypedDict):
    """Text content part for multimodal messages."""

    type: Literal["text"]
    text: str


class ImageUrlData(TypedDict, total=False):
    """Image URL data for multimodal input."""

    url: str  # URL or base64 data URI
    detail: Optional[Literal["auto", "low", "high"]]


class ImageContentPart(TypedDict):
    """Image content part for multimodal messages."""

    type: Literal["image_url"]
    image_url: ImageUrlData


# Content can be text, audio, or image
ContentPart = Union[TextContentPart, AudioContentPart, ImageContentPart]


class MultimodalMessage(TypedDict, total=False):
    """A chat message that can contain multimodal content (text, audio, images)."""

    role: str
    content: Union[str, List[ContentPart]]
    name: Optional[str]


# Type aliases
ChatCompletionStreamResponse = Iterator[ChatCompletionChunk]
CompletionResponse = Union[ChatCompletionResponse, ChatCompletionStreamResponse]
