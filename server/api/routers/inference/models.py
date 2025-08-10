from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


# OpenAI-compatible chat message
class ChatMessage(BaseModel):
    role: str
    content: str


# OpenAI-compatible chat completion request
class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    metadata: dict[str, str] | None = None
    modalities: list[str] | None = None
    response_format: dict[str, str] | None = None
    stream: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] | None = None


class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# OpenAI-compatible chat completion response
class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str | None = None
    choices: list[ChatChoice]
    usage: Usage | None = None


class IntegrationType(Enum):
    NATIVE = "native_atomic_agents"
    MANUAL = "manual_execution"
    MANUAL_FAILED = "manual_execution_failed"


class ProjectAction(Enum):
    LIST = "list"
    CREATE = "create"


@dataclass
class ToolResult:
    success: bool
    action: str
    namespace: str
    message: str = ""
    result: Any = None
    integration_type: IntegrationType = IntegrationType.MANUAL


@dataclass
class ModelCapabilities:
    supports_tools: bool
    instructor_mode: Any  # instructor.Mode type