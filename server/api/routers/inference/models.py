from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# OpenAI-compatible chat message
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# OpenAI-compatible chat completion request
class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    modalities: list[str] = Field(default_factory=list)
    response_format: dict[str, str] = Field(default_factory=dict)
    stream: bool | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    stop: list[str] = Field(default_factory=list)
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    logit_bias: dict[str, float] = Field(default_factory=dict)
    # RAG parameters (None means use config defaults)
    rag_enabled: bool | None = None
    database: str | None = None
    rag_top_k: int | None = None
    rag_score_threshold: float | None = None


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
