"""LlamaFarm SDK types — request/response models matching server APIs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Chat ─────────────────────────────────────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in a chat conversation."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        description="Message role: system, user, assistant, or tool"
    )
    content: str | None = Field(default=None, description="Text content of the message")
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Tool calls made by the assistant"
    )
    tool_call_id: str | None = Field(default=None, description="ID of the tool call being responded to")
    name: str | None = Field(default=None, description="Optional name for the participant")


class ChatChoice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatResponse(BaseModel):
    """Response from a chat completion request (OpenAI-compatible)."""

    id: str = ""
    object: str = "chat.completion"
    created: int = 0
    model: str = ""
    choices: list[ChatChoice] = Field(default_factory=list)
    usage: ChatUsage | None = None

    @property
    def text(self) -> str:
        """Shortcut: get the first choice's message content."""
        if self.choices and self.choices[0].message.content:
            return self.choices[0].message.content
        return ""


class StreamChunk(BaseModel):
    """A single SSE chunk from a streaming chat response."""

    id: str = ""
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = ""
    choices: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def delta_content(self) -> str | None:
        """Extract text delta from the first choice."""
        if self.choices:
            delta = self.choices[0].get("delta", {})
            return delta.get("content")
        return None


# ── Projects ─────────────────────────────────────────────────────────────────


class ProjectInfo(BaseModel):
    """Project metadata returned by list/get endpoints."""

    id: str | None = Field(default=None, description="Unique project identifier")
    name: str = Field(description="Human-readable project name")
    namespace: str = Field(default="default", description="Project namespace")
    config: dict[str, Any] = Field(default_factory=dict, description="Project configuration")
    created_at: str | None = None
    updated_at: str | None = None


# ── RAG ──────────────────────────────────────────────────────────────────────


class RAGResult(BaseModel):
    """A single RAG retrieval result."""

    content: str = ""
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str | None = None
    chunk_id: str | None = None


class QueryResponse(BaseModel):
    """Response from a RAG query."""

    results: list[RAGResult] = Field(default_factory=list)
    query: str = ""
    database: str | None = None
    strategy: str | None = None


class DatabaseInfo(BaseModel):
    """RAG database metadata."""

    name: str
    type: str = "ChromaStore"
    document_count: int = 0
    chunk_count: int = 0


# ── Vision ───────────────────────────────────────────────────────────────────


class Detection(BaseModel):
    """A single object detection result."""

    label: str
    confidence: float
    bbox: list[float] = Field(default_factory=list, description="[x1, y1, x2, y2]")


class DetectResponse(BaseModel):
    """Response from vision detection."""

    detections: list[Detection] = Field(default_factory=list)
    model: str = ""
    inference_time_ms: float = 0


class Classification(BaseModel):
    """A single classification result."""

    label: str
    confidence: float


class ClassifyResponse(BaseModel):
    """Response from vision classification."""

    classifications: list[Classification] = Field(default_factory=list)
    model: str = ""
    inference_time_ms: float = 0


# ── Fine-tuning ──────────────────────────────────────────────────────────────


class FineTuneJob(BaseModel):
    """Fine-tuning job status."""

    job_id: str = Field(description="Unique job identifier")
    status: str = Field(description="Job status: queued, running, completed, failed, cancelled")
    model: str = ""
    method: str = ""
    progress: float = 0.0
    metrics: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    output_model: str | None = Field(
        default=None, description="Path to the fine-tuned model when completed"
    )


class FineTuneTemplate(BaseModel):
    """A fine-tuning configuration template."""

    name: str
    method: str
    description: str = ""
    defaults: dict[str, Any] = Field(default_factory=dict)


# ── KV Cache ─────────────────────────────────────────────────────────────────


class CacheStats(BaseModel):
    """KV cache statistics."""

    total_entries: int = 0
    total_size_bytes: int = 0
    hit_rate: float = 0.0
    entries: list[dict[str, Any]] = Field(default_factory=list)


# ── Config generation ────────────────────────────────────────────────────────


def build_project_config(
    name: str,
    namespace: str = "default",
    *,
    model: str = "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
    provider: str = "universal",
    system_prompt: str = "You are a helpful assistant.",
    temperature: float | None = None,
    max_tokens: int | None = None,
    rag_databases: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a llamafarm.yaml-compatible project config dict.

    Args:
        name: Project name.
        namespace: Project namespace.
        model: Model identifier (e.g. "Qwen/Qwen3-8B").
        provider: Model provider (default "universal").
        system_prompt: Default system prompt.
        temperature: Default temperature.
        max_tokens: Default max tokens.
        rag_databases: List of RAG database configs.

    Returns:
        Config dict matching the llamafarm.yaml schema.
    """
    model_config: dict[str, Any] = {
        "name": "default",
        "provider": provider,
        "model": model,
    }
    if temperature is not None:
        model_config["temperature"] = temperature
    if max_tokens is not None:
        model_config["max_tokens"] = max_tokens

    config: dict[str, Any] = {
        "version": "v1",
        "name": name,
        "namespace": namespace,
        "runtime": {"models": [model_config]},
        "prompts": [
            {
                "name": "default",
                "messages": [{"role": "system", "content": system_prompt}],
            }
        ],
    }

    if rag_databases:
        config["rag"] = {"databases": rag_databases}

    return config
