from dataclasses import dataclass
from enum import Enum
from typing import Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    namespace: str | None = None
    project_id: str | None = None

class ChatResponse(BaseModel):
    message: str
    session_id: str
    tool_results: list[dict] | None = None

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