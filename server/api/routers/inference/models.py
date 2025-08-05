from typing import Dict, Optional, List, Any
from pydantic import BaseModel
from dataclasses import dataclass
from enum import Enum

class ChatRequest(BaseModel):
    message: str
    namespace: Optional[str] = None
    project_id: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    session_id: str
    tool_results: Optional[List[Dict]] = None

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