"""
Base Skill abstraction for modular agent capabilities.

Skills provide:
- Named capabilities with descriptions
- Tool definitions for function calling
- Execution logic
- Configuration
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, Awaitable

logger = logging.getLogger(__name__)


class ToolParameterType(Enum):
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"


@dataclass
class ToolParameter:
    """A parameter for a tool."""
    name: str
    param_type: ToolParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[str]] = None  # For string enums
    items_type: Optional[ToolParameterType] = None  # For arrays
    
    def to_json_schema(self) -> Dict[str, Any]:
        """Convert to JSON Schema for OpenAI-style tools."""
        schema: Dict[str, Any] = {
            "type": self.param_type.value,
            "description": self.description
        }
        
        if self.enum:
            schema["enum"] = self.enum
        
        if self.param_type == ToolParameterType.ARRAY and self.items_type:
            schema["items"] = {"type": self.items_type.value}
        
        if self.default is not None:
            schema["default"] = self.default
        
        return schema


@dataclass
class Tool:
    """
    A tool that can be called by an agent.
    
    Tools are functions with structured parameters that
    agents can invoke during execution.
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable[..., Awaitable[Any]]] = None
    
    # Execution settings
    timeout_sec: float = 30.0
    requires_confirmation: bool = False
    dangerous: bool = False  # Destructive operations
    
    def to_openai_tool(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        properties = {}
        required = []
        
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given arguments."""
        if not self.handler:
            raise NotImplementedError(f"No handler for tool: {self.name}")
        
        return await self.handler(**kwargs)


@dataclass
class SkillConfig:
    """Configuration for a skill."""
    enabled: bool = True
    priority: int = 5  # 1-10, higher = preferred
    require_confirmation: bool = False
    allowed_agents: Optional[List[str]] = None  # None = all
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillContext:
    """Context passed to skill execution."""
    agent_id: str
    session_key: Optional[str] = None
    channel: Optional[str] = None
    user_message: Optional[str] = None
    conversation_history: List[Dict] = field(default_factory=list)
    memory: Optional[Any] = None  # AgentMemory
    router_service: Optional[Any] = None  # RouterService
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillResult:
    """Result from skill execution."""
    success: bool
    output: str
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    tools_used: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "execution_time_ms": self.execution_time_ms,
            "tools_used": self.tools_used,
            "metadata": self.metadata
        }


class Skill(ABC):
    """
    Abstract base class for agent skills.
    
    A skill provides:
    - A named capability with description
    - Zero or more tools
    - Execution logic
    
    Example:
        class WeatherSkill(Skill):
            name = "weather"
            description = "Get weather forecasts and conditions"
            
            def __init__(self):
                super().__init__()
                self.tools = [
                    Tool(
                        name="get_weather",
                        description="Get current weather for a location",
                        parameters=[
                            ToolParameter(
                                name="location",
                                param_type=ToolParameterType.STRING,
                                description="City name or coordinates"
                            )
                        ],
                        handler=self._get_weather
                    )
                ]
            
            async def _get_weather(self, location: str) -> dict:
                # Implementation
                pass
            
            async def execute(self, context: SkillContext) -> SkillResult:
                # Main skill execution
                pass
    """
    
    name: str = "base_skill"
    description: str = "A base skill"
    version: str = "1.0.0"
    
    def __init__(self, config: Optional[SkillConfig] = None):
        self.config = config or SkillConfig()
        self.tools: List[Tool] = []
        self._initialized = False
    
    @property
    def skill_id(self) -> str:
        """Unique skill identifier."""
        return f"{self.name}@{self.version}"
    
    async def initialize(self) -> bool:
        """
        Initialize the skill.
        
        Called once when the skill is loaded.
        Override to perform setup (API connections, loading data, etc.)
        """
        self._initialized = True
        return True
    
    async def cleanup(self) -> None:
        """
        Clean up skill resources.
        
        Called when the skill is unloaded.
        """
        self._initialized = False
    
    @abstractmethod
    async def execute(self, context: SkillContext) -> SkillResult:
        """
        Execute the skill.
        
        This is the main entry point for skill execution.
        """
        pass
    
    async def call_tool(
        self,
        tool_name: str,
        **kwargs
    ) -> Any:
        """
        Call a tool by name.
        
        Convenience method for executing tools.
        """
        for tool in self.tools:
            if tool.name == tool_name:
                return await tool.execute(**kwargs)
        
        raise ValueError(f"Tool not found: {tool_name}")
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def get_tools_for_openai(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.to_openai_tool() for tool in self.tools]
    
    def matches_intent(self, intent: str) -> float:
        """
        Score how well this skill matches an intent.
        
        Returns a score from 0.0 to 1.0.
        Override for custom matching logic.
        
        Default: simple keyword matching.
        """
        intent_lower = intent.lower()
        name_lower = self.name.lower()
        desc_lower = self.description.lower()
        
        # Exact name match
        if name_lower in intent_lower:
            return 0.9
        
        # Description keyword overlap
        intent_words = set(intent_lower.split())
        desc_words = set(desc_lower.split())
        overlap = len(intent_words & desc_words)
        if overlap > 0:
            return min(0.8, 0.2 * overlap)
        
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize skill metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "skill_id": self.skill_id,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "dangerous": t.dangerous
                }
                for t in self.tools
            ],
            "config": {
                "enabled": self.config.enabled,
                "priority": self.config.priority
            }
        }


# --- Built-in Skills ---

class EchoSkill(Skill):
    """Simple echo skill for testing."""
    
    name = "echo"
    description = "Echo back messages for testing"
    version = "1.0.0"
    
    def __init__(self, config: Optional[SkillConfig] = None):
        super().__init__(config)
        self.tools = [
            Tool(
                name="echo",
                description="Echo back a message",
                parameters=[
                    ToolParameter(
                        name="message",
                        param_type=ToolParameterType.STRING,
                        description="Message to echo"
                    )
                ],
                handler=self._echo
            )
        ]
    
    async def _echo(self, message: str) -> str:
        return f"Echo: {message}"
    
    async def execute(self, context: SkillContext) -> SkillResult:
        start = time.time()
        
        message = context.user_message or "No message provided"
        output = await self._echo(message)
        
        return SkillResult(
            success=True,
            output=output,
            execution_time_ms=(time.time() - start) * 1000,
            tools_used=["echo"]
        )


class MemorySkill(Skill):
    """Skill for memory operations."""
    
    name = "memory"
    description = "Store and recall information from memory"
    version = "1.0.0"
    
    def __init__(self, config: Optional[SkillConfig] = None):
        super().__init__(config)
        self.tools = [
            Tool(
                name="remember",
                description="Store something in long-term memory",
                parameters=[
                    ToolParameter(
                        name="content",
                        param_type=ToolParameterType.STRING,
                        description="Content to remember"
                    ),
                    ToolParameter(
                        name="key",
                        param_type=ToolParameterType.STRING,
                        description="Optional key for the memory",
                        required=False
                    )
                ],
                handler=self._remember
            ),
            Tool(
                name="recall",
                description="Search memory for relevant information",
                parameters=[
                    ToolParameter(
                        name="query",
                        param_type=ToolParameterType.STRING,
                        description="What to search for"
                    )
                ],
                handler=self._recall
            ),
            Tool(
                name="get_fact",
                description="Get a specific fact by key",
                parameters=[
                    ToolParameter(
                        name="key",
                        param_type=ToolParameterType.STRING,
                        description="Fact key to retrieve"
                    )
                ],
                handler=self._get_fact
            )
        ]
        self._context: Optional[SkillContext] = None
    
    async def _remember(self, content: str, key: Optional[str] = None) -> str:
        if not self._context or not self._context.memory:
            return "No memory available"
        
        if key:
            self._context.memory.set_fact(key, content)
            return f"Stored fact '{key}'"
        else:
            self._context.memory.memorize(content)
            return "Stored in long-term memory"
    
    async def _recall(self, query: str) -> str:
        if not self._context or not self._context.memory:
            return "No memory available"
        
        memories = self._context.memory.get_relevant_memories(query, max_results=5)
        if not memories:
            return "No relevant memories found"
        
        return "\n".join([f"- {m.content}" for m in memories])
    
    async def _get_fact(self, key: str) -> str:
        if not self._context or not self._context.memory:
            return "No memory available"
        
        fact = self._context.memory.get_fact(key)
        return fact if fact else f"No fact found for key: {key}"
    
    async def execute(self, context: SkillContext) -> SkillResult:
        self._context = context
        start = time.time()
        
        # This skill is primarily used via tools
        return SkillResult(
            success=True,
            output="Memory skill ready. Use tools: remember, recall, get_fact",
            execution_time_ms=(time.time() - start) * 1000
        )
