"""Chat data models."""

from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum


class MessageRole(str, Enum):
    """Message roles in chat."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """A single chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class ChatSession(BaseModel):
    """A chat session with history."""
    session_id: str
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def add_message(self, message: ChatMessage):
        """Add a message to the session."""
        self.messages.append(message)
        self.updated_at = datetime.utcnow()
    
    def get_context(self, max_messages: int = 10) -> List[Dict[str, str]]:
        """Get recent messages as context for the model."""
        recent_messages = self.messages[-max_messages:] if max_messages else self.messages
        return [
            {"role": msg.role if isinstance(msg.role, str) else msg.role.value, "content": msg.content}
            for msg in recent_messages
        ]
    
    def get_last_user_message(self) -> str:
        """Get the last user message content."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.USER:
                return msg.content
        return ""
    
    class Config:
        use_enum_values = True


class AgentAction(BaseModel):
    """An action to be performed by an agent."""
    agent_type: str  # "model", "rag", "prompt"
    action: str      # "create", "edit", "validate"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_file: Optional[str] = None
    
    
class ChatResponse(BaseModel):
    """Response from the chat system."""
    message: str
    actions: List[AgentAction] = Field(default_factory=list)
    context_used: List[Dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.8)
    suggestions: List[str] = Field(default_factory=list)
    executed_actions: List[Dict[str, Any]] = Field(default_factory=list)