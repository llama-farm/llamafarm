"""
Session Manager for OpenClaw Lite.

Manages agent sessions with:
- Session lifecycle (create, get, destroy)
- Session-to-session communication
- Session state persistence
- Session spawning for sub-agents
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Awaitable

logger = logging.getLogger(__name__)


class SessionStatus(Enum):
    ACTIVE = "active"
    IDLE = "idle"
    PAUSED = "paused"
    TERMINATED = "terminated"


@dataclass
class SessionMessage:
    """A message within a session."""
    id: str
    timestamp: float
    role: str  # user, assistant, system, tool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "role": self.role,
            "content": self.content,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionMessage":
        return cls(
            id=data["id"],
            timestamp=data["timestamp"],
            role=data["role"],
            content=data["content"],
            metadata=data.get("metadata", {})
        )


@dataclass
class SessionConfig:
    """Configuration for a session."""
    model: Optional[str] = None
    system_prompt: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    tools_enabled: bool = True
    memory_enabled: bool = True
    max_history: int = 100
    channel: Optional[str] = None  # Source channel (telegram, slack, etc.)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """
    An agent session representing a conversation context.
    
    Sessions maintain:
    - Conversation history
    - Configuration
    - State (variables, flags)
    - Parent/child relationships for spawned sub-agents
    """
    
    session_key: str
    agent_id: str
    config: SessionConfig
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    status: SessionStatus = SessionStatus.ACTIVE
    history: List[SessionMessage] = field(default_factory=list)
    state: Dict[str, Any] = field(default_factory=dict)
    parent_session: Optional[str] = None
    child_sessions: List[str] = field(default_factory=list)
    label: Optional[str] = None  # Human-friendly label
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> SessionMessage:
        """Add a message to the session history."""
        msg = SessionMessage(
            id=f"msg-{uuid.uuid4().hex[:8]}",
            timestamp=time.time(),
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.history.append(msg)
        self.last_active = time.time()
        
        # Trim history if needed
        if len(self.history) > self.config.max_history:
            self.history = self.history[-self.config.max_history:]
        
        return msg
    
    def get_history(
        self,
        limit: int = 50,
        include_system: bool = False
    ) -> List[SessionMessage]:
        """Get recent message history."""
        messages = self.history
        if not include_system:
            messages = [m for m in messages if m.role != "system"]
        return messages[-limit:]
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state variable."""
        self.state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state variable."""
        return self.state.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_key": self.session_key,
            "agent_id": self.agent_id,
            "config": {
                "model": self.config.model,
                "system_prompt": self.config.system_prompt,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "tools_enabled": self.config.tools_enabled,
                "memory_enabled": self.config.memory_enabled,
                "max_history": self.config.max_history,
                "channel": self.config.channel,
                "metadata": self.config.metadata
            },
            "created_at": self.created_at,
            "last_active": self.last_active,
            "status": self.status.value,
            "history": [m.to_dict() for m in self.history],
            "state": self.state,
            "parent_session": self.parent_session,
            "child_sessions": self.child_sessions,
            "label": self.label
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        config_data = data.get("config", {})
        config = SessionConfig(
            model=config_data.get("model"),
            system_prompt=config_data.get("system_prompt"),
            max_tokens=config_data.get("max_tokens", 4096),
            temperature=config_data.get("temperature", 0.7),
            tools_enabled=config_data.get("tools_enabled", True),
            memory_enabled=config_data.get("memory_enabled", True),
            max_history=config_data.get("max_history", 100),
            channel=config_data.get("channel"),
            metadata=config_data.get("metadata", {})
        )
        return cls(
            session_key=data["session_key"],
            agent_id=data["agent_id"],
            config=config,
            created_at=data.get("created_at", time.time()),
            last_active=data.get("last_active", time.time()),
            status=SessionStatus(data.get("status", "active")),
            history=[SessionMessage.from_dict(m) for m in data.get("history", [])],
            state=data.get("state", {}),
            parent_session=data.get("parent_session"),
            child_sessions=data.get("child_sessions", []),
            label=data.get("label")
        )


class SessionManager:
    """
    Manages agent sessions across the system.
    
    Features:
    - Session creation and lookup
    - Session persistence
    - Cross-session messaging
    - Sub-agent spawning
    - Session cleanup
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        max_sessions: int = 1000,
        idle_timeout_sec: int = 3600  # 1 hour
    ):
        self.storage_path = storage_path
        self.max_sessions = max_sessions
        self.idle_timeout_sec = idle_timeout_sec
        
        self._sessions: Dict[str, Session] = {}
        self._labels: Dict[str, str] = {}  # label -> session_key
        self._message_handlers: Dict[str, Callable] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        if storage_path:
            self._load()
    
    def create_session(
        self,
        agent_id: str,
        session_key: Optional[str] = None,
        config: Optional[SessionConfig] = None,
        label: Optional[str] = None,
        parent_session: Optional[str] = None
    ) -> Session:
        """Create a new session."""
        session_key = session_key or f"session-{uuid.uuid4().hex[:12]}"
        
        if session_key in self._sessions:
            raise ValueError(f"Session already exists: {session_key}")
        
        if label and label in self._labels:
            raise ValueError(f"Label already in use: {label}")
        
        session = Session(
            session_key=session_key,
            agent_id=agent_id,
            config=config or SessionConfig(),
            parent_session=parent_session,
            label=label
        )
        
        self._sessions[session_key] = session
        if label:
            self._labels[label] = session_key
        
        # Update parent if exists
        if parent_session and parent_session in self._sessions:
            self._sessions[parent_session].child_sessions.append(session_key)
        
        logger.info(f"Created session: {session_key} (agent: {agent_id})")
        self._save()
        
        return session
    
    def get_session(self, session_key: str) -> Optional[Session]:
        """Get a session by key."""
        return self._sessions.get(session_key)
    
    def get_session_by_label(self, label: str) -> Optional[Session]:
        """Get a session by label."""
        session_key = self._labels.get(label)
        if session_key:
            return self._sessions.get(session_key)
        return None
    
    def list_sessions(
        self,
        agent_id: Optional[str] = None,
        status: Optional[SessionStatus] = None,
        limit: int = 100
    ) -> List[Session]:
        """List sessions with optional filters."""
        sessions = list(self._sessions.values())
        
        if agent_id:
            sessions = [s for s in sessions if s.agent_id == agent_id]
        if status:
            sessions = [s for s in sessions if s.status == status]
        
        # Sort by last_active descending
        sessions.sort(key=lambda s: s.last_active, reverse=True)
        
        return sessions[:limit]
    
    async def send_to_session(
        self,
        target_key: str,
        message: str,
        sender_key: Optional[str] = None,
        timeout_sec: float = 30.0
    ) -> Optional[str]:
        """
        Send a message to another session.
        
        Returns the response if a handler is registered.
        """
        session = self._sessions.get(target_key)
        if not session:
            # Try by label
            session = self.get_session_by_label(target_key)
        
        if not session:
            logger.warning(f"Target session not found: {target_key}")
            return None
        
        # Add incoming message
        session.add_message(
            role="user",
            content=message,
            metadata={"sender_session": sender_key} if sender_key else {}
        )
        
        # Check if there's a handler
        handler = self._message_handlers.get(session.session_key)
        if handler:
            try:
                response = await asyncio.wait_for(
                    handler(message, session),
                    timeout=timeout_sec
                )
                if response:
                    session.add_message(role="assistant", content=response)
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Message handler timed out for session: {target_key}")
                return None
            except Exception as e:
                logger.error(f"Message handler error: {e}")
                return None
        
        return None
    
    def register_handler(
        self,
        session_key: str,
        handler: Callable[[str, Session], Awaitable[Optional[str]]]
    ) -> None:
        """Register a message handler for a session."""
        self._message_handlers[session_key] = handler
    
    def unregister_handler(self, session_key: str) -> None:
        """Unregister a message handler."""
        self._message_handlers.pop(session_key, None)
    
    async def spawn_session(
        self,
        parent_key: str,
        agent_id: str,
        task: str,
        config: Optional[SessionConfig] = None,
        label: Optional[str] = None,
        timeout_sec: float = 300.0
    ) -> Session:
        """
        Spawn a sub-agent session.
        
        Creates a new session as a child of the parent,
        sends the initial task, and returns the session.
        """
        parent = self._sessions.get(parent_key)
        if not parent:
            raise ValueError(f"Parent session not found: {parent_key}")
        
        # Create child session
        child = self.create_session(
            agent_id=agent_id,
            config=config,
            label=label,
            parent_session=parent_key
        )
        
        # Send initial task
        child.add_message(
            role="user",
            content=task,
            metadata={"spawned_from": parent_key, "task": True}
        )
        
        logger.info(f"Spawned session {child.session_key} from {parent_key}")
        
        return child
    
    def terminate_session(self, session_key: str) -> bool:
        """Terminate a session."""
        session = self._sessions.get(session_key)
        if not session:
            return False
        
        session.status = SessionStatus.TERMINATED
        self.unregister_handler(session_key)
        
        # Remove from label index
        if session.label:
            self._labels.pop(session.label, None)
        
        # Update parent
        if session.parent_session and session.parent_session in self._sessions:
            parent = self._sessions[session.parent_session]
            if session_key in parent.child_sessions:
                parent.child_sessions.remove(session_key)
        
        logger.info(f"Terminated session: {session_key}")
        self._save()
        
        return True
    
    def delete_session(self, session_key: str) -> bool:
        """Permanently delete a session."""
        self.terminate_session(session_key)
        if session_key in self._sessions:
            del self._sessions[session_key]
            self._save()
            return True
        return False
    
    async def start_cleanup_task(self) -> None:
        """Start background cleanup of idle sessions."""
        if self._cleanup_task:
            return
        
        async def cleanup_loop():
            while True:
                await asyncio.sleep(60)  # Check every minute
                self._cleanup_idle()
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def _cleanup_idle(self) -> int:
        """Clean up idle sessions. Returns count of cleaned."""
        now = time.time()
        cleaned = 0
        
        for session_key, session in list(self._sessions.items()):
            if session.status == SessionStatus.TERMINATED:
                continue
            
            idle_time = now - session.last_active
            if idle_time > self.idle_timeout_sec:
                session.status = SessionStatus.IDLE
                # Don't delete, just mark idle
                cleaned += 1
        
        if cleaned:
            logger.info(f"Marked {cleaned} sessions as idle")
        
        return cleaned
    
    def _save(self) -> None:
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "sessions": {k: v.to_dict() for k, v in self._sessions.items()},
            "labels": self._labels
        }
        self.storage_path.write_text(json.dumps(data, indent=2))
    
    def _load(self) -> None:
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            for key, session_data in data.get("sessions", {}).items():
                self._sessions[key] = Session.from_dict(session_data)
            self._labels = data.get("labels", {})
            logger.info(f"Loaded {len(self._sessions)} sessions from storage")
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager(storage_path: Optional[Path] = None) -> SessionManager:
    """Get or create the global session manager."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(storage_path=storage_path)
    return _session_manager
